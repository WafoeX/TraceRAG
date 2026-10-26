"""DeepEval custom-metric execution over explicit main-assistant AI judgments.

No built-in judge metric, cloud dashboard or extra model request. Use a separate
pinned environment; actual human/independent validation is NOT implied.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

from scripts.complete_scifact_six_metrics import (
    ROOT,
    digest,
    parse_object,
    require,
    rows,
    write_json,
)

FLAGS = {
    "DEEPEVAL_DISABLE_DOTENV": "1",
    "DEEPEVAL_DISABLE_LEGACY_KEYFILE": "1",
    "DEEPEVAL_TELEMETRY_OPT_OUT": "1",
    "DEEPEVAL_TELEMETRY_ENABLED": "0",
    "DEEPEVAL_UPDATE_WARNING_OPT_IN": "0",
    "DEEPEVAL_FILE_SYSTEM": "READ_ONLY",
    "CONFIDENT_OPEN_BROWSER": "false",
    "OTEL_SDK_DISABLED": "true",
}


def prepare(pack: Path, annotation_paths: list[Path]) -> tuple[list[dict], list[dict], dict]:
    manifest = parse_object((pack / "manifest.json").read_text())
    for name, expected in manifest["outputs_sha256"].items():
        require(digest((pack / name).read_bytes()) == expected, "Frozen pack output changed")
    pending = list(rows(pack / "contexts.pending.jsonl"))
    answers = {(r["query_id"], r["arm"]): r for r in rows(pack / "answers.pending.jsonl")}
    all_reviewed, cases, seen_queries = [], [], set()
    inputs = {
        str((pack / "manifest.json").relative_to(ROOT)): digest(
            (pack / "manifest.json").read_bytes()
        )
    }
    for path in annotation_paths:
        annotation = parse_object(path.read_text())
        require(annotation["reviewer"] == "main_assistant_AI", "AI provenance required")
        require(
            annotation["human_verified"] is False
            and annotation["independent_blind_review"] is False,
            "Do not invent human or independent validation",
        )
        query_ids = list(dict.fromkeys(r["query_id"] for r in annotation["context_reviews"]))
        require(not (set(query_ids) & seen_queries), "Overlapping batches would double count")
        seen_queries.update(query_ids)
        require(
            query_ids
            == [
                r["query_id"]
                for r in sorted(
                    {r["query_id"]: r for r in pending if r["query_id"] in query_ids}.values(),
                    key=lambda r: r["query_position"],
                )
            ],
            "Unfrozen query order",
        )
        expected = {
            (r["query_id"], r["chunk_id"]): r for r in pending if r["query_id"] in query_ids
        }
        reviewed = {}
        for label in annotation["context_reviews"]:
            matching = [
                r
                for r in expected.values()
                if r["query_id"] == label["query_id"]
                and r["document_id"] == label["document_id"]
                and ("chunk_id" not in label or r["chunk_id"] == label["chunk_id"])
            ]
            require(len(matching) == 1, "Ambiguous document: provide exact chunk_id")
            row = matching[0]
            key = (row["query_id"], row["chunk_id"])
            require(key not in reviewed, "Duplicate query-chunk annotation")
            grade = label["grade"]
            require(grade is None or (type(grade) is int and grade in (0, 1, 2)), "Invalid grade")
            require(
                bool(label["reason"].strip()) and bool(label["quote"]), "Reason and quote required"
            )
            start = row["text"].find(label["quote"])
            require(start >= 0, "Quote does not occur in actual chunk")
            reviewed[key] = {
                **row,
                "review_status": "ai_reviewed",
                "grade": grade,
                "reason": label["reason"],
                "reviewer": annotation["reviewer"],
                "evidence_spans": [
                    {"start": start, "end": start + len(label["quote"]), "quote": label["quote"]}
                ],
                "annotation_sha256": digest(path.read_bytes()),
            }
        require(set(reviewed) == set(expected), "All actual batch chunks must be reviewed")
        sufficiency = {}
        for label in annotation["sufficiency_reviews"]:
            require(
                label["query_id"] in query_ids and bool(label["reason"].strip()),
                "Invalid reason/query",
            )
            require(
                label["label"] in ("sufficient", "insufficient", "uncertain"), "Invalid sufficiency"
            )
            for arm in label["arms"]:
                key = (label["query_id"], arm)
                require(arm in ("A", "B") and key not in sufficiency, "Duplicate/invalid arm")
                sufficiency[key] = label
        require(
            set(sufficiency) == {(q, a) for q in query_ids for a in ("A", "B")},
            "Missing sufficiency",
        )
        for q in query_ids:
            for arm in ("A", "B"):
                group = [
                    r
                    for r in reviewed.values()
                    if r["query_id"] == q and any(p["arm"] == arm for p in r["positions"])
                ]
                group.sort(
                    key=lambda r: next(p["source"] for p in r["positions"] if p["arm"] == arm)
                )
                require(
                    len(group) == 4, "Four actual chunks required; do not deduplicate documents"
                )
                answer = answers[q, arm]
                require(
                    [r["chunk_id"] for r in group] == [c["chunk_id"] for c in answer["contexts"]],
                    "Context order mismatch",
                )
                cases.append(
                    {
                        "query_id": q,
                        "arm": arm,
                        "input": answer["query"],
                        "actual_output": answer["answer"],
                        "retrieval_context": [r["text"] for r in group],
                        "metadata": {
                            "query_id": q,
                            "arm": arm,
                            "grades": [r["grade"] for r in group],
                            "context_hashes": [r["text_sha256"] for r in group],
                            "sufficiency": sufficiency[q, arm]["label"],
                            "sufficiency_reason": sufficiency[q, arm]["reason"],
                            "source_answer_row_sha256": answer["source_answer_row_sha256"],
                        },
                    }
                )
        all_reviewed.extend(reviewed.values())
        inputs[str(path.relative_to(ROOT))] = digest(path.read_bytes())
    positions = {r["query_id"]: r["query_position"] for r in pending}
    require(
        sorted(positions[q] for q in seen_queries) == list(range(len(seen_queries))),
        "Coverage must be an uninterrupted original prefix, not selected good queries",
    )
    return cases, all_reviewed, inputs


def execute(cases: list[dict]) -> list[dict]:
    # Imports happen only after disabling dotenv/telemetry and installing the audit guard.
    from deepeval.metrics import BaseMetric
    from deepeval.test_case import LLMTestCase

    class SavedAIContextMetric(BaseMetric):
        def __init__(self, kind: str, bound: str):
            self.kind, self.bound = kind, bound
            self.threshold = None  # Descriptive scores, not an invented acceptance threshold.
            self.async_mode = False
            self.verbose_mode = False
            self.evaluation_model = "main_assistant_AI_saved_annotations"
            self.evaluation_cost = None  # Main-conversation AI cost is not in the app ledger.
            self.score = None
            self.success = None
            self.error = None
            self.reason = None

        def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
            metadata = test_case.metadata
            if not isinstance(metadata, dict):
                raise ValueError("Missing explicit AI judgments")
            context = test_case.retrieval_context
            if not isinstance(context, list) or len(context) != 4:
                raise ValueError("Need actual four chunks")
            hashes = []
            for text in context:
                if not isinstance(text, str):
                    raise ValueError("Plain source text required")
                hashes.append(digest(text.encode()))
            require(
                hashes == metadata["context_hashes"],
                "DeepEval case context hash mismatch",
            )
            grades = metadata["grades"]
            unknown = sum(g is None for g in grades)
            if self.kind == "sufficiency":
                label = metadata["sufficiency"]
                value = 1.0 if label == "sufficient" else 0.0
                if self.bound == "upper" and label == "uncertain":
                    value = 1.0
            else:
                if self.kind == "useful_precision":
                    value = sum(g == 2 for g in grades) / 4
                    increment = unknown / 4
                elif self.kind == "related_precision":
                    value = sum(g is not None and g >= 1 for g in grades) / 4
                    increment = unknown / 4
                else:
                    value = sum(g for g in grades if g is not None) / 8
                    increment = unknown / 4
                if self.bound == "upper":
                    value += increment
            self.score = value
            self.reason = (
                "Replay explicit, non-blind main-assistant AI grades; not a built-in judge score"
            )
            self.success = None
            return value

        async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
            return self.measure(test_case)

    results = []
    for case in cases:
        test_case = LLMTestCase(
            input=case["input"],
            actual_output=case["actual_output"],
            retrieval_context=case["retrieval_context"],
            metadata=case["metadata"],
        )
        values = {}
        for kind in ("related_precision", "useful_precision", "normalized_grade", "sufficiency"):
            for bound in ("lower", "upper"):
                metric = SavedAIContextMetric(kind, bound)
                value = metric.measure(test_case)
                require(
                    metric.score == value and metric.success is None, "Metric contract mismatch"
                )
                values[f"{kind}_{bound}"] = value
        results.append(
            {
                "query_id": case["query_id"],
                "arm": case["arm"],
                "scores": values,
                "context_sufficiency": case["metadata"]["sufficiency"],
                "context_sufficiency_reason": case["metadata"]["sufficiency_reason"],
                "threshold": None,
                "success": None,
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    require(
        output.is_relative_to(ROOT / "runs") and not output.exists(), "Fresh run output required"
    )
    require(importlib.metadata.version("deepeval") == "4.3.1", "Use pinned isolated DeepEval 4.3.1")
    cases, reviewed, inputs = prepare(args.pack.resolve(), [p.resolve() for p in args.annotations])
    inputs[str(Path(__file__).relative_to(ROOT))] = digest(Path(__file__).read_bytes())
    flags = dict(FLAGS)
    blocked = []
    local_probes = []

    def audit(event, audit_args):
        # urllib3 checks local IPv6 support during import; this is not an outbound connection.
        if event == "socket.bind":
            caller = sys._getframe(1)
            if (
                audit_args[1] == ("::1", 0)
                and caller.f_code.co_name == "_has_ipv6"
                and caller.f_globals.get("__name__") == "urllib3.util.connection"
            ):
                local_probes.append("urllib3 IPv6 loopback ephemeral bind")
                return
        if event in {
            "socket.connect",
            "socket.connect_ex",
            "socket.getaddrinfo",
            "socket.bind",
            "sqlite3.connect",
        }:
            blocked.append(f"{event}; family={getattr(audit_args[0], 'family', None)}")
            raise RuntimeError("No network/database allowed in evaluation")
        if event == "open" and isinstance(audit_args[0], str | bytes):
            path = str(audit_args[0])
            if Path(path).name.startswith(".env") or "/splits/test/" in path:
                blocked.append("secret_or_test_open")
                raise RuntimeError("No secrets/test files allowed")

    sys.addaudithook(audit)
    original_cwd = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="budgetrag-deepeval-sandbox-") as sandbox:
        os.environ.update(flags)
        os.environ.update(HOME=sandbox, ENV_DIR_PATH=sandbox, XDG_CONFIG_HOME=sandbox)
        os.chdir(sandbox)
        try:
            results = execute(cases)
        finally:
            os.chdir(original_cwd)
        require(not blocked, f"Framework forbidden runtime events: {dict(Counter(blocked))}")
    require(
        not any(name == "app" or name.startswith("app.") for name in sys.modules), "App imported"
    )
    summaries = {}
    for arm in ("A", "B"):
        group = [r for r in results if r["arm"] == arm]
        counts = Counter(r["context_sufficiency"] for r in group)
        summaries[arm] = {
            "queries": len(group),
            "metrics": {
                name: math.fsum(r["scores"][name] for r in group) / len(group)
                for name in group[0]["scores"]
            },
            "sufficiency_counts": dict(counts),
        }
    require(all(digest((ROOT / p).read_bytes()) == h for p, h in inputs.items()), "Input changed")
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "partial_context_evaluation",
        "framework": "deepeval",
        "version": "4.3.1",
        "execution": "LLMTestCase + custom BaseMetric.measure; no built-in judge/dashboard",
        "query_count": len(cases) // 2,
        "test_case_count": len(cases),
        "metric_measures": len(cases) * 8,
        "unique_query_chunks_reviewed": len(reviewed),
        "context_positions": len(cases) * 4,
        "summaries": summaries,
        "answer_reviews_completed_under_new_protocol": 0,
        "full100_semantic_evaluation_complete": False,
        "judge": "main_assistant_AI_nonblind",
        "human_verified": False,
        "independent_blind_review": False,
        "new_judge_API_calls": 0,
        "main_session_AI_cost_cny": None,
        "blocked_network_or_secret_events": blocked,
        "allowed_local_import_probes": local_probes,
        "environment_flags": flags,
        "inputs_sha256": inputs,
        "not_a_standard_deepeval_triad_score": True,
    }
    write_json(output / "report.json", report)
    for filename, data in (
        ("test_cases.jsonl", cases),
        ("metrics.jsonl", results),
        ("contexts.reviewed.jsonl", reviewed),
    ):
        (output / filename).write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in data)
        )
    (output / "adapter_source.py").write_bytes(Path(__file__).read_bytes())
    write_json(
        output / "verification.json",
        {
            "status": "passed",
            "inputs_sha256": inputs,
            "outputs_sha256": {
                p.name: digest(p.read_bytes()) for p in output.iterdir() if p.is_file()
            },
            "all_actual_context_hashes_verified_in_framework": True,
            "all_annotation_quotes_and_chunk_identities_verified": True,
            "forbidden_runtime_events": blocked,
            "allowed_local_import_probes": local_probes,
        },
    )
    print(
        json.dumps({"cases": len(cases), "unique_contexts": len(reviewed), "summaries": summaries})
    )


if __name__ == "__main__":
    main()
