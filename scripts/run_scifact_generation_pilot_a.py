"""Execute ONLY frozen pilot A (24 generation calls). Plan-only unless --execute.

No B executor, retries, regeneration, SQL, retrieval, rerank, or API judge.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from datetime import UTC, datetime
from pathlib import Path

from scripts import evaluate_scifact_answers as ev
from scripts import prepare_scifact_generation_pilot as prep
from scripts.audit_scifact_ai_reviews import expanded_citations
from scripts.prepare_scifact import ROOT, hash_file, write_json, write_rows
from scripts.render_scifact_answers import usage_summary
from scripts.run_scifact_baseline import append, latency_summary

PHASE = "A_diagnostic_regression"
IDS = ["0", "14", "24", "35", "37", "44", "61", "81", "96", "98", "109", "119"]


def utc_now():
    return datetime.now(UTC).isoformat()


def load_inputs(plan):
    manifest = ev.read_object(plan / "manifest.json")
    if (
        manifest["generation_config"] != ev.CONFIG
        or manifest["stages"]["A"] != {"query_ids": IDS, "calls": 24}
        or manifest["only_experimental_factor"] != "system_prompt_suffix"
    ):
        raise ValueError("Not the frozen 12-query pilot A")
    for name, sha in manifest["source_hashes"].items():
        if hash_file(ROOT / name) != sha:
            raise ValueError(f"Frozen source changed: {name}")
    for name, sha in manifest["output_hashes"].items():
        if hash_file(plan / name) != sha:
            raise ValueError(f"Frozen preview artifact changed: {name}")
    ev.verify_pack(prep.PACK)
    cases = ev.unique(ev.read_rows(plan / "cases.jsonl"), "query_id")
    if set(cases) != set(IDS):
        raise ValueError("Diagnostic casebook mismatch")
    # Read the frozen file but select ONLY A; no B question/answer is displayed or executed.
    rows = [r for r in ev.read_rows(plan / "requests.plan.jsonl") if r["phase"] == PHASE]
    expected = [
        (q, v)
        for i, q in enumerate(IDS)
        for v in (prep.VARIANTS if i % 2 == 0 else tuple(reversed(prep.VARIANTS)))
    ]
    if [(r["query_id"], r["variant"]) for r in rows] != expected:
        raise ValueError("Pilot A request count/order/variants changed")
    suffix = (plan / "candidate_prompt.txt").read_text().strip()
    if suffix != prep.PROMPT.read_text().strip():
        raise ValueError("Frozen candidate snapshot changed")
    for row in rows:
        c = cases[row["query_id"]]
        context = c["historical_answers"]["B"]["contexts"]
        case = {"query_id": c["query_id"], "claim": c["claim"], "contexts": {"B": context}}
        if (
            ev.digest(context) != row["context_sha256"]
            or prep.make_payload(case, row["variant"], suffix) != row["request"]
        ):
            raise ValueError("Request differs from frozen single-factor payload")
    return manifest, cases, rows


def collect_one(item, contexts, client, pricing):
    start = time.perf_counter()
    row = {
        "query_id": item["query_id"],
        "variant": item["variant"],
        "phase": PHASE,
        "request": copy.deepcopy(item["request"]),
        "contexts": contexts,
        "started_at_utc": utc_now(),
        "response": None,
        "answer": "",
        "usage": None,
        "finish_reason": None,
        "status": "error",
        "error_type": None,
        "estimated_cost_cny": None,
        "actual_cost_cny": None,
    }
    try:
        response = client.chat.completions.create(**item["request"])
        row["response"] = response.model_dump(mode="json")
        row["usage"] = response.usage.model_dump(mode="json") if response.usage else None
        row["answer"] = response.choices[0].message.content or ""
        row["finish_reason"] = response.choices[0].finish_reason
        row["status"] = (
            "ok" if ev.text(row["answer"]) and row["finish_reason"] == "stop" else "invalid"
        )
        row["estimated_cost_cny"] = ev.estimate_cost(row["usage"], pricing)
    except Exception as exc:
        row["error_type"] = type(exc).__name__  # Never log credential-bearing exception text.
    row["finished_at_utc"] = utc_now()
    row["generation_ms"] = (time.perf_counter() - start) * 1000
    row["citations"] = expanded_citations(row["answer"])
    return row


def summarize(rows):
    summaries = {}
    for variant in prep.VARIANTS:
        rs = [r for r in rows if r["variant"] == variant]
        costs = [r["estimated_cost_cny"] for r in rs]
        summaries[variant] = {
            "answers": len(rs),
            "ok": sum(r["status"] == "ok" for r in rs),
            "usage": usage_summary(rs),
            "generation_latency": latency_summary([r["generation_ms"] for r in rs]),
            "known_only_estimated_cost_cny": sum(c for c in costs if c is not None),
            "unknown_cost_calls": sum(c is None for c in costs),
            "estimated_total_cost_cny": sum(costs) if all(c is not None for c in costs) else None,
            "actual_cost_cny": None,
        }
    return summaries


def run(plan, output, execute=False, pricing_path=None):
    manifest, cases, items = load_inputs(plan)
    pricing = ev.load_pricing(pricing_path)
    if output.exists():
        raise ValueError("Use a fresh directory; no overwrite, resume or retry")
    client = None
    if execute:
        from openai import OpenAI

        from app.config import get_settings

        settings = get_settings()
        if (
            settings.deepseek_base_url != ev.CONFIG["base_url"]
            or settings.openai_chat_model != ev.CONFIG["model"]
            or not settings.deepseek_api_key.strip()
        ):
            raise ValueError("Frozen official DeepSeek endpoint/model/key required")
        client = OpenAI(
            api_key=settings.deepseek_api_key,
            base_url=ev.CONFIG["base_url"],
            max_retries=0,
            timeout=60,
        )
    ev.fresh(output)
    frozen = {
        "phase": PHASE,
        "planned_calls": 24,
        "B_calls_authorized_by_this_runner": 0,
        "source_plan_manifest_sha256": hash_file(plan / "manifest.json"),
        "source_request_plan_sha256": hash_file(plan / "requests.plan.jsonl"),
        "config": manifest["generation_config"],
        "pricing": pricing,
        "collector_sha256": hash_file(Path(__file__)),
    }
    write_json(output / "run_config.json", frozen)
    write_rows(output / "requests.A.jsonl", items)
    (output / "collector_source.py").write_bytes(Path(__file__).read_bytes())
    (output / "protocol.md").write_bytes((plan / "protocol.md").read_bytes())
    write_json(
        output / "status.json",
        {
            "status": "running" if execute else "plan_only",
            "started": 0,
            "finished": 0,
            "planned": 24,
            "B_calls": 0,
        },
    )
    if client is None:
        return frozen
    rows = []
    try:
        for index, item in enumerate(items):
            request_id = f"A-{item['query_id']}-{item['variant']}"
            append(
                output / "events.jsonl",
                {
                    "event": "started",
                    "request_id": request_id,
                    "query_id": item["query_id"],
                    "variant": item["variant"],
                    "at_utc": utc_now(),
                    "request": item["request"],
                },
            )
            write_json(
                output / "status.json",
                {
                    "status": "running",
                    "started": index + 1,
                    "finished": index,
                    "planned": 24,
                    "B_calls": 0,
                },
            )
            contexts = cases[item["query_id"]]["historical_answers"]["B"]["contexts"]
            row = {"request_id": request_id, **collect_one(item, contexts, client, pricing)}
            append(output / "answers.jsonl", row)
            append(
                output / "events.jsonl",
                {
                    "event": "finished",
                    "request_id": request_id,
                    "status": row["status"],
                    "at_utc": utc_now(),
                },
            )
            rows.append(row)
            write_json(
                output / "status.json",
                {
                    "status": "running",
                    "started": index + 1,
                    "finished": len(rows),
                    "planned": 24,
                    "B_calls": 0,
                },
            )
    finally:
        client.close()
    write_json(
        output / "generation_summary.json",
        {
            "status": "awaiting_main_assistant_AI_review",
            "new_generation_calls": len(rows),
            "B_calls": 0,
            "judge_API_calls": 0,
            "summaries": summarize(rows),
            "main_session_review_cost_cny": None,
        },
    )
    write_json(
        output / "run_manifest.json",
        {
            "files": {
                n: hash_file(output / n)
                for n in (
                    "run_config.json",
                    "requests.A.jsonl",
                    "collector_source.py",
                    "protocol.md",
                    "answers.jsonl",
                    "events.jsonl",
                    "generation_summary.json",
                )
            }
        },
    )
    write_json(
        output / "status.json",
        {
            "status": "awaiting_main_assistant_AI_review",
            "started": len(rows),
            "finished": len(rows),
            "planned": 24,
            "B_calls": 0,
        },
    )
    return frozen


def verify_run(plan, output):
    _, cases, items = load_inputs(plan)
    config = ev.read_object(output / "run_config.json")
    manifest = ev.read_object(output / "run_manifest.json")
    for name, sha in manifest["files"].items():
        if hash_file(output / name) != sha:
            raise ValueError("Run artifact changed")
    if (
        config["source_plan_manifest_sha256"] != hash_file(plan / "manifest.json")
        or config["source_request_plan_sha256"] != hash_file(plan / "requests.plan.jsonl")
        or hash_file(output / "collector_source.py") != config["collector_sha256"]
        or (output / "protocol.md").read_bytes() != (plan / "protocol.md").read_bytes()
        or config["config"] != ev.CONFIG
        or config["planned_calls"] != 24
        or config["B_calls_authorized_by_this_runner"] != 0
        or ev.read_rows(output / "requests.A.jsonl") != items
    ):
        raise ValueError("Run not bound to frozen A-only configuration")
    rows = ev.read_rows(output / "answers.jsonl")
    events = ev.read_rows(output / "events.jsonl")
    if len(rows) != 24 or len(events) != 48:
        raise ValueError("Incomplete pilot A; never implicitly retry")
    for row, item, start, finish in zip(rows, items, events[::2], events[1::2], strict=True):
        if (
            row["query_id"] != item["query_id"]
            or row["variant"] != item["variant"]
            or row["request"] != item["request"]
            or row["phase"] != PHASE
            or row["contexts"] != cases[row["query_id"]]["historical_answers"]["B"]["contexts"]
            or start["event"] != "started"
            or finish["event"] != "finished"
            or row["request_id"] != f"A-{item['query_id']}-{item['variant']}"
            or start["request_id"] != row["request_id"]
            or finish["request_id"] != row["request_id"]
            or start["query_id"] != item["query_id"]
            or start["variant"] != item["variant"]
            or start["request"] != item["request"]
            or finish["status"] != row["status"]
        ):
            raise ValueError("Raw A request/context/event mismatch")
        legacy = {**row, "citations": ev.citations(row["answer"])}
        ev.replay_response(legacy)
        if row["citations"] != expanded_citations(row["answer"]) or row[
            "estimated_cost_cny"
        ] != ev.estimate_cost(row["usage"], config["pricing"]):
            raise ValueError("Usage/citation/cost replay mismatch")
    summary = ev.read_object(output / "generation_summary.json")
    if summary["summaries"] != summarize(rows):
        raise ValueError("Summary replay mismatch")
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--pricing", type=Path)
    p.add_argument("--verify-only", action="store_true")
    args = p.parse_args(argv)
    if args.verify_only:
        if args.execute:
            raise ValueError("Verification must not execute")
        verify_run(args.plan, args.output)
    else:
        run(args.plan, args.output, args.execute, args.pricing)
    print(json.dumps({"status": "ok", "phase": "A_only", "execute": args.execute}))


if __name__ == "__main__":
    main()
