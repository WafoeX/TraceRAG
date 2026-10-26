"""Frozen SciFact answer replay: prepare -> human seal -> generate -> human score.

No network on prepare/seal/score or generate without --execute. No SQL, retrieval,
reranking or automatic judge. Never infer answer correctness from qrels.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import random
import re
import statistics
import time
from decimal import Decimal
from pathlib import Path

from scripts.evaluate_scifact import DEFAULT_PREPARED, load_prepared, read_object, read_rows
from scripts.prepare_scifact import ROOT, hash_file, write_json, write_rows
from scripts.run_scifact_baseline import append, latency_summary

PROTOCOL = ROOT / "docs/refactor/SCIFACT_ANSWER_PROTOCOL.md"
UPSTREAM = ROOT / "runs/scifact-answer-upstream-v1"
CHUNKS = ROOT / "runs/scifact-evaluation-plan-v1/chunks.jsonl"
OLD = ROOT / "runs/scifact-nonthinking-development807-v1/query_results.jsonl"
NEW = ROOT / "runs/scifact-fallback-development807-v1/query_results.jsonl"
ARMS = {"A": "baseline", "B": "relevance_v1"}
LABELS = {"SUPPORT", "REFUTE", "INSUFFICIENT"}
CITATION = re.compile(r"\[(\d+)\]")
QUESTION = (
    "Assess this scientific claim using only the provided context. State whether it is "
    "supported, refuted, or there is insufficient evidence; explain the essential "
    "evidence and cite sources inline. Claim: {}"
)
CONFIG = {
    "base_url": "https://api.deepseek.com/v1",
    "model": "deepseek-flash",
    "temperature": 0.2,
    "max_tokens": 2048,
    "stream": False,
    "sdk_retries": 0,
    "timeout_seconds": 60,
    "top_k": 4,
}


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def text(value):
    return isinstance(value, str) and bool(value.strip())


def normalize(value):
    return " ".join(value.split())


def unique(rows, key):
    result = {r[key]: r for r in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate {key}")
    return result


def fresh(path):
    if path.exists():
        raise ValueError("Use a fresh output directory; no overwrite/resume/retry")
    path.mkdir(parents=True)


def approval(review):
    if review.get("status") != "approved" or any(
        not text(review.get(k)) for k in ("reviewer", "reviewed_at", "notes")
    ):
        raise ValueError("Real human approval, reviewer, time and notes required")


def pending():
    return {"status": "pending", "reviewer": "", "reviewed_at": "", "notes": ""}


def context_messages(case, arm):
    # Reuse application prompt/context formatter, not a second subtly different prompt.
    from app.services.generation import _messages
    from app.services.retrieval import RetrievedChunk

    chunks = [
        RetrievedChunk(i, i, c["title"], c["chunk_index"], c["text"], 0.0)
        for i, c in enumerate(case["contexts"][arm], 1)
    ]
    return _messages(QUESTION.format(case["claim"]), chunks)


def render_cases(path, cases, gold):
    """Escaped local review sheet; export edits without any network or auto-approval."""
    sections = []
    by_id = unique(gold, "query_id")
    for case in cases:
        g = by_id[case["query_id"]]
        parts = [
            f"<h2>{html.escape(case['query_id'])}: {html.escape(case['claim'])}</h2>",
            "<p>待人工核验 / NOT HUMAN VERIFIED</p>",
            "<h3>标签与要点草稿（可编辑此题JSON，导出保存）</h3>"
            + '<textarea class="gold" rows="24" spellcheck="false">'
            + html.escape(json.dumps(g, ensure_ascii=False, indent=2))
            + "</textarea>",
        ]
        for arm, contexts in case["contexts"].items():
            parts.append(f"<details><summary>匿名上下文 {arm}（4 chunks）</summary>")
            for i, c in enumerate(contexts, 1):
                parts.append(
                    f"<h4>[{i}] {html.escape(c['title'])}</h4><pre>"
                    + html.escape(c["text"])
                    + "</pre>"
                )
            parts.append("</details>")
        for doc in case["reference_documents"]:
            parts.append(
                "<details><summary>参考全文 "
                + html.escape(doc["document_id"])
                + "</summary><pre>"
                + html.escape(doc["title"] + "\n" + doc["text"])
                + "</pre></details>"
            )
        sections.append("<section>" + "\n".join(parts) + "</section>")
    path.write_text(
        "<!doctype html><meta charset='utf-8'><title>SciFact 人工核验包</title>"
        "<style>body{max-width:1100px;margin:2em auto;font:16px system-ui}"
        "pre{white-space:pre-wrap;overflow-wrap:anywhere}section{border-top:2px solid #888;"
        "margin:2em 0}summary{cursor:pointer;padding:1em;background:#eee}"
        "textarea{box-sizing:border-box;width:100%;font:14px monospace}"
        "header{position:sticky;top:0;background:white;padding:1em;border:1px solid #888}</style>"
        "<h1>固定100题人工金标核验包</h1><p>修改下方JSON后点击导出；刷新/关闭前务必保存，"
        "本页不自动保存、不联网、不自动批准。也可直接编辑相邻 gold.draft.jsonl。"
        "不要把pending改成approved而未实际核验。上下文A/B不显示重排版本。</p>"
        '<header><button id="export">导出全部金标 JSONL</button> '
        '<input id="find" placeholder="查找题号或断言"><span id="notice"></span></header>'
        + "\n".join(sections)
        + """<script>
        document.getElementById('find').oninput = function() {
          const q = this.value.toLowerCase();
          document.querySelectorAll('section').forEach(s => {
            s.hidden = !s.querySelector('h2').textContent.toLowerCase().includes(q);
          });
        };
        document.getElementById('export').onclick = function() {
          try {
            const rows = Array.from(document.querySelectorAll('textarea.gold'), t =>
              JSON.stringify(JSON.parse(t.value)));
            const blob = new Blob([rows.join('\\n') + '\\n'], {type:'application/x-ndjson'});
            const url = URL.createObjectURL(blob), a = document.createElement('a');
            a.href = url; a.download = 'gold.reviewed.jsonl'; a.click();
            URL.revokeObjectURL(url);
            document.getElementById('notice').textContent = '已导出 ' + rows.length +
              ' 题。把下载文件传给 seal --gold；未核验条目须保持pending。';
          } catch(e) {
            document.getElementById('notice').textContent = 'JSON格式错误: ' + e.message;
          }
        };
        </script>""",
        encoding="utf-8",
    )


def prepare(output):
    data = load_prepared(DEFAULT_PREPARED, "development")
    selected = list(data["queries"])[:100]
    claims = unique(read_rows(UPSTREAM / "claims_train.jsonl"), "id")
    corpus = unique(read_rows(UPSTREAM / "corpus.jsonl"), "doc_id")
    source = read_object(UPSTREAM / "source.json")
    for name, sha in source["files"].items():
        if hash_file(UPSTREAM / name) != sha:
            raise ValueError("Upstream source hash mismatch")
    # Whitespace-only differences are expected; don't silently accept changed science.
    if set(map(str, corpus)) != set(data["corpus"]):
        raise ValueError("Upstream/BEIR corpus ID mismatch")
    for did, doc in data["corpus"].items():
        original = corpus[int(did)]
        if (
            normalize(" ".join(original["abstract"])) != normalize(doc["text"])
            or original["title"] != doc["title"]
        ):
            raise ValueError("Upstream/BEIR abstract mismatch")
    chunks = unique(read_rows(CHUNKS), "chunk_id")
    chunk_status = read_object(CHUNKS.parent / "status.json")
    if (
        hash_file(CHUNKS) != chunk_status["manifest"]["chunks.jsonl"]["sha256"]
        or chunk_status["source_preparation_report_sha256"] != data["report_sha256"]
    ):
        raise ValueError("Frozen chunk/preparation hash mismatch")
    rankings = {"A": unique(read_rows(OLD), "query_id"), "B": unique(read_rows(NEW), "query_id")}
    cases, gold = [], []
    for q in selected:
        claim = claims[int(q)]
        if claim["claim"] != data["queries"][q]:
            raise ValueError("Official claim/BEIR text mismatch")
        evidence, labels = [], set()
        for did, groups in claim["evidence"].items():
            for group in groups:
                label = {"SUPPORT": "SUPPORT", "CONTRADICT": "REFUTE"}[group["label"]]
                labels.add(label)
                evidence.append(
                    {
                        "document_id": did,
                        "label": label,
                        "sentence_indices": group["sentences"],
                        "sentences": [corpus[int(did)]["abstract"][i] for i in group["sentences"]],
                    }
                )
        candidate = (
            next(iter(labels)) if len(labels) == 1 else ("INSUFFICIENT" if not labels else None)
        )
        refs = sorted(set(map(str, claim["cited_doc_ids"])) | set(claim["evidence"]))
        contexts = {}
        for arm in ARMS:
            ranking = rankings[arm][q]
            if (arm == "A" and ranking["mode"] != "llm_no_thinking") or (
                arm == "B" and ranking["mode"] != "relevance_v1"
            ):
                raise ValueError("Wrong frozen rerank variant")
            contexts[arm] = []
            for row in ranking["chunks"][:4]:
                c = chunks[row["chunk_id"]]
                contexts[arm].append({**c, "title": data["corpus"][c["document_id"]]["title"]})
            if len(contexts[arm]) != 4:
                raise ValueError("Missing frozen context")
        case = {
            "query_id": q,
            "claim": claim["claim"],
            "contexts": contexts,
            "reference_documents": [data["corpus"][d] for d in refs],
        }
        cases.append(case)
        points = [
            {
                "id": f"p{i}",
                "text": " ".join(e["sentences"]),
                "basis": f"{e['document_id']} sentences {e['sentence_indices']}",
            }
            for i, e in enumerate(evidence, 1)
        ]
        if not points:
            points = [
                {
                    "id": "p1",
                    "text": "The annotated evidence is insufficient to determine "
                    "whether this claim holds. HUMAN: specify what evidence is missing.",
                    "basis": "Empty upstream evidence; NOT a global absence-of-evidence proof",
                }
            ]
        gold.append(
            {
                "query_id": q,
                "case_sha256": digest(case),
                "label": candidate,
                "upstream_evidence": evidence,
                "required_points": points,
                "context_sufficient": {"A": None, "B": None},
                "review": pending(),
            }
        )
    fresh(output)
    write_rows(output / "cases.jsonl", cases)
    write_rows(output / "gold.draft.jsonl", gold)
    render_cases(output / "review.html", cases, gold)
    paths = [
        OLD,
        NEW,
        CHUNKS,
        UPSTREAM / "source.json",
        UPSTREAM / "claims_train.jsonl",
        UPSTREAM / "corpus.jsonl",
        DEFAULT_PREPARED / "report.json",
        PROTOCOL,
        Path(__file__),
        ROOT / "app/services/generation.py",
    ]
    manifest = {
        "schema_version": 1,
        "split": "development",
        "query_ids": selected,
        "arms": ARMS,
        "generation": CONFIG,
        "planned_calls": 200,
        "source_hashes": {str(p.relative_to(ROOT)): hash_file(p) for p in paths},
        "cases_sha256": hash_file(output / "cases.jsonl"),
        "messages_sha256": digest([context_messages(c, a) for c in cases for a in ARMS]),
    }
    write_json(output / "manifest.json", manifest)
    from collections import Counter

    write_json(
        output / "status.json",
        {
            "status": "awaiting_human_gold_review",
            "human_reviewed": 0,
            "generated_answers": 0,
            "model_calls": 0,
            "draft_label_counts": dict(Counter(g["label"] for g in gold)),
            "corpus_normalized_matches": len(corpus),
            "planned_calls_after_seal": 200,
        },
    )
    return manifest


def verify_pack(pack):
    manifest = read_object(pack / "manifest.json")
    if (
        manifest["schema_version"] != 1
        or manifest["generation"] != CONFIG
        or manifest["arms"] != ARMS
        or manifest["split"] != "development"
    ):
        raise ValueError("Unknown frozen protocol")
    for name, sha in manifest["source_hashes"].items():
        if hash_file(ROOT / name) != sha:
            raise ValueError(f"Frozen source changed: {name}")
    cases = read_rows(pack / "cases.jsonl")
    if (
        hash_file(pack / "cases.jsonl") != manifest["cases_sha256"]
        or [c["query_id"] for c in cases] != manifest["query_ids"]
        or len(cases) != 100
        or len(unique(cases, "query_id")) != 100
        or digest([context_messages(c, a) for c in cases for a in ARMS])
        != manifest["messages_sha256"]
    ):
        raise ValueError("Frozen cases/messages changed")
    return manifest, cases


def validate_gold(cases, gold):
    by_id = unique(gold, "query_id")
    if set(by_id) != {c["query_id"] for c in cases}:
        raise ValueError("Gold cohort mismatch")
    for case in cases:
        row = by_id[case["query_id"]]
        approval(row["review"])
        if row["case_sha256"] != digest(case) or row["label"] not in LABELS:
            raise ValueError("Gold label/case mismatch")
        if set(row["context_sufficient"]) != set(ARMS) or any(
            type(v) is not bool for v in row["context_sufficient"].values()
        ):
            raise ValueError("Human context sufficiency for both arms required")
        points = row["required_points"]
        if (
            not points
            or len(unique(points, "id")) != len(points)
            or any(not all(text(p.get(k)) for k in ("id", "text", "basis")) for p in points)
            or any("HUMAN:" in p["text"] for p in points)
        ):
            raise ValueError("Human-verified essential points and bases required")
    return by_id


def seal(pack, gold_path):
    _, cases = verify_pack(pack)
    gold = read_rows(gold_path)
    validate_gold(cases, gold)
    # Exclusive directory: signing off a different gold requires a new explicit study.
    target = pack / "sealed"
    fresh(target)
    write_rows(target / "gold.jsonl", gold)
    write_json(
        target / "seal.json",
        {
            "gold_sha256": hash_file(target / "gold.jsonl"),
            "manifest_sha256": hash_file(pack / "manifest.json"),
            "human_reviewed": 100,
        },
    )


def sealed_gold(pack, cases):
    seal_doc = read_object(pack / "sealed/seal.json")
    gold = pack / "sealed/gold.jsonl"
    if (
        hash_file(gold) != seal_doc["gold_sha256"]
        or hash_file(pack / "manifest.json") != seal_doc["manifest_sha256"]
    ):
        raise ValueError("Sealed human gold changed")
    return validate_gold(cases, read_rows(gold))


def citations(answer):
    return [
        {"offset": m.start(), "source": int(m.group(1)), "token": m.group()}
        for m in CITATION.finditer(answer)
    ]


def estimate_cost(usage, pricing):
    if not pricing or not usage:
        return None
    hit, miss, completion = (
        usage.get(k)
        for k in ("prompt_cache_hit_tokens", "prompt_cache_miss_tokens", "completion_tokens")
    )
    if any(type(x) is not int or x < 0 for x in (hit, miss, completion)) or hit + miss != usage.get(
        "prompt_tokens"
    ):
        return None
    result = (
        Decimal(hit) * Decimal(str(pricing["input_cache_hit_per_million"]))
        + Decimal(miss) * Decimal(str(pricing["input_cache_miss_per_million"]))
        + Decimal(completion) * Decimal(str(pricing["output_per_million"]))
    ) / 1000000
    return float(result)


def load_pricing(path):
    if path is None:
        return None
    value = read_object(path)
    if (
        value.get("currency") != "CNY"
        or not text(value.get("source_url"))
        or not text(value.get("verified_at"))
    ):
        raise ValueError("Verified independent official CNY prices required")
    for k in ("input_cache_hit_per_million", "input_cache_miss_per_million", "output_per_million"):
        rate = Decimal(str(value[k]))
        if not rate.is_finite() or rate < 0:
            raise ValueError("Invalid official price")
    return value


def review_template(row, gold_sha):
    return {
        "query_id": row["query_id"],
        "arm": row["arm"],
        "answer_sha256": digest(row),
        "gold_sha256": gold_sha,
        "conclusion": None,
        "abstained": None,
        "facts": [],
        "citations": [{**c, "statement": "", "supported": None} for c in row["citations"]],
        "covered_point_ids": [],
        "review": pending(),
        "second_review": pending(),
    }


def generate(pack, output, execute=False, pricing_path=None):
    manifest, cases = verify_pack(pack)
    sealed_gold(pack, cases)  # Even planning refuses unreviewed gold.
    pricing = load_pricing(pricing_path)
    plan = {
        "manifest_sha256": hash_file(pack / "manifest.json"),
        "gold_sha256": hash_file(pack / "sealed/gold.jsonl"),
        "config": manifest["generation"],
        "pricing": pricing,
        "planned_calls": 200,
        "order": [
            {"query_id": c["query_id"], "arm": a}
            for i, c in enumerate(cases)
            for a in (("A", "B") if i % 2 == 0 else ("B", "A"))
        ],
    }
    if output.exists():
        raise ValueError("Use a fresh output directory; no overwrite/resume/retry")
    client = None
    if execute:
        from openai import OpenAI

        from app.config import get_settings

        settings = get_settings()
        if (
            settings.deepseek_base_url != CONFIG["base_url"]
            or settings.openai_chat_model != CONFIG["model"]
            or not settings.deepseek_api_key.strip()
        ):
            raise ValueError("Frozen official DeepSeek generation config/key required")
        client = OpenAI(
            api_key=settings.deepseek_api_key,
            base_url=CONFIG["base_url"],
            max_retries=0,
            timeout=CONFIG["timeout_seconds"],
        )
    fresh(output)
    write_json(output / "plan.json", plan)
    write_json(
        output / "status.json",
        {
            "status": "running" if execute else "plan_only",
            "started": 0,
            "finished": 0,
            "planned": 200,
        },
    )
    if client is None:
        return plan
    by_id = unique(cases, "query_id")
    rows = []
    try:
        for item in plan["order"]:
            case, arm = by_id[item["query_id"]], item["arm"]
            request = {k: CONFIG[k] for k in ("model", "temperature", "max_tokens", "stream")}
            request["messages"] = context_messages(case, arm)
            started = time.perf_counter()
            append(output / "events.jsonl", {**item, "event": "started", "request": request})
            row = {
                **item,
                "request": request,
                "contexts": case["contexts"][arm],
                "status": "error",
                "error_type": None,
                "response": None,
                "answer": "",
                "usage": None,
                "finish_reason": None,
                "estimated_cost_cny": None,
                "actual_cost_cny": None,
            }
            try:
                response = client.chat.completions.create(**request)
                row["response"] = response.model_dump(mode="json")
                row["usage"] = response.usage.model_dump(mode="json") if response.usage else None
                row["answer"] = response.choices[0].message.content or ""
                row["finish_reason"] = response.choices[0].finish_reason
                row["status"] = (
                    "ok" if text(row["answer"]) and row["finish_reason"] == "stop" else "invalid"
                )
                row["estimated_cost_cny"] = estimate_cost(row["usage"], pricing)
            except Exception as exc:
                # Error messages/headers may contain credentials; store class only.
                row["error_type"] = type(exc).__name__
            row["generation_ms"] = (time.perf_counter() - started) * 1000
            row["citations"] = citations(row["answer"])
            append(output / "answers.jsonl", row)
            append(output / "events.jsonl", {**item, "event": "finished", "status": row["status"]})
            rows.append(row)
            write_json(
                output / "status.json",
                {"status": "running", "started": len(rows), "finished": len(rows), "planned": 200},
            )
    finally:
        client.close()
    write_rows(
        output / "answer_reviews.draft.jsonl",
        [review_template(r, plan["gold_sha256"]) for r in rows],
    )
    write_json(
        output / "status.json",
        {
            "status": "awaiting_answer_reviews",
            "started": len(rows),
            "finished": len(rows),
            "planned": 200,
        },
    )
    write_json(
        output / "answers_manifest.json",
        {
            "answers_sha256": hash_file(output / "answers.jsonl"),
            "plan_sha256": hash_file(output / "plan.json"),
            "events_sha256": hash_file(output / "events.jsonl"),
        },
    )
    return plan


def replay_response(row):
    """Reconstruct derived fields from saved wire response, never from review labels."""
    if row["response"] is not None:
        from openai.types.chat import ChatCompletion

        response = ChatCompletion.model_validate(row["response"])
        answer = response.choices[0].message.content or ""
        finish = response.choices[0].finish_reason
        usage = response.usage.model_dump(mode="json") if response.usage else None
        status = "ok" if text(answer) and finish == "stop" else "invalid"
        if (
            row["answer"] != answer
            or row["finish_reason"] != finish
            or row["usage"] != usage
            or row["status"] != status
        ):
            raise ValueError("Raw response/derived fields mismatch")
    elif row["status"] != "error" or row["answer"] or row["usage"] is not None:
        raise ValueError("Request error cannot invent answer/usage")
    if row["citations"] != citations(row["answer"]):
        raise ValueError("Citation extraction mismatch")


def validate_review(row, review, gold, gold_sha):
    approval(review["review"])
    if review["answer_sha256"] != digest(row) or review["gold_sha256"] != gold_sha:
        raise ValueError("Review bound to different answer/gold")
    if review["conclusion"] not in LABELS | {"INVALID"} or type(review["abstained"]) is not bool:
        raise ValueError("Human conclusion and abstention required")
    if row["status"] != "ok" and review["conclusion"] != "INVALID":
        raise ValueError("Failed/truncated answer must count as incorrect")
    expected = citations(row["answer"])
    judged = review["citations"]
    if [{k: c[k] for k in ("offset", "source", "token")} for c in judged] != expected:
        raise ValueError("Must judge every citation occurrence, without duplicates")
    for c in judged:
        if type(c["supported"]) is not bool or not text(c["statement"]):
            raise ValueError("Citation statement and entailment required")
        if not 1 <= c["source"] <= len(row["contexts"]) and c["supported"]:
            raise ValueError("Out-of-range citation cannot be supported")
    offsets = {c["offset"] for c in expected}
    for f in review["facts"]:
        if (
            not text(f.get("span"))
            or f["span"] not in row["answer"]
            or type(f.get("supported")) is not bool
            or not isinstance(f.get("citation_offsets"), list)
            or not set(f["citation_offsets"]) <= offsets
        ):
            raise ValueError("Each atomic fact needs an answer span, support and citation offsets")
    if row["status"] == "ok" and not review["facts"]:
        raise ValueError("Successful answer needs atomic fact annotations (including NEI)")
    points = {p["id"] for p in gold["required_points"]}
    covered = review["covered_point_ids"]
    if len(covered) != len(set(covered)) or not set(covered) <= points:
        raise ValueError("Unknown/duplicate essential point")


def ratio(n, d):
    return n / d if d else None


def summarize_quality(rows, reviews, gold):
    correct, supported, facts, cited_facts, accurate, citation_count = 0, 0, 0, 0, 0, 0
    complete, fact_macro = [], []
    sufficient = insufficient = false_refusal = proper_refusal = refusal = 0
    nei_correct = nei_count = 0
    for row, review in zip(rows, reviews, strict=True):
        g = gold[row["query_id"]]
        ok = row["status"] == "ok"
        match = ok and review["conclusion"] == g["label"]
        correct += match
        if g["label"] == "INSUFFICIENT":
            nei_count += 1
            nei_correct += match
        fs = review["facts"]
        supported += sum(f["supported"] for f in fs)
        facts += len(fs)
        fact_macro.append(ratio(sum(f["supported"] for f in fs), len(fs)))
        cs = review["citations"]
        valid_offsets = {c["offset"] for c in cs if c["supported"]}
        cited_facts += sum(bool(set(f["citation_offsets"]) & valid_offsets) for f in fs)
        accurate += sum(c["supported"] for c in cs)
        citation_count += len(cs)
        complete.append(len(review["covered_point_ids"]) / len(g["required_points"]) if ok else 0)
        enough = g["context_sufficient"][row["arm"]]
        sufficient += enough
        insufficient += not enough
        refusal += review["abstained"]
        false_refusal += enough and review["abstained"]
        proper_refusal += not enough and review["abstained"]
    return {
        "n": len(rows),
        "conclusion_accuracy": ratio(correct, len(rows)),
        "supported_facts": supported,
        "facts": facts,
        "fact_support_rate": ratio(supported, facts),
        "fact_support_macro": statistics.mean([x for x in fact_macro if x is not None])
        if any(x is not None for x in fact_macro)
        else None,
        "accurate_citations": accurate,
        "citations": citation_count,
        "citation_accuracy": ratio(accurate, citation_count),
        "fact_citation_coverage": ratio(cited_facts, facts),
        "completeness": statistics.mean(complete),
        "refusal_rate": ratio(refusal, len(rows)),
        "sufficient_context_n": sufficient,
        "false_refusal_rate": ratio(false_refusal, sufficient),
        "insufficient_context_n": insufficient,
        "proper_refusal_rate": ratio(proper_refusal, insufficient),
        "nei_n": nei_count,
        "nei_accuracy": ratio(nei_correct, nei_count),
        "request_success_rate": ratio(sum(r["status"] == "ok" for r in rows), len(rows)),
    }


def quality_gate(metrics):
    targets = {
        "conclusion_accuracy": 0.85,
        "fact_support_rate": 0.95,
        "citation_accuracy": 0.95,
        "completeness": 0.90,
        "false_refusal_rate": 0.05,
    }
    if any(metrics[k] is None for k in targets):
        return "not_evaluable"
    passed = (
        all(
            metrics[k] <= v if k == "false_refusal_rate" else metrics[k] >= v
            for k, v in targets.items()
        )
        and metrics["request_success_rate"] == 1
    )
    return "pass" if passed else "fail"


def paired_interval(deltas):
    rng = random.Random(20261020)
    values = sorted(statistics.mean(rng.choices(deltas, k=len(deltas))) for _ in range(10000))

    def percentile(p):
        pos = (len(values) - 1) * p
        lo = int(pos)
        return values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (pos - lo)

    return [percentile(0.025), percentile(0.975)]


def score(pack, answers, reviews_path, output):
    _, cases = verify_pack(pack)
    gold = sealed_gold(pack, cases)
    plan = read_object(answers / "plan.json")
    if (
        plan["manifest_sha256"] != hash_file(pack / "manifest.json")
        or plan["gold_sha256"] != hash_file(pack / "sealed/gold.jsonl")
        or plan["config"] != CONFIG
    ):
        raise ValueError("Answers from different frozen study")
    am = read_object(answers / "answers_manifest.json")
    for name in ("answers", "plan", "events"):
        suffix = ".json" if name == "plan" else ".jsonl"
        if hash_file(answers / (name + suffix)) != am[name + "_sha256"]:
            raise ValueError("Generated artifact hash changed")
    rows = read_rows(answers / "answers.jsonl")
    reviews = read_rows(reviews_path)

    def key(r):
        return r["query_id"], r["arm"]

    by_row, by_review = {key(r): r for r in rows}, {key(r): r for r in reviews}
    wanted = {(c["query_id"], a) for c in cases for a in ARMS}
    if len(rows) != 200 or len(reviews) != 200 or set(by_row) != wanted or set(by_review) != wanted:
        raise ValueError("Exactly 200 distinct paired answers/reviews required")
    expected_order = [
        {"query_id": c["query_id"], "arm": a}
        for i, c in enumerate(cases)
        for a in (("A", "B") if i % 2 == 0 else ("B", "A"))
    ]
    if plan["order"] != expected_order or [key(r) for r in rows] != [
        key(r) for r in expected_order
    ]:
        raise ValueError("Frozen alternating execution order changed")
    events = read_rows(answers / "events.jsonl")
    if len(events) != 400:
        raise ValueError("Unexpected event count")
    for event in ("started", "finished"):
        subset = [e for e in events if e["event"] == event]
        if len(subset) != 200 or {key(e) for e in subset} != wanted:
            raise ValueError("Every planned call needs one start and one finish")
    by_case = unique(cases, "query_id")
    second_missing = []
    for k, row in by_row.items():
        c, a = by_case[k[0]], k[1]
        request = {
            field: CONFIG[field] for field in ("model", "temperature", "max_tokens", "stream")
        }
        request["messages"] = context_messages(c, a)
        if row["request"] != request or row["contexts"] != c["contexts"][a]:
            raise ValueError("Gold leakage or changed generation inputs")
        replay_response(row)
        started_event = next(e for e in events if key(e) == k and e["event"] == "started")
        finished_event = next(e for e in events if key(e) == k and e["event"] == "finished")
        if started_event["request"] != request or finished_event["status"] != row["status"]:
            raise ValueError("Event/result mismatch")
        review = by_review[k]
        validate_review(row, review, gold[k[0]], plan["gold_sha256"])
        mandatory = (
            cases.index(c) % 10 == 0
            or row["status"] != "ok"
            or review["conclusion"] != gold[k[0]]["label"]
            or review["abstained"]
            or any(not f["supported"] for f in review["facts"])
            or any(not x["supported"] for x in review["citations"])
        )
        if mandatory:
            try:
                approval(review["second_review"])
                if review["second_review"]["reviewer"] == review["review"]["reviewer"]:
                    raise ValueError("Independent second reviewer required")
            except ValueError:
                second_missing.append(list(k))
    summaries = {}
    for arm in ARMS:
        ordered = [by_row[(c["query_id"], arm)] for c in cases]
        judged = [by_review[(c["query_id"], arm)] for c in cases]
        m = summarize_quality(ordered, judged, gold)
        costs = [r["estimated_cost_cny"] for r in ordered]
        # Recompute estimates from raw usage and the frozen independent price profile.
        if costs != [estimate_cost(r["usage"], plan["pricing"]) for r in ordered]:
            raise ValueError("Cost estimate replay mismatch")
        m.update(
            {
                "absolute_gate": quality_gate(m),
                "generation_latency": latency_summary([r["generation_ms"] for r in ordered]),
                "known_estimated_cost_cny": sum(x for x in costs if x is not None),
                "unknown_cost_calls": sum(x is None for x in costs),
                "estimated_cost_cny": sum(costs) if all(x is not None for x in costs) else None,
                "actual_cost_cny": None,
            }
        )
        summaries[ARMS[arm]] = m
    paired = []
    for c in cases:
        q = c["query_id"]
        a, b = by_review[(q, "A")], by_review[(q, "B")]
        ca = by_row[(q, "A")]["status"] == "ok" and a["conclusion"] == gold[q]["label"]
        cb = by_row[(q, "B")]["status"] == "ok" and b["conclusion"] == gold[q]["label"]
        paired.append(
            {"query_id": q, "old_correct": ca, "new_correct": cb, "delta": int(cb) - int(ca)}
        )
    deltas = [p["delta"] for p in paired]
    interval = paired_interval(deltas)
    new = summaries["relevance_v1"]
    gate = new["absolute_gate"]
    if second_missing:
        gate = "awaiting_second_review"
    elif statistics.mean(deltas) < 0 or any(
        m["request_success_rate"] != 1 for m in summaries.values()
    ):
        gate = "fail"
    report = {
        "split": "development_first100",
        "status": gate,
        "metrics": summaries,
        "paired_conclusion_delta": statistics.mean(deltas),
        "paired_95ci": interval,
        "improved": deltas.count(1),
        "worsened": deltas.count(-1),
        "tied": deltas.count(0),
        "improvement_evidence": not second_missing
        and statistics.mean(deltas) > 0
        and interval[0] > 0,
        "pending_second_reviews": second_missing,
        "pairs": paired,
        "reviews_sha256": hash_file(reviews_path),
        "answers_sha256": am["answers_sha256"],
        "gold_sha256": plan["gold_sha256"],
        "judge_model_calls": 0,
    }
    fresh(output)
    write_json(output / "report.json", report)
    lines = [
        "# SciFact 回答质量：固定100题配对评测",
        "",
        f"正式门禁：**{gate}**。",
        "开发集已用于选参；离线生成回放，不是在线全链路或独立test效果。",
        "",
        "```json",
        json.dumps({k: v for k, v in report.items() if k != "pairs"}, ensure_ascii=False, indent=2),
        "```",
        "",
        "## 逐题对照（全量，含失败）",
    ]
    for c in cases:
        q = c["query_id"]
        lines += [f"### {q}: {c['claim']}", f"核验金标：{gold[q]['label']}"]
        for arm in ARMS:
            row = by_row[(q, arm)]
            lines += [
                f"**{ARMS[arm]} / {row['status']}**",
                "",
                row["answer"],
                "",
                "引用映射："
                + json.dumps(
                    [
                        {"source": i, "document_id": x["document_id"], "chunk_id": x["chunk_id"]}
                        for i, x in enumerate(row["contexts"], 1)
                    ]
                ),
            ]
    (output / "report.md").write_text("\n\n".join(lines) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("seal")
    p.add_argument("--pack", type=Path, required=True)
    p.add_argument("--gold", type=Path, required=True)
    p = sub.add_parser("generate")
    p.add_argument("--pack", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--pricing", type=Path)
    p = sub.add_parser("score")
    p.add_argument("--pack", type=Path, required=True)
    p.add_argument("--answers", type=Path, required=True)
    p.add_argument("--reviews", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepare(args.output)
    elif args.command == "seal":
        seal(args.pack, args.gold)
    elif args.command == "generate":
        generate(args.pack, args.output, args.execute, args.pricing)
    else:
        score(args.pack, args.answers, args.reviews, args.output)
    print(json.dumps({"command": args.command, "status": "ok"}))


if __name__ == "__main__":
    main()
