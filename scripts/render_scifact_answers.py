"""Verify saved paired answers and render an unscored human-review package offline.

No model/SQL calls, no automatic claim-correctness judge, no change to frozen inputs.
"""

from __future__ import annotations

import argparse
import html
import json
from collections import Counter
from pathlib import Path

from scripts import evaluate_scifact_answers as ev
from scripts.prepare_scifact import hash_file, write_json
from scripts.run_scifact_baseline import latency_summary


def load_verified(pack: Path, answers: Path):
    _, cases = ev.verify_pack(pack)
    gold = ev.sealed_gold(pack, cases)
    plan = ev.read_object(answers / "plan.json")
    manifest = ev.read_object(answers / "answers_manifest.json")
    if (
        plan["manifest_sha256"] != hash_file(pack / "manifest.json")
        or plan["gold_sha256"] != hash_file(pack / "sealed/gold.jsonl")
        or plan["config"] != ev.CONFIG
    ):
        raise ValueError("Answers not bound to frozen inputs and reviewed gold")
    for name in ("plan", "answers", "events"):
        suffix = ".json" if name == "plan" else ".jsonl"
        if hash_file(answers / (name + suffix)) != manifest[name + "_sha256"]:
            raise ValueError("Saved generation artifact hash mismatch")
    rows = ev.read_rows(answers / "answers.jsonl")
    events = ev.read_rows(answers / "events.jsonl")
    expected = [
        {"query_id": c["query_id"], "arm": a}
        for i, c in enumerate(cases)
        for a in (("A", "B") if i % 2 == 0 else ("B", "A"))
    ]
    if (
        len(cases) != 100
        or len(rows) != 200
        or len(events) != 400
        or plan["order"] != expected
        or [{k: r[k] for k in ("query_id", "arm")} for r in rows] != expected
    ):
        raise ValueError("Incomplete/duplicate/changed 200-pair execution")
    by_case = ev.unique(cases, "query_id")
    for row, item, start, finish in zip(rows, expected, events[::2], events[1::2], strict=True):
        case, arm = by_case[row["query_id"]], row["arm"]
        request = {k: ev.CONFIG[k] for k in ("model", "temperature", "max_tokens", "stream")}
        request["messages"] = ev.context_messages(case, arm)
        if (
            row["request"] != request
            or row["contexts"] != case["contexts"][arm]
            or start != {**item, "event": "started", "request": request}
            or finish != {**item, "event": "finished", "status": row["status"]}
        ):
            raise ValueError("Request/context/event replay mismatch")
        ev.replay_response(row)
        if row["estimated_cost_cny"] != ev.estimate_cost(row["usage"], plan["pricing"]):
            raise ValueError("Cost replay mismatch")
    return cases, gold, rows, plan, manifest


def complete_sum(values):
    if not values:
        return None
    total = 0
    for value in values:
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            return None
        total += value
    return total


def usage_summary(rows):
    usages = [r["usage"] for r in rows if r["usage"] is not None]
    result: dict[str, int | None] = {"calls": len(rows), "calls_with_usage": len(usages)}
    for field in (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "prompt_cache_hit_tokens",
        "prompt_cache_miss_tokens",
    ):
        values = [u.get(field) for u in usages]
        result[field] = complete_sum(values) if len(usages) == len(rows) else None
    reasoning = [(u.get("completion_tokens_details") or {}).get("reasoning_tokens") for u in usages]
    result["reasoning_tokens_if_reported"] = (
        complete_sum(reasoning) if len(usages) == len(rows) else None
    )
    return result


def source_details(contexts):
    parts = ["<details><summary>实际生成上下文与引用编号</summary>"]
    for i, c in enumerate(contexts, 1):
        parts += [
            f"<h4>[{i}] {html.escape(c['title'])}</h4>",
            "<p>document_id="
            + html.escape(c["document_id"])
            + "</p><pre>"
            + html.escape(c["text"])
            + "</pre>",
        ]
    return "\n".join(parts) + "</details>"


def render(pack: Path, answers: Path, output: Path):
    cases, gold, rows, plan, manifest = load_verified(pack, answers)
    by_key = {(r["query_id"], r["arm"]): r for r in rows}
    reviews = ev.read_rows(answers / "answer_reviews.draft.jsonl")
    if len(reviews) != 200 or any(r["review"]["status"] != "pending" for r in reviews):
        raise ValueError("Render from untouched pending review template, not existing human edits")
    by_review = {(r["query_id"], r["arm"]): r for r in reviews}
    if len(by_review) != 200 or any(
        by_review.get((r["query_id"], r["arm"])) != ev.review_template(r, plan["gold_sha256"])
        for r in rows
    ):
        raise ValueError("Pending review template bindings changed")
    summaries = {}
    for arm, version in ev.ARMS.items():
        selected = [r for r in rows if r["arm"] == arm]
        costs = [r["estimated_cost_cny"] for r in selected]
        summaries[version] = {
            "answers": len(selected),
            "statuses": dict(Counter(r["status"] for r in selected)),
            "usage": usage_summary(selected),
            "generation_only_latency": latency_summary([r["generation_ms"] for r in selected]),
            "known_estimated_cost_cny": sum(v for v in costs if v is not None),
            "unknown_estimated_cost_calls": sum(v is None for v in costs),
            "estimated_total_cost_cny": sum(costs) if all(v is not None for v in costs) else None,
            "actual_cost_cny": None,
            "citation_occurrences": sum(len(r["citations"]) for r in selected),
            "out_of_range_citation_occurrences": sum(
                not 1 <= c["source"] <= len(r["contexts"]) for r in selected for c in r["citations"]
            ),
        }
    report = {
        "status": "generated_awaiting_answer_reviews",
        "query_count": len(cases),
        "answers": len(rows),
        "model_calls": len(rows),
        "judge_model_calls": 0,
        "human_gold_count": len(gold),
        "human_answer_reviews": 0,
        "gold_label_counts": dict(Counter(g["label"] for g in gold.values())),
        "conclusion_accuracy": None,
        "quality_gate": "not_yet_evaluated",
        "quality_improved": None,
        "response_model_ids": sorted({r["response"]["model"] for r in rows if r["response"]}),
        "summaries": summaries,
        "usage_total": usage_summary(rows),
        "answers_sha256": manifest["answers_sha256"],
        "gold_sha256": plan["gold_sha256"],
        "verification": {
            "raw_response_replay": True,
            "frozen_request_context_replay": True,
            "all_200_starts_finishes_matched": True,
            "cost_replay": True,
            "automatic_retries": 0,
        },
        "limits": [
            "Offline retrieval replay, not online HTTP E2E latency",
            "Citation existence/range is NOT semantic citation correctness",
            "Usage logged; prices not frozen/verified for this run, total expense unknown",
            "No final quality conclusions before answer scoring and independent review",
        ],
    }
    sections = []
    for case in cases:
        q = case["query_id"]
        g = gold[q]
        parts = [
            "<h2>" + html.escape(q + ": " + case["claim"]) + "</h2>",
            "<details><summary>人工金标与必要要点</summary><pre>"
            + html.escape(
                json.dumps(
                    {k: g[k] for k in ("label", "required_points", "context_sufficient")},
                    ensure_ascii=False,
                    indent=2,
                )
            )
            + "</pre></details>",
        ]
        for arm in ev.ARMS:
            row = by_key[(q, arm)]
            review = by_review[(q, arm)]
            parts += [
                f"<details><summary>答案 {arm} / {html.escape(row['status'])}</summary>",
                "<h3>模型原文（尚未评分）</h3><pre>" + html.escape(row["answer"]) + "</pre>",
                source_details(row["contexts"]),
                "<p>先独立审核此答案，再查看另一个答案。"
                "填写事实原文片段、每个引用的支持性、结论和必要要点覆盖；不要自动批准。</p>",
                '<textarea class="review" rows="25" spellcheck="false">'
                + html.escape(json.dumps(review, ensure_ascii=False, indent=2))
                + "</textarea></details>",
            ]
        sections.append("<section>" + "\n".join(parts) + "</section>")
    page = (
        "<!doctype html><meta charset='utf-8'><title>SciFact 200份回答待审</title>"
        "<style>body{max-width:1100px;margin:2em auto;font:16px system-ui}"
        "pre{white-space:pre-wrap;overflow-wrap:anywhere}textarea{width:100%;box-sizing:border-box;"
        "font:14px monospace}summary{cursor:pointer;padding:1em;background:#eee}"
        "section{border-top:2px solid #888;margin:2em 0}"
        "header{position:sticky;top:0;background:white;"
        "padding:1em;border:1px solid #888;z-index:1}</style>"
        "<h1>固定100题、200份配对回答：待评分</h1>"
        "<p>金标已封存；答案尚未人工评分，不能宣称质量达标或改善。"
        "引用编号合法不等于引用支持陈述。按冻结协议审核；指定样本和错误/拒答等需独立第二人复核。</p>"
        "<p>本页不联网、不自动保存、不自动批准。关闭/刷新前请导出全部评分JSONL。"
        "事实citation_offsets使用下面引用列表中的offset值；每个原子事实必须对应答案原文span。</p>"
        '<header><button id="export">导出200份评分JSONL</button> '
        '<input id="find" placeholder="查找题号/断言"><span id="notice"></span></header>'
        + "\n".join(sections)
        + """<script>
        document.getElementById('find').oninput=function(){
          const q=this.value.toLowerCase();
          document.querySelectorAll('section').forEach(s=>{
            s.hidden=!s.querySelector('h2').textContent.toLowerCase().includes(q);
          });
        };
        document.getElementById('export').onclick=function(){
          try {
            const rows=Array.from(document.querySelectorAll('textarea.review'),t=>
              JSON.stringify(JSON.parse(t.value)));
            const blob=new Blob([rows.join('\\n')+'\\n'],{type:'application/x-ndjson'});
            const url=URL.createObjectURL(blob),a=document.createElement('a');
            a.href=url;a.download='answer_reviews.reviewed.jsonl';a.click();
            URL.revokeObjectURL(url);
            document.getElementById('notice').textContent='已导出 '+rows.length+' 份评分。';
          } catch(e) {document.getElementById('notice').textContent='JSON格式错误: '+e.message;}
        };
        </script>"""
    )
    ev.fresh(output)
    write_json(output / "generation_report.json", report)
    (output / "answer_review.html").write_text(page, encoding="utf-8")
    lines = [
        "# SciFact 回答配对生成完成（尚未评分）",
        "",
        "100题、两组共200份真实模型回答；人工金标已封存。",
        "**当前没有结论正确率等正式质量分数，不能宣称达标/改善。**",
        "",
        "## 生成与离线核验",
        "```json",
        json.dumps(report, ensure_ascii=False, indent=2),
        "```",
        "",
        "## 下一步",
        "",
        "打开 answer_review.html 逐份评分并导出 answer_reviews.reviewed.jsonl。",
        "评分规则见 docs/refactor/SCIFACT_ANSWER_PROTOCOL.md。",
        "完成指定独立复核后，用冻结 score 命令生成正式新旧质量对照报告。",
        "",
        "## 全部逐题回答（原文，含失败）",
    ]
    for case in cases:
        q = case["query_id"]
        lines += [f"### {q}: {case['claim']}", f"人工金标：{gold[q]['label']}"]
        for arm, version in ev.ARMS.items():
            row = by_key[(q, arm)]
            lines += [
                f"#### {version}（{row['status']}）",
                row["answer"],
                "引用编号映射："
                + json.dumps(
                    [
                        {"n": i, "document_id": c["document_id"], "chunk_id": c["chunk_id"]}
                        for i, c in enumerate(row["contexts"], 1)
                    ],
                    ensure_ascii=False,
                ),
            ]
    (output / "generation_report.md").write_text("\n\n".join(lines) + "\n", encoding="utf-8")
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pack", type=Path, required=True)
    p.add_argument("--answers", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    report = render(args.pack, args.answers, args.output)
    print(json.dumps({"status": report["status"], "answers": report["answers"]}))


if __name__ == "__main__":
    main()
