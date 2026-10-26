"""Materialize frozen B top4/top50 views and bind a partial witness audit.

This is not an exhaustive top50 sufficiency scorer. No keyword-based negatives,
model calls, application imports, test queries, or changes to frozen labels.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from scripts import analyze_scifact_evidence_gaps as gaps

ROOT = gaps.ROOT
RUNS = gaps.RUNS
GAP_RUN = RUNS / "scifact-evidence-gap-attribution-v2"
SPECS = ROOT / "configs/evaluation/scifact.context_window_witnesses.v1.json"
PROTOCOL = ROOT / "docs/refactor/SCIFACT_EVIDENCE_GAP_ANALYSIS.md"


def bind_document(spec, document_id, ranked, chunks, corpus):
    document = corpus[document_id]
    full = document["title"] + "\n\n" + document["text"]
    quote = spec["quote"]
    if not quote or quote not in full or not spec["reason"]:
        raise ValueError("Witness requires a literal quote and an explicit scope judgment")
    if spec["assessment"] not in ("does_not_close_gap", "partial_support_scope_uncertain"):
        raise ValueError("This partial pass does not establish a complete-claim witness")
    matches, reviewed = [], []
    for rank, item in enumerate(ranked[:50], 1):
        chunk = chunks[item["chunk_id"]]
        if chunk["document_id"] != document_id:
            continue
        if rank <= 4:
            raise ValueError("Witness must be new evidence at saved ranks 5-50")
        if gaps.text_sha(chunk["text"]) != chunk["text_sha256"]:
            raise ValueError("Witness chunk hash mismatch")
        if chunk["text"] not in full:
            raise ValueError("Witness chunk not in the fully read local record")
        reviewed.append(chunk["chunk_id"])
        start = chunk["text"].find(quote)
        if start >= 0:
            matches.append(
                {
                    "rank": rank,
                    "chunk_id": chunk["chunk_id"],
                    "text_sha256": chunk["text_sha256"],
                    "start": start,
                    "end": start + len(quote),
                    "quote": quote,
                }
            )
    if not reviewed or not matches:
        raise ValueError("Quote not present in new saved top50 evidence")
    return {
        "document_id": document_id,
        "local_record_sha256": gaps.text_sha(full),
        "local_record_chars": len(full),
        "full_local_record_read_by_reviewer": True,
        "machine_can_prove_review_completeness": False,
        "assessment": spec["assessment"],
        "reason": spec["reason"],
        "reviewed_new_chunk_ids": reviewed,
        "quote_bindings": matches,
    }


def provisional_result(query_id, baseline, witness_records, ranked):
    reviewed = {cid for w in witness_records for cid in w["reviewed_new_chunk_ids"]}
    new_ids = {c["chunk_id"] for c in ranked[4:50]}
    if not reviewed or not reviewed <= new_ids:
        raise ValueError("Each diagnosis requires explicitly read new candidate evidence")
    return {
        "query_id": query_id,
        "frozen_top4_sufficiency": baseline,
        "top50_sufficiency": "uncertain",
        "basis": "partial_witness_review_not_exhaustive_top50_semantic_review",
        "confirmed_recovered": False,
        "confirmed_unrecoverable": False,
        "inherited_top4_positions": 4,
        "directly_reviewed_new_positions": len(reviewed),
        "unreviewed_new_positions": len(new_ids - reviewed),
        "witnesses": witness_records,
        "whole_corpus_answer_absence": "not_established",
    }


def audit_window(output):
    output = output.resolve()
    if output.exists() or output.parent != RUNS.resolve():
        raise ValueError("Use a new immediate child of runs/; no overwrites")
    previous = gaps.load(GAP_RUN / "verification.json")
    # Includes frozen collector/context/pipeline, raw corpus, chunks and rankings.
    inputs = dict(previous["inputs_sha256"])
    for rel, expected in inputs.items():
        if gaps.sha(ROOT / rel) != expected:
            raise ValueError(f"Frozen upstream source changed: {rel}")
    for name, expected in previous["outputs_sha256"].items():
        path = GAP_RUN / name
        if gaps.sha(path) != expected:
            raise ValueError("Frozen gap-case output changed")
        inputs[str(path.relative_to(ROOT))] = expected
    for path in (GAP_RUN / "verification.json", Path(__file__), SPECS, PROTOCOL):
        inputs[str(path.relative_to(ROOT))] = gaps.sha(path)
    specs = gaps.load(SPECS)
    if specs["schema_version"] != 1 or specs["window"] != 50:
        raise ValueError("Unknown witness protocol/window")
    if specs["confirmed_complete_claim_witnesses"]:
        raise ValueError("Complete-claim judgments require a separate explicit scoring pass")
    gap_cases = gaps.rows(GAP_RUN / "cases.jsonl")
    diagnoses = gaps.unique(specs["cases"], lambda r: r["query_id"])
    if set(diagnoses) != {c["query_id"] for c in gap_cases}:
        raise ValueError("Witness annotations must cover exactly the frozen 26 diagnoses")
    used_docs = {d for c in diagnoses.values() for d in c["reviewed_documents"]}
    if used_docs != set(specs["documents"]):
        raise ValueError("Unbound or missing document annotation")
    pairs = gaps.rows(gaps.ASSESSMENT / "pipeline_per_query.jsonl")
    pending = gaps.unique(
        gaps.rows(gaps.PACK / "answers.pending.jsonl"), lambda r: (r["query_id"], r["arm"])
    )
    metrics = gaps.unique(
        gaps.rows(gaps.CONTEXT / "metrics.jsonl"), lambda r: (r["query_id"], r["arm"])
    )
    rankings = gaps.unique(gaps.rows(gaps.RANKINGS["B"]), lambda r: r["query_id"])
    chunks = gaps.unique(gaps.rows(gaps.CHUNKS), lambda r: r["chunk_id"])
    corpus = gaps.unique(gaps.rows(gaps.PREPARED / "corpus.jsonl"), lambda r: r["document_id"])
    diagnostic_ids = set(diagnoses)
    control_ids = {p["query_id"] for p in pairs if p["B"]["context_sufficiency"] == "sufficient"}
    if diagnostic_ids & control_ids or (len(diagnostic_ids), len(control_ids)) != (26, 44):
        raise ValueError("Frozen diagnostic/control cohort changed")
    views, diagnostics, controls = [], [], []
    pool_ids = set()
    for pair in pairs:
        q = pair["query_id"]
        if q not in diagnostic_ids | control_ids:
            continue
        baseline = pair["B"]["context_sufficiency"]
        if baseline != metrics[q, "B"]["context_sufficiency"]:
            raise ValueError("Baseline sufficiency source disagreement")
        ranked = rankings[q]["chunks"]
        ids = [c["chunk_id"] for c in ranked[:50]]
        actual = [c["chunk_id"] for c in pending[q, "B"]["contexts"]]
        if len(ids) != len(set(ids)) or len(ids) != 50 or ids[:4] != actual:
            raise ValueError("Top50 or original sent top4 mismatch")
        for cid in ids:
            if gaps.text_sha(chunks[cid]["text"]) != chunks[cid]["text_sha256"]:
                raise ValueError("Materialized chunk hash mismatch")
        pool_ids.update(ids)
        views.append(
            {
                "query_id": q,
                "query_position": pair["query_position"],
                "query": pending[q, "B"]["query"],
                "cohort": "diagnostic" if q in diagnostic_ids else "control",
                "frozen_top4_sufficiency": baseline,
                "top4_chunk_ids": ids[:4],
                "top50_chunk_ids": ids,
            }
        )
        if q in control_ids:
            controls.append(
                {
                    "query_id": q,
                    "frozen_top4_sufficiency": baseline,
                    "top4_witness_preserved_byte_identically": True,
                    "new_top50_semantic_reassessment": "not_performed",
                    "new_answer_quality_or_nonregression_claim": False,
                }
            )
            continue
        dids = diagnoses[q]["reviewed_documents"]
        if len(dids) != len(set(dids)):
            raise ValueError("Duplicate reviewed document within a query")
        witnesses = [
            bind_document(specs["documents"][did], did, ranked, chunks, corpus) for did in dids
        ]
        diagnostics.append(provisional_result(q, baseline, witnesses, ranked))
    primary = {c["query_id"] for c in gap_cases if c["arms"]["B"]["in_target_stratum"]}
    direct_ids = {
        cid for c in diagnostics for w in c["witnesses"] for cid in w["reviewed_new_chunk_ids"]
    }
    summary = {
        "status": "partial_top50_witness_audit_not_exhaustive_sufficiency_evaluation",
        "reviewer": specs["reviewer"],
        "diagnostic_queries": len(diagnostics),
        "controls": len(controls),
        "materialized_top50_positions": len(views) * 50,
        "unique_materialized_chunks": len(pool_ids),
        "diagnostic_unique_top50_chunks": len(
            {cid for v in views if v["cohort"] == "diagnostic" for cid in v["top50_chunk_ids"]}
        ),
        "full_local_records_read": len(used_docs),
        "directly_reviewed_unique_new_chunks": len(direct_ids),
        "directly_reviewed_query_new_positions": sum(
            c["directly_reviewed_new_positions"] for c in diagnostics
        ),
        "unreviewed_diagnostic_new_positions": sum(
            c["unreviewed_new_positions"] for c in diagnostics
        ),
        "diagnostic_provisional_sufficiency": dict(
            Counter(c["top50_sufficiency"] for c in diagnostics)
        ),
        "primary_B_insufficient_queries": len(primary),
        "primary_confirmed_recovered_witnesses": sum(
            c["confirmed_recovered"] for c in diagnostics if c["query_id"] in primary
        ),
        "primary_not_resolved": len(primary),
        "confirmed_unrecoverable_queries": 0,
        "control_top4_witnesses_preserved": len(controls),
        "controls_new_semantic_reassessments": 0,
        "new_generation_or_judge_API_calls": 0,
        "database_requests": 0,
        "main_session_AI_cost_cny": None,
        "selection_rule_implemented": False,
        "limitations": [
            "Partial purposeful witness reading, not a complete semantic review of 3500 positions.",
            "Zero confirmed recovery witnesses is not an oracle recovery rate of zero.",
            "Unreviewed candidates prevent confirmed absence/insufficiency conclusions.",
            "Controls retain old evidence, but are not 44 fresh semantic nonregression passes.",
            "Development window50 was selected after seeing query62 rank45; not held-out.",
            "No evidence that a deployable K4 selector or generated answers improved.",
        ],
    }
    if gaps.BLOCKED or any(gaps.sha(ROOT / p) != h for p, h in inputs.items()):
        raise ValueError("Forbidden event or input mutation")
    output.mkdir()
    for name, data in (
        ("candidate_views.jsonl", views),
        ("diagnostics.jsonl", diagnostics),
        ("controls.jsonl", controls),
        ("candidate_chunks.jsonl", [chunks[cid] for cid in sorted(pool_ids)]),
    ):
        gaps.write_new(output / name, data, lines=True)
    gaps.write_new(output / "summary.json", summary)
    with (output / "protocol_snapshot.md").open("x") as handle:
        handle.write(PROTOCOL.read_text())
    names = [
        "candidate_views.jsonl",
        "diagnostics.jsonl",
        "controls.jsonl",
        "candidate_chunks.jsonl",
        "summary.json",
        "protocol_snapshot.md",
    ]
    gaps.write_new(
        output / "verification.json",
        {
            "status": "passed",
            "inputs_sha256": inputs,
            "outputs_sha256": {name: gaps.sha(output / name) for name in names},
            "forbidden_runtime_events": gaps.BLOCKED,
            "checks_prove_bindings_not_AI_semantic_completeness": True,
        },
    )
    print(summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.addaudithook(gaps.audit)
    audit_window(args.output)


if __name__ == "__main__":
    main()
