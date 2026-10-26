"""Synthetic checks for partial witness auditing; no application fixtures."""

import copy
import unittest

import scripts.analyze_scifact_evidence_gaps as gaps
import scripts.audit_scifact_context_window as window


class ContextWindowTests(unittest.TestCase):
    def setUp(self):
        self.document = {"document_id": "target", "title": "Target title", "text": "Narrow result."}
        self.full = "Target title\n\nNarrow result."
        self.chunks = {
            f"c{i}": {
                "chunk_id": f"c{i}",
                "document_id": "other",
                "text": "Other",
                "text_sha256": gaps.text_sha("Other"),
            }
            for i in range(50)
        }
        self.chunks["c4"] = {
            "chunk_id": "c4",
            "document_id": "target",
            "text": self.full,
            "text_sha256": gaps.text_sha(self.full),
        }
        self.ranked = [{"chunk_id": f"c{i}"} for i in range(50)]
        self.spec = {
            "quote": "Narrow result.",
            "assessment": "does_not_close_gap",
            "reason": "This witness does not establish the full claim.",
        }
        self.corpus = {"target": self.document}

    def bind(self):
        return window.bind_document(self.spec, "target", self.ranked, self.chunks, self.corpus)

    def test_quote_bound_to_saved_rank_hash_and_offset(self):
        result = self.bind()
        binding = result["quote_bindings"][0]
        self.assertEqual(binding["rank"], 5)
        self.assertEqual(self.full[binding["start"] : binding["end"]], self.spec["quote"])
        self.assertEqual(binding["text_sha256"], gaps.text_sha(self.full))
        self.assertFalse(result["machine_can_prove_review_completeness"])

    def test_top4_cannot_be_counted_as_new_witness(self):
        self.ranked[0], self.ranked[4] = self.ranked[4], self.ranked[0]
        with self.assertRaises(ValueError):
            self.bind()

    def test_rank51_cannot_enter_window(self):
        target = self.ranked.pop(4)
        self.ranked.append({"chunk_id": "c1"})
        self.ranked.append(target)
        with self.assertRaises(ValueError):
            self.bind()

    def test_unquoted_target_word_is_not_evidence_binding(self):
        self.spec["quote"] = "not present"
        with self.assertRaises(ValueError):
            self.bind()

    def test_empty_quote_or_reason_rejected(self):
        for key in ("quote", "reason"):
            original = self.spec[key]
            self.spec[key] = ""
            with self.assertRaises(ValueError):
                self.bind()
            self.spec[key] = original

    def test_complete_sufficiency_not_inferred_from_keyword(self):
        self.spec["assessment"] = "complete_claim_supported"
        with self.assertRaises(ValueError):
            self.bind()

    def test_changed_chunk_hash_rejected(self):
        self.chunks["c4"]["text_sha256"] = "wrong"
        with self.assertRaises(ValueError):
            self.bind()

    def test_chunk_outside_local_record_rejected(self):
        self.chunks["c4"]["text"] = "not in source"
        self.chunks["c4"]["text_sha256"] = gaps.text_sha("not in source")
        with self.assertRaises(ValueError):
            self.bind()

    def test_partial_reading_stays_uncertain_not_insufficient(self):
        result = window.provisional_result("q", "insufficient", [self.bind()], self.ranked)
        self.assertEqual(result["top50_sufficiency"], "uncertain")
        self.assertEqual(result["directly_reviewed_new_positions"], 1)
        self.assertEqual(result["unreviewed_new_positions"], 45)
        self.assertFalse(result["confirmed_recovered"])
        self.assertFalse(result["confirmed_unrecoverable"])
        self.assertEqual(result["whole_corpus_answer_absence"], "not_established")

    def test_old_uncertainty_not_rewritten(self):
        witness = self.bind()
        result = window.provisional_result("q", "uncertain", [witness], self.ranked)
        self.assertEqual(result["frozen_top4_sufficiency"], "uncertain")
        self.assertEqual(result["top50_sufficiency"], "uncertain")

    def test_unreviewed_or_out_of_window_witness_rejected(self):
        with self.assertRaises(ValueError):
            window.provisional_result("q", "insufficient", [], self.ranked)
        bad = {"reviewed_new_chunk_ids": ["c0"]}
        with self.assertRaises(ValueError):
            window.provisional_result("q", "insufficient", [bad], self.ranked)

    def test_binding_does_not_mutate_sources(self):
        before = copy.deepcopy((self.spec, self.ranked, self.chunks, self.corpus))
        self.bind()
        self.assertEqual((self.spec, self.ranked, self.chunks, self.corpus), before)

    def test_existing_or_outside_output_rejected(self):
        for path in (window.RUNS, window.ROOT / "not-a-run"):
            with self.assertRaises(ValueError):
                window.audit_window(path)


if __name__ == "__main__":
    unittest.main()
