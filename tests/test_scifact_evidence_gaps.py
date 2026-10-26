"""Synthetic-only checks; run with unittest without application pytest fixtures."""

import copy
import tempfile
import unittest
from pathlib import Path

from scripts import analyze_scifact_evidence_gaps as gaps


class EvidenceGapTests(unittest.TestCase):
    def setUp(self):
        self.document = {"document_id": "doc", "title": "Title", "text": "abcdefghij"}
        self.raw = {"_id": "doc", "title": "Title", "text": "abcdefghij"}
        self.full = "Title\n\nabcdefghij"
        self.pieces = [self.piece("c0", self.full[:12]), self.piece("c1", self.full[9:])]

    @staticmethod
    def piece(cid, text):
        return {"chunk_id": cid, "text": text, "text_sha256": gaps.text_sha(text)}

    def coverage(self, sent):
        return gaps.document_coverage(self.document, self.raw, self.pieces, sent)

    def test_overlap_is_counted_once(self):
        self.assertEqual(gaps.covered_chars(10, [(0, 7), (4, 10)]), 10)
        self.assertEqual(gaps.covered_chars(10, [(0, 3), (5, 10)]), 8)

    def test_invalid_interval_rejected(self):
        for span in [(-1, 3), (2, 2), (0, 11)]:
            with self.assertRaises(ValueError):
                gaps.covered_chars(10, [span])

    def test_all_overlapping_chunks_cover_document(self):
        result = self.coverage({"c0", "c1"})
        self.assertTrue(result["full_local_document_sent"])
        self.assertEqual(result["covered_characters"], len(self.full))
        self.assertEqual(result["whole_corpus_answer_absence"], "not_established")

    def test_partial_and_absent_documents_not_full(self):
        for sent, expected in [({"c0"}, 12), (set(), 0)]:
            result = self.coverage(sent)
            self.assertFalse(result["full_local_document_sent"])
            self.assertEqual(result["covered_characters"], expected)

    def test_disconnected_chunks_do_not_hide_gap(self):
        self.pieces = [self.piece("c0", self.full[:8]), self.piece("c1", self.full[10:])]
        self.assertFalse(self.coverage({"c0", "c1"})["full_local_document_sent"])

    def test_changed_raw_corpus_rejected(self):
        self.raw["text"] = "changed"
        with self.assertRaises(ValueError):
            self.coverage({"c0", "c1"})

    def test_bad_chunk_hash_rejected(self):
        self.pieces[0]["text_sha256"] = "wrong"
        with self.assertRaises(ValueError):
            self.coverage({"c0"})

    def test_missing_and_ambiguous_chunk_anchors_rejected(self):
        for text in ["not in document", "", "e"]:
            self.pieces = [self.piece("c0", text)]
            with self.assertRaises(ValueError):
                self.coverage({"c0"})

    def test_uncertain_or_missed_qrel_not_in_target(self):
        arm = {"document_qrels_metrics": {"Hit@4": 1}, "context_sufficiency": "insufficient"}
        self.assertTrue(gaps.target(arm))
        arm["context_sufficiency"] = "uncertain"
        self.assertFalse(gaps.target(arm))
        arm["context_sufficiency"] = "insufficient"
        arm["document_qrels_metrics"]["Hit@4"] = 0
        self.assertFalse(gaps.target(arm))

    def test_duplicate_keys_rejected(self):
        with self.assertRaises(ValueError):
            gaps.unique([{"id": "1"}, {"id": "1"}], lambda r: r["id"])

    def test_labels_must_exactly_match_cohort_and_category(self):
        specs = {
            "schema_version": 1,
            "categories": {"gap": "Gap"},
            "cases": [{"query_id": "q", "category": "gap", "missing_conditions": ["x"]}],
        }
        self.assertEqual(set(gaps.validate_labels(specs, ["q"])), {"q"})
        for ids in [[], ["other"], ["q", "other"]]:
            with self.assertRaises(ValueError):
                gaps.validate_labels(specs, ids)
        invalid = copy.deepcopy(specs)
        invalid["cases"][0]["category"] = "unknown"
        with self.assertRaises(ValueError):
            gaps.validate_labels(invalid, ["q"])

    def test_output_must_be_new_direct_child_of_runs(self):
        with self.assertRaises(ValueError):
            gaps.analyze(gaps.RUNS)
        with self.assertRaises(ValueError):
            gaps.analyze(gaps.ROOT / "not-a-run")

    def test_write_new_cannot_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "out.json"
            gaps.write_new(path, {"keep": True})
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                gaps.write_new(path, {"keep": False})
            self.assertEqual(path.read_bytes(), original)

    def test_invalid_json_and_missing_input_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            for reader in (gaps.load, gaps.rows):
                with self.assertRaises(ValueError):
                    reader(path)
            path.write_text("{invalid")
            for reader in (gaps.load, gaps.rows):
                with self.assertRaises(ValueError):
                    reader(path)


if __name__ == "__main__":
    unittest.main()
