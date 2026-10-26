"""Tiny synthetic ZIP contracts; no public data, DB or model calls in tests."""

from __future__ import annotations

import io
import json
import zipfile
from copy import deepcopy
from http.client import HTTPMessage
from unittest.mock import Mock
from urllib.request import Request

import pytest

from scripts import prepare_scifact as prep


def line(value):
    return json.dumps(value, ensure_ascii=False) + "\n"


@pytest.fixture
def fixture(tmp_path):
    config = json.loads(prep.CONFIG.read_text())
    config["expected_counts"] = {"corpus": 2, "train_queries": 1, "test_queries": 1}
    members = {
        "corpus.jsonl": line({"_id": "d1", "title": "one", "text": "正文一"})
        + line({"_id": "d2", "title": "two", "text": "正文二"}),
        "queries.jsonl": line({"_id": "q1", "text": "development claim"})
        + line({"_id": "q2", "text": "test claim"}),
        "qrels/train.tsv": "query-id\tcorpus-id\tscore\nq1\td1\t1\n",
        "qrels/test.tsv": "query-id\tcorpus-id\tscore\nq2\td2\t1\n",
    }

    def archive(changes=None):
        content = {**members, **(changes or {})}
        target = tmp_path / "fixture.zip"
        with zipfile.ZipFile(target, "w") as package:
            for key, value in content.items():
                package.writestr("scifact/" + key, value)
        config["publisher_md5"] = prep.hash_file(target, "md5")
        config["archive_sha256"] = prep.hash_file(target)
        return target

    return config, members, archive


def test_full_offline_conversion_hashes_and_immutability(fixture, tmp_path):
    config, _, archive = fixture
    source = archive()
    output = tmp_path / "prepared"
    report = prep.prepare(source, output, config)
    assert report["counts"] == config["expected_counts"]
    assert report["checks"]["query_id_overlap"] == 0
    assert report["real_model_calls"] == report["database_writes"] == 0
    assert report["retrieval_metrics"] is None
    assert report["model_api_cost_cny"] == "0"
    for name, metadata in report["files"].items():
        assert prep.hash_file(output / name) == metadata["sha256"]
    assert (output / "source/scifact.zip").read_bytes() == source.read_bytes()
    assert json.loads((output / "splits/development/queries.jsonl").read_text())["query_id"] == "q1"
    assert json.loads((output / "splits/test/qrels.jsonl").read_text())["document_id"] == "d2"
    with pytest.raises(prep.DatasetError, match="immutable"):
        prep.prepare(source, output, config)


@pytest.mark.parametrize(
    "change",
    [
        "duplicate_doc",
        "empty_text",
        "dangling_qrel",
        "duplicate_qrel",
        "zero_score",
        "header",
        "split_overlap",
        "text_overlap",
        "unassigned",
        "count",
        "hash",
        "unsafe",
        "extra",
    ],
)
def test_negative_controls(fixture, tmp_path, change):
    config, members, archive = fixture
    modified = deepcopy(members)
    if change == "duplicate_doc":
        modified["corpus.jsonl"] += members["corpus.jsonl"].splitlines()[0] + "\n"
    elif change == "empty_text":
        modified["queries.jsonl"] = line({"_id": "q1", "text": ""})
    elif change == "dangling_qrel":
        modified["qrels/test.tsv"] = "query-id\tcorpus-id\tscore\nq2\tmissing\t1\n"
    elif change == "duplicate_qrel":
        modified["qrels/test.tsv"] += "q2\td2\t1\n"
    elif change == "zero_score":
        modified["qrels/test.tsv"] = "query-id\tcorpus-id\tscore\nq2\td2\t0\n"
    elif change == "header":
        modified["qrels/test.tsv"] = "query\tdocument\tscore\nq2\td2\t1\n"
    elif change == "split_overlap":
        modified["qrels/test.tsv"] += "q1\td2\t1\n"
    elif change == "text_overlap":
        modified["queries.jsonl"] = line({"_id": "q1", "text": "Claim  A"}) + line(
            {"_id": "q2", "text": " claim a "}
        )
    elif change == "unassigned":
        modified["queries.jsonl"] += line({"_id": "q3", "text": "unassigned claim"})
    elif change == "unsafe":
        modified["../escape.txt"] = "blocked"
    elif change == "extra":
        modified["unexpected.txt"] = "blocked"
    path = archive(modified)
    if change == "count":
        config["expected_counts"]["corpus"] = 100
    elif change == "hash":
        config["publisher_md5"] = "0" * 32
    output = tmp_path / "out"
    with pytest.raises(prep.DatasetError):
        prep.prepare(path, output, config)
    assert not output.exists()
    assert not (tmp_path / "escape.txt").exists()


def test_missing_members_and_invalid_json(fixture, tmp_path):
    config, _, archive = fixture
    path = archive({"queries.jsonl": "not JSON"})
    with pytest.raises(prep.DatasetError):
        prep.prepare(path, tmp_path / "bad", config)
    with zipfile.ZipFile(path, "w") as package:
        package.writestr("scifact/corpus.jsonl", "{}")
    with pytest.raises(prep.DatasetError, match="Missing"):
        prep.read_archive(path, prep.hash_file(path, "md5"))


def test_default_cli_offline_missing_archive(tmp_path, monkeypatch):
    opener = Mock(side_effect=AssertionError("network must not be called"))
    monkeypatch.setattr(prep.urllib.request, "build_opener", opener)
    assert (
        prep.main(["--archive", str(tmp_path / "missing.zip"), "--output", str(tmp_path / "out")])
        == 1
    )
    opener.assert_not_called()


def test_download_explicit_fixed_url_checksum_no_overwrite(tmp_path, fixture, monkeypatch):
    config, _, archive = fixture
    data = archive().read_bytes()
    response = io.BytesIO(data)
    opener = Mock()
    opener.open.return_value = response
    factory = Mock(return_value=opener)
    monkeypatch.setattr(prep.urllib.request, "build_opener", factory)
    target = tmp_path / "download.zip"
    prep.download(target, config)
    assert target.read_bytes() == data
    assert not target.with_name(target.name + ".part").exists()
    opener.open.assert_called_once_with(config["download_url"], timeout=60)
    with pytest.raises(prep.DatasetError):
        prep.download(target, config)
    with pytest.raises(prep.DatasetError):
        prep.download(tmp_path / "evil.zip", {**config, "download_url": "http://localhost/"})
    assert opener.open.call_count == 1


def test_quarantine_preserves_official_data_and_never_changes_test(fixture, tmp_path):
    config, members, archive = fixture
    config["expected_counts"]["train_queries"] = 2
    path = archive(
        {
            "queries.jsonl": line({"_id": "q1", "text": "TEST claim"})
            + line({"_id": "q2", "text": "test  claim"})
            + line({"_id": "q3", "text": "distinct development claim"}),
            "qrels/train.tsv": members["qrels/train.tsv"] + "q3\td1\t1\n",
        }
    )
    output = tmp_path / "prepared"
    report = prep.prepare(path, output, config)
    assert report["counts"]["train_queries"] == 2
    assert report["development_queries"] == 1
    assert report["quarantined_train_queries"] == [
        {"train_query_id": "q1", "matching_test_query_ids": ["q2"]}
    ]
    assert report["qrels"]["original_train"] == 2
    assert report["qrels"]["development"] == 1
    assert (output / "raw/qrels/test.tsv").read_text() == members["qrels/test.tsv"]
    dev = json.loads((output / "splits/development/queries.jsonl").read_text())
    assert dev["query_id"] == "q3"
    assert json.loads((output / "splits/test/queries.jsonl").read_text())["query_id"] == "q2"


def test_pinned_sha256_mismatch_rejected(fixture, tmp_path):
    config, _, archive = fixture
    path = archive()
    config["archive_sha256"] = "0" * 64
    with pytest.raises(prep.DatasetError, match="SHA256"):
        prep.prepare(path, tmp_path / "bad", config)


def test_redirects_are_not_followed():
    with pytest.raises(prep.DatasetError):
        prep.NoRedirect().redirect_request(
            Request("https://example.org"),
            io.BytesIO(),
            302,
            "",
            HTTPMessage(),
            "https://other.example/",
        )
