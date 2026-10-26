"""HTTP probe safety without real DB/model calls; sockets stay disabled."""

from __future__ import annotations

import hashlib
import json
import sys
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from openai.types.chat import ChatCompletion
from sqlalchemy.engine import make_url

import scripts.siliconflow_e2e as e2e
import scripts.siliconflow_http_smoke as smoke
from app.config import Settings


def test_readonly_url_pins_database_and_forces_readonly():
    schema = "baseline_" + "a" * 32
    url = make_url(smoke.readonly_url(schema))
    assert url.host == "127.0.0.1" and url.port == 55432
    assert url.database == "budgetrag_baseline"
    assert (
        url.query["options"] == f"-csearch_path={schema},public -cdefault_transaction_read_only=on"
    )


@pytest.mark.parametrize(
    "schema",
    ["public", "baseline_bad;DROP TABLE chunks", "baseline_" + "a" * 31, "infra_" + "a" * 32],
)
def test_unapproved_or_injected_schema_rejected(schema):
    with pytest.raises(ValueError):
        smoke.readonly_url(schema)


def test_query_only_guard_blocks_ingestion_and_fourth_request(tmp_path):
    state = {"calls": [], "known_estimated_cost_cny": "0", "estimated_cost_cny": "0"}
    guard = e2e.FixtureCalls(
        state, tmp_path / "report.json", Decimal("20"), "SECRET", query_only=True
    )
    embed = Mock(
        return_value=SimpleNamespace(
            usage=SimpleNamespace(model_dump=lambda: {"total_tokens": 11}),
            data=[SimpleNamespace(index=0, embedding=[0.1] * 1024)],
        )
    )
    with pytest.raises(ValueError):
        guard("embedding", "BAAI/bge-m3", embed, {"model": "BAAI/bge-m3", "input": [e2e.DOCUMENT]})
    embed.assert_not_called()
    guard("embedding", "BAAI/bge-m3", embed, {"model": "BAAI/bge-m3", "input": [e2e.QUESTION]})
    for role, content in (("llm_rerank", "[10]"), ("generation", "Yes, within 30 days [1].")):
        response = ChatCompletion.model_validate(
            {
                "id": "fake",
                "object": "chat.completion",
                "created": 0,
                "model": "deepseek-ai/DeepSeek-V4.1-Flash",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": content},
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }
        )
        guard(
            role,
            "deepseek-ai/DeepSeek-V4.1-Flash",
            lambda response=response: response,
            {"model": "deepseek-ai/DeepSeek-V4.1-Flash", "messages": [], "max_tokens": 64},
        )
    extra = Mock()
    with pytest.raises(ValueError):
        guard("generation", "deepseek-ai/DeepSeek-V4.1-Flash", extra, {})
    extra.assert_not_called()
    assert len(state["calls"]) == 3
    assert [r["role"] for r in state["calls"]] == ["embedding", "llm_rerank", "generation"]


def valid_answer():
    return {
        "answer": "Customers may request a refund within 30 days [1].",
        "citations": [
            {
                "document_id": 1,
                "document_title": "Public refund policy",
                "chunk_index": 0,
                "score": 10.0,
                "snippet": e2e.DOCUMENT[:240].strip(),
            }
        ],
    }


def test_response_citation_contract():
    smoke.validate_answer(valid_answer(), 1)


@pytest.mark.parametrize(
    "field,value", [("document_id", 2), ("chunk_index", 7), ("snippet", "wrong")]
)
def test_wrong_fixture_citation_rejected(field, value):
    body = valid_answer()
    body["citations"][0][field] = value
    with pytest.raises(ValueError):
        smoke.validate_answer(body, 1)


def test_plan_does_not_construct_db_model_or_asgi_client(tmp_path, monkeypatch):
    prices = json.loads((smoke.ROOT / "configs/providers/siliconflow.pricing.json").read_text())
    monkeypatch.setattr(smoke, "ROOT", tmp_path)
    source = tmp_path / "runs" / "source"
    source.mkdir(parents=True)
    (source / "report.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "schema": "baseline_" + "a" * 32,
                "vector_column_type": "vector(1024)",
                "fixture_sha256": hashlib.sha256(e2e.DOCUMENT.encode()).hexdigest(),
                "question": e2e.QUESTION,
                "old_data_unchanged": True,
                "document_id": 1,
            }
        )
    )
    (source / "accounting.json").write_text(
        json.dumps(
            {"currency": "CNY", "reference_cumulative_reported_plus_estimates_cny": "0.003790"}
        )
    )
    config_path = tmp_path / "configs/providers/siliconflow.pricing.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(json.dumps(prices))
    settings = Settings.model_construct(
        openai_api_key="offline",
        model_provider="siliconflow",
        openai_base_url="https://api.siliconflow.cn/v1",
        embedding_dim=1024,
        openai_embedding_model="BAAI/bge-m3",
        openai_chat_model="deepseek-ai/DeepSeek-V4.1-Flash",
        siliconflow_max_cost_cny=Decimal("20"),
    )
    monkeypatch.setattr(smoke, "get_settings", lambda: settings)
    forbidden = Mock(side_effect=AssertionError("No DB/model/HTTP in plan"))
    for name in ("database_snapshot", "OpenAI", "TestClient"):
        monkeypatch.setattr(smoke, name, forbidden)
    state = smoke.run(tmp_path / "plan", source)
    assert state["status"] == "planned" and state["calls"] == []
    assert state["available_budget_cny"] == "19.996210"
    latest = tmp_path / "runs" / "latest"
    latest.mkdir()
    (latest / "report.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "schema": "baseline_" + "a" * 32,
                "database_data_unchanged": True,
            }
        )
    )
    ledger = latest / "accounting.json"
    ledger.write_text(
        json.dumps(
            {"currency": "CNY", "reference_cumulative_reported_plus_estimates_cny": "0.005040"}
        )
    )
    streamed = smoke.run(tmp_path / "stream_plan", source, stream=True, accounting_run=latest)
    assert streamed["stream_mode"] is True
    assert streamed["token_latency_verified"] is False
    assert streamed["available_budget_cny"] == "19.994960"
    assert streamed["accounting_source_run"] == "runs/latest"
    ledger.write_text(
        json.dumps({"currency": "CNY", "reference_cumulative_reported_plus_estimates_cny": "0.001"})
    )
    with pytest.raises(ValueError, match="stale"):
        smoke.run(tmp_path / "never", source, stream=True, accounting_run=latest)
    assert not (tmp_path / "never").exists()
    forbidden.assert_not_called()


def test_preloaded_database_execute_rejected_before_other_work(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "app.database", object())
    with pytest.raises(ValueError, match="fresh process"):
        smoke.run(tmp_path / "never", tmp_path / "unknown", True)
    assert not (tmp_path / "never").exists()


def test_stream_parser_reassembles_fragmented_utf8_and_citation_frame():
    wire = (
        "退款可在30天内申请 [1]\x1f"
        + json.dumps({"citations": [{"document_id": 1, "snippet": "中文"}]}, ensure_ascii=False)
    ).encode("utf-8")
    chunks = [wire[:1], wire[1:5], wire[5:-2], wire[-2:]]
    body = smoke.decode_stream(b"".join(chunks))
    assert body["answer"] == "退款可在30天内申请 [1]"
    assert body["citations"][0]["snippet"] == "中文"


def test_refusal_stream_can_have_empty_citations():
    assert smoke.decode_stream(b'No relevant context.\x1f{"citations": []}') == {
        "answer": "No relevant context.",
        "citations": [],
    }


@pytest.mark.parametrize(
    "wire",
    [
        b"",
        b"answer without frame",
        b"answer\x1f{}",
        b"answer\x1f[]",
        b'answer\x1f{"citations":null}',
        b'answer\x1f{"citations":[]}garbage',
        b"answer\x1f{bad json}",
        b'a\x1f{"citations":[]}\x1f',
        b'\xff\x1f{"citations":[]}',
        b'\x1f{"citations":[]}',
        b'answer\x1f{"citations":[],"extra":1}',
    ],
)
def test_truncated_malformed_extra_or_invalid_utf8_stream_rejected(wire):
    with pytest.raises(ValueError):
        smoke.decode_stream(wire)
