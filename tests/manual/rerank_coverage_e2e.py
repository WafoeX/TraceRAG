"""Existing frozen PG schema read-only, fake SDK; no HTTP or paid requests."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any
from unittest.mock import patch

from scripts.baseline_support import read_json, write_json
from scripts.compare_rerank_coverage import main as compare
from tests.manual.siliconflow_baseline_e2e import FakeClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from app.config import Settings

    fields: dict[str, Any] = {
        "_env_file": None,
        "openai_api_key": "fixture-not-real-key",
        "model_provider": "siliconflow",
        "openai_base_url": "https://api.siliconflow.cn/v1",
        "openai_embedding_model": "BAAI/bge-m3",
        "embedding_dim": 1024,
        "openai_chat_model": "deepseek-ai/DeepSeek-V4.1-Flash",
        "run_db_init": False,
    }
    settings = Settings(**fields)
    client = FakeClient()
    with (
        patch("app.config.get_settings", return_value=settings),
        patch("openai.OpenAI", return_value=client),
        patch("httpx.Client.send", side_effect=AssertionError("HTTP forbidden")) as http,
    ):
        code = compare(["--output", str(args.output), "--execute"])
        http.assert_not_called()
    state = read_json(args.output / "status.json")
    assert code == 0 and state["status"] == "complete", state
    assert state["completed_queries"] == 16
    assert state["database_read_only"] and state["database_data_unchanged"]
    assert state["model_attempts"] == 32
    assert client.closed and client.calls == {"embedding": 16, "llm_rerank": 16}
    from scripts.compare_rerank_coverage import read_rows

    calls = read_rows(args.output / "model_calls.jsonl")
    prompts = [r["request"]["messages"][1]["content"] for r in calls if r["role"] == "llm_rerank"]
    chunks = read_json(args.output / "stored_chunks.json")
    assert all(all(c["content"] in p for c in chunks) for p in prompts)
    report = {
        "status": "passed",
        "fixture_queries": 16,
        "fake_sdk_calls": dict(client.calls),
        "real_model_calls": 0,
        "http_calls": 0,
        "read_only": True,
        "all_tables_unchanged": True,
        "full_chunk_in_every_prompt": True,
        "quality_claim": False,
    }
    write_json(args.output / "protocol_verification.json", report)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
