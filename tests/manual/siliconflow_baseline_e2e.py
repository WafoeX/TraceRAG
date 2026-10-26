"""Real guarded PG, fake SDK only. No paid calls or HTTP allowed."""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from openai.types import CreateEmbeddingResponse
from openai.types.chat import ChatCompletion

from scripts.baseline_support import read_json, write_json
from scripts.capture_siliconflow_baseline import main as capture


class FakeClient:
    def __init__(self):
        self.calls: Counter = Counter()
        self.closed = False
        self.embeddings = SimpleNamespace(create=self.embed)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.rerank))

    def embed(self, **kwargs):
        assert "dimensions" not in kwargs
        self.calls["embedding"] += 1
        return CreateEmbeddingResponse.model_validate(
            {
                "object": "list",
                "model": "BAAI/bge-m3",
                "data": [
                    {"object": "embedding", "index": i, "embedding": [1.0] + [0.0] * 1023}
                    for i, _ in enumerate(kwargs["input"])
                ],
                "usage": {"prompt_tokens": 10, "total_tokens": 10},
            }
        )

    def rerank(self, **kwargs):
        self.calls["llm_rerank"] += 1
        match = re.search(r"JSON array of (\d+)", kwargs["messages"][1]["content"])
        assert match is not None
        return ChatCompletion.model_validate(
            {
                "id": "fake",
                "object": "chat.completion",
                "created": 0,
                "model": kwargs["model"],
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": "[" + ",".join(["1"] * int(match.group(1))) + "]",
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                    "completion_tokens_details": {"reasoning_tokens": 2},
                },
            }
        )

    def close(self):
        self.closed = True


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
        code = capture(["--output", str(args.output), "--execute"])
        http.assert_not_called()
    state = read_json(args.output / "status.json")
    assert code == 0 and state["status"] == "complete", state
    assert state["completed_queries"] == 32
    assert state["old_data_unchanged"]
    assert client.closed
    assert client.calls == {"embedding": 35, "llm_rerank": 16}
    report = {
        "status": "passed",
        "fixture_queries": 32,
        "real_model_calls": 0,
        "http_calls": 0,
        "fake_sdk_calls": dict(client.calls),
        "old_data_unchanged": True,
        "quality_claim": False,
    }
    write_json(args.output / "protocol_verification.json", report)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
