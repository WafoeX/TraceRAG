"""Opt-in real-PG collector protocol test, never a model-quality benchmark.

Run: uv run --frozen python -m tests.manual.collector_e2e --output .test-cache/collector-e2e
Requires compose.baseline.yml. OpenAI construction is stubbed and HTTP send is forbidden.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts import capture_baseline as capture
from scripts.baseline_support import BaselineError, read_json, write_json


class FakeResponse:
    def __init__(self, payload: dict):
        self.payload = payload
        self.usage = SimpleNamespace(model_dump=lambda: payload["usage"])
        self.data = [SimpleNamespace(**item) for item in payload.get("data", [])]
        self.choices = [
            SimpleNamespace(message=SimpleNamespace(**item["message"]))
            for item in payload.get("choices", [])
        ]

    def model_dump(self, **kwargs):
        return self.payload


class FakeClient:
    def __init__(self):
        self.calls: Counter = Counter()
        self.closed = False
        self.embeddings = SimpleNamespace(create=self.embed)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.rerank))

    def with_options(self, **kwargs):
        if kwargs != {"max_retries": 0}:
            raise BaselineError("Expected retries disabled in collector")
        return self

    def embed(self, **kwargs):
        self.calls["embedding"] += 1
        if kwargs["dimensions"] != 3:
            raise BaselineError("Fixture expects its own 3-D schema")
        return FakeResponse(
            {
                "data": [
                    {"index": i, "embedding": [1.0, 0.0, 0.0]}
                    for i, _ in enumerate(kwargs["input"])
                ],
                "usage": {"prompt_tokens": 10, "total_tokens": 10},
            }
        )

    def rerank(self, **kwargs):
        self.calls["rerank"] += 1
        match = re.search(r"JSON array of (\d+)", kwargs["messages"][1]["content"])
        if match is None:
            raise BaselineError("Original rerank prompt changed")
        content = json.dumps([1] * int(match.group(1)))
        return FakeResponse(
            {
                "choices": [{"message": {"content": content}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }
        )

    def close(self):
        self.closed = True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    config = capture.load_config(capture.DEFAULT_CONFIG, False)
    config.update(
        provider="FAKE-SDK-INFRASTRUCTURE-ONLY",
        base_url="https://fixture.example.test/v1",
        embedding_model="fake-dense",
        chat_model="fake-rerank",
        embedding_dim=3,
        pricing_and_bounds_source="FICTIONAL UNIT TEST RATES; NOT BILLING DATA",
        pricing={
            kind: {
                "input_per_million_usd": "1",
                "output_per_million_usd": "1",
                "input_token_upper_bound": 100,
                "output_token_upper_bound": 50,
            }
            for kind in ("embedding", "rerank")
        },
    )
    config_path = args.output / "fixture-config.json"
    write_json(config_path, config)
    client = FakeClient()
    with (
        patch.dict(os.environ, {"OPENAI_API_KEY": "fixture-no-real-key"}),
        patch("openai.OpenAI", return_value=client),
        patch("httpx.Client.send", side_effect=AssertionError("HTTP is forbidden")) as http,
    ):
        exit_code = capture.main(
            [
                "--output",
                str(args.output / "run"),
                "--config",
                str(config_path),
                "--execute-models",
                "--max-cost-usd",
                "1",
            ]
        )
        http.assert_not_called()
    state = read_json(args.output / "run/status.json")
    if (
        exit_code != 0
        or state["status"] != "complete"
        or not client.closed
        or client.calls != {"embedding": 35, "rerank": 16}
    ):
        raise BaselineError("Collector protocol integration failed; inspect private fixture output")
    rankings = [
        json.loads(line) for line in (args.output / "run/rankings.jsonl").read_text().splitlines()
    ]
    stages = [
        json.loads(line) for line in (args.output / "run/stages.jsonl").read_text().splitlines()
    ]
    if len(rankings) != 32 or {row["stage"] for row in stages} != {
        "vector_search",
        "keyword_search",
        "reciprocal_rank_fusion",
        "rerank",
    }:
        raise BaselineError("Expected rankings/stage traces missing")
    report = {
        "status": "passed",
        "schema": state["schema"],
        "completed_fixture_queries": 32,
        "fake_sdk_calls": dict(client.calls),
        "real_model_calls": 0,
        "http_requests": 0,
        "quality_metrics": None,
        "scope": "Real PG and original search; fake vectors/scores/usage/prices only",
    }
    if args.report:
        write_json(args.report, report)
    print(
        "Collector protocol E2E passed: 32 fixture queries, 51 fake SDK calls, 0 HTTP/model calls"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
