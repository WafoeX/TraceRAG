"""Offline browser fixture, NEVER a production server or real model benchmark.

Uses only the fixed disposable integration DB, synthetic model responses, a
separate schema and port 18089. Run setup, api + worker, then cleanup.
"""

from __future__ import annotations

import argparse
import os
from decimal import Decimal
from typing import cast
from unittest.mock import patch

from sqlalchemy import create_engine, text

from app.config import Settings
from scripts.prepare_test_cache import CACHE_DIR, verify_cache

TEST_URL = (
    "postgresql+psycopg://tracerag_test:local_test_only@127.0.0.1:55439/tracerag_application_test"
)


def fake_embed(texts: list[str]) -> list[list[float]]:
    return [[1.0] + [0.0] * 1535 for _ in texts]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["setup", "api", "worker", "cleanup"])
    parser.add_argument(
        "--resilience-demo",
        action="store_true",
        help="Synthetic UI failures + real account limiter",
    )
    parser.add_argument(
        "--budget-demo", action="store_true", help="Synthetic usage + real budget ledger"
    )
    args = parser.parse_args()
    if args.resilience_demo and args.budget_demo:
        parser.error("Choose one demo fixture mode")
    mode = args.mode
    if not verify_cache():
        raise SystemExit("Prepare the offline tokenizer cache first")
    os.environ["TIKTOKEN_CACHE_DIR"] = str(CACHE_DIR)
    settings = Settings.model_construct(
        openai_api_key="offline-fixture-not-a-key",
        database_url=TEST_URL + "?options=-csearch_path%3Dtracerag_browser_test%2Cpublic",
        run_db_init=False,
        rate_limit_enabled=args.resilience_demo,
        rate_limit_per_minute=6 if args.resilience_demo else 10,
        use_rerank=False,
        session_cookie_secure=False,  # this fixture binds local HTTP only
    )
    with patch("app.config.get_settings", return_value=settings):
        # Patch before importing ingestion/retrieval: no provider client is used.
        from app.services import embeddings

        if args.budget_demo:
            settings.application_pricing_file = "tests/fixtures/application-budget-pricing.json"
            settings.openai_base_url = "https://offline.invalid/v1"
            settings.openai_embedding_model = "fixture-embedding"
            settings.openai_chat_model = "fixture-chat"

        def fixture_embed(texts: list[str]) -> list[list[float]]:
            if not args.budget_demo:
                return fake_embed(texts)
            from app.services.application_budget import priced_call

            response = priced_call(
                "embedding",
                "fixture-embedding",
                {"model": "fixture-embedding", "input": texts},
                lambda: {"usage": {"total_tokens": 10}, "vectors": fake_embed(texts)},
            )
            return cast(list[list[float]], response["vectors"])

        embeddings.embed_texts = fixture_embed
        embeddings.embed_query = lambda question: fixture_embed([question])[0]
        from app import models  # noqa: F401
        from app.database import Base, engine

        if mode in {"setup", "cleanup"}:
            admin = create_engine(TEST_URL)
            with admin.begin() as conn:
                assert conn.scalar(text("SELECT current_database()")) == "tracerag_application_test"
                if mode == "setup":
                    conn.exec_driver_sql("CREATE EXTENSION IF NOT EXISTS vector")
                    conn.exec_driver_sql("CREATE SCHEMA tracerag_browser_test")
                else:
                    conn.exec_driver_sql("DROP SCHEMA tracerag_browser_test CASCADE")
            if mode == "setup":
                Base.metadata.create_all(engine)
                from sqlalchemy.orm import Session

                from app.security import hash_password

                users = []
                with Session(engine) as db:
                    for username in ("alice", "bob"):
                        user = models.User(
                            username=username, password_hash=hash_password("TraceRAG-demo-123")
                        )
                        db.add(user)
                        db.flush()
                        users.append(user.id)
                        db.add(models.KnowledgeBase(owner_id=user.id, name=username + " papers"))
                    db.commit()
                if args.budget_demo:
                    from app.services.budget_ledger import set_limit

                    set_limit(engine, None, Decimal("1"), "synthetic browser fixture allocation")
                    for user_id in users:
                        set_limit(
                            engine, user_id, Decimal("0.05"), "synthetic browser fixture allocation"
                        )
            admin.dispose()
            engine.dispose()
            return
        if mode == "worker":
            from app.worker import main as worker_main

            worker_main()
            return
        from app.routers import query

        def fake_answer(question, chunks):
            if args.resilience_demo:
                from app.services.resilience import ProviderUnavailable, warn

                if "模拟生成超时" in question:
                    raise ProviderUnavailable("provider_timeout")
                if "模拟重排超时" in question:
                    warn("rerank_provider_timeout")
                    warn("rerank_retrieval_order")
            answer = (
                "【离线UI测试：确定性替身，非模型输出】\n"
                "文献描述了实验观察；请核验下方完整原文。[1]\n"
                "编号越界告警测试：[99]。"
            )
            if args.budget_demo:
                from app.services.application_budget import priced_call

                return priced_call(
                    "generation",
                    "fixture-chat",
                    {
                        "model": "fixture-chat",
                        "max_tokens": 2048,
                        "messages": [{"role": "user", "content": question}],
                    },
                    lambda: {
                        "usage": None
                        if "模拟费用未知" in question
                        else {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                        "answer": answer,
                    },
                )["answer"]
            return answer

        query.generate_answer = fake_answer
        import uvicorn

        from app.main import app

        uvicorn.run(app, host="127.0.0.1", port=18089, log_level="warning", proxy_headers=False)


if __name__ == "__main__":
    main()
