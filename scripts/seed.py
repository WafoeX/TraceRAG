"""Seed the store with a sample document and run one query end-to-end.

Usage:
    python -m scripts.seed

Requires a running database (`make db`) and OPENAI_API_KEY in your env/.env.
"""
from __future__ import annotations

from app.database import SessionLocal, init_db
from app.services.generation import generate_answer
from app.services.ingestion import ingest_document
from app.services.search import search

SAMPLE = """
Acme Inc. Refund Policy

Customers may request a full refund within 30 days of purchase, no questions
asked. Refunds are processed to the original payment method within 5-7 business
days. After 30 days, refunds are offered as store credit only.

Subscriptions can be cancelled at any time from the account dashboard. A
cancellation stops future billing but does not retroactively refund the current
billing period.

For enterprise contracts, refund terms are defined in the signed agreement and
override this general policy.
""".strip()

QUESTION = "How many days do I have to get a full refund?"


def main() -> None:
    init_db()
    db = SessionLocal()
    try:
        doc = ingest_document(db, "Acme Refund Policy", SAMPLE, "seed")
        print(f"✓ Ingested document #{doc.id} ({len(doc.chunks)} chunks)\n")

        result = search(db, QUESTION, top_k=3)
        answer = generate_answer(QUESTION, result.chunks)

        print(f"Q: {QUESTION}")
        print(f"A: {answer}  (confidence {result.confidence})\n")
        print("Sources:")
        for i, c in enumerate(result.chunks, start=1):
            print(f"  [{i}] {c.document_title} (chunk {c.chunk_index}, score {c.score})")
    finally:
        db.close()


if __name__ == "__main__":
    main()
