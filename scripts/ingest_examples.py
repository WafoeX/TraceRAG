"""Ingest the bundled example documents (handbook, FAQ, onboarding PDF).

Usage:
    python -m scripts.ingest_examples

Requires a running database (`make db`) and OPENAI_API_KEY in your env/.env.
"""
from __future__ import annotations

import io
from pathlib import Path

from pypdf import PdfReader

from app.database import SessionLocal, init_db
from app.services.ingestion import ingest_document

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
SUPPORTED = {".md", ".txt", ".pdf"}


def read_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(io.BytesIO(path.read_bytes()))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def main() -> None:
    init_db()
    db = SessionLocal()
    try:
        for path in sorted(EXAMPLES.glob("*")):
            if path.name.startswith("_") or path.suffix.lower() not in SUPPORTED:
                continue
            doc = ingest_document(db, path.stem, read_file(path), path.name)
            print(f"✓ {path.name:<26} → document #{doc.id} ({len(doc.chunks)} chunks)")
    finally:
        db.close()


if __name__ == "__main__":
    main()
