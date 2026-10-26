"""Download a large public-domain document for large-scale ingestion testing.

Pulls a full book from Project Gutenberg (public domain) into ``examples/_large/``,
which is git-ignored — so we can demonstrate large-document handling WITHOUT
committing big binaries to the repository.

Usage:
    python -m scripts.fetch_large_sample              # War and Peace (~3.3 MB)
    python -m scripts.fetch_large_sample 1342         # Pride and Prejudice (~770 KB)
    python -m scripts.fetch_large_sample --ingest     # download AND ingest it
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

DEFAULT_ID = 2600  # "War and Peace" — deliberately large
OUT_DIR = Path(__file__).resolve().parent.parent / "examples" / "_large"


def fetch(book_id: int) -> Path:
    url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dest = OUT_DIR / f"gutenberg-{book_id}.txt"

    print(f"Downloading {url} ...")
    request = urllib.request.Request(url, headers={"User-Agent": "rag-starter/1.0"})
    with urllib.request.urlopen(request) as response, open(dest, "wb") as out:
        out.write(response.read())

    print(f"✓ Saved {dest} ({dest.stat().st_size / 1_000_000:.2f} MB)")
    return dest


def main() -> None:
    args = sys.argv[1:]
    do_ingest = "--ingest" in args
    ids = [int(a) for a in args if a.isdigit()]
    book_id = ids[0] if ids else DEFAULT_ID

    dest = fetch(book_id)

    if do_ingest:
        from app.database import SessionLocal, init_db
        from app.services.ingestion import ingest_document

        text = dest.read_text(encoding="utf-8", errors="ignore")
        init_db()
        db = SessionLocal()
        try:
            print("Ingesting the full book (this embeds every chunk — give it a minute)...")
            doc = ingest_document(db, dest.stem, text, str(dest))
            print(f"✓ Ingested document #{doc.id} with {len(doc.chunks)} chunks")
        finally:
            db.close()


if __name__ == "__main__":
    main()
