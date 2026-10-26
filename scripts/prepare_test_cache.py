"""Explicit, hash-verified tokenizer preparation; never downloads during pytest.

Run once with network access: uv run --frozen python -m scripts.prepare_test_cache
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import urllib.request
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parents[1] / ".test-cache" / "tiktoken"
# Matches tiktoken==0.8.0's cl100k_base expected_hash.
TOKENIZER_SHA256 = "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7"
# tiktoken's cache filename is the URL's SHA-1, not a security/integrity check.
# Integrity is independently verified using TOKENIZER_SHA256 above.
CACHE_FILE = CACHE_DIR / "9b5ad71b2ce5302211f9c61530b329a4922fc6a4"


def verify_cache() -> bool:
    """Check the real tokenizer bytes without importing tiktoken or using the network."""
    return (
        CACHE_FILE.is_file()
        and hashlib.sha256(CACHE_FILE.read_bytes()).hexdigest() == TOKENIZER_SHA256
    )


def main() -> None:
    if verify_cache():
        print(f"Tokenizer cache verified: {CACHE_FILE}")
        return
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # Fixed HTTPS endpoint, never caller-controlled. Verify bytes before replacing cache.
    with urllib.request.urlopen(
        "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken", timeout=30
    ) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != TOKENIZER_SHA256:
        raise ValueError("Tokenizer SHA-256 mismatch; cache was not replaced")
    fd, temporary = tempfile.mkstemp(dir=CACHE_DIR, prefix=".download-")
    try:
        with os.fdopen(fd, "wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, CACHE_FILE)
    finally:
        Path(temporary).unlink(missing_ok=True)
    print(f"Tokenizer cache prepared and verified: {CACHE_FILE}")


if __name__ == "__main__":
    main()
