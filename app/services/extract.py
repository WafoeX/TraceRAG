"""Extract plain text from uploaded files (.txt, .md, .pdf)."""

from __future__ import annotations

import io

from fastapi import HTTPException, UploadFile
from pypdf import PdfReader
from starlette.concurrency import run_in_threadpool

from app.services.ingestion import ALLOWED_SUFFIXES, MAX_PDF_PAGES, MAX_UPLOAD_BYTES


async def extract_text(file: UploadFile) -> str:
    try:
        data = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "Upload exceeds 10 MiB")
        name = (file.filename or "").lower()
        if not name.endswith(ALLOWED_SUFFIXES):
            raise HTTPException(422, "Only .txt, .md and .pdf are supported")
        if len(name) > 512:
            raise HTTPException(422, "Filename exceeds 512 characters")
        if name.endswith(".pdf"):

            def parse():
                reader = PdfReader(io.BytesIO(data))
                if reader.is_encrypted or len(reader.pages) > MAX_PDF_PAGES:
                    raise ValueError("Encrypted or oversized PDF")
                return "\n".join(page.extract_text() or "" for page in reader.pages)

            try:
                return await run_in_threadpool(parse)
            except Exception:
                raise HTTPException(422, "Invalid, encrypted or oversized PDF") from None
        return data.decode("utf-8", errors="ignore")
    finally:
        await file.close()
