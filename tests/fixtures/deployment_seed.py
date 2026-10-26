"""Synthetic disposable deployment fixture; executed only by opt-in drill."""

import hashlib
import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from io import BytesIO

from pypdf import PdfWriter

from app.database import SessionLocal, engine
from app.models import (
    Chunk,
    ChunkLocation,
    Document,
    DocumentVersion,
    KnowledgeBase,
    LogicalDocument,
    RequestFeedback,
    RequestTrace,
    User,
)
from app.security import hash_password
from app.services.budget_ledger import reserve, retain_unknown, set_limit, settle

now = datetime.now(UTC)
with SessionLocal() as db:
    for name in ("alice", "bob"):
        u = User(username=name, password_hash=hash_password("Deployment-demo-only-123"))
        db.add(u)
        db.flush()
        db.add(KnowledgeBase(owner_id=u.id, name=name + " fixture"))
    db.flush()
    logical = LogicalDocument(knowledge_base_id=1)
    db.add(logical)
    db.flush()
    for n in (1, 2):
        d = Document(title="Synthetic version " + str(n), source="fixture")
        db.add(d)
        db.flush()
        output = BytesIO()
        writer = PdfWriter()
        for _ in range(3):
            writer.add_blank_page(width=100, height=100)
        writer.write(output)
        raw = output.getvalue()
        db.add(
            DocumentVersion(
                document_id=d.id,
                logical_id=logical.id,
                number=n,
                active=n == 2,
                filename="fixture.pdf",
                raw_content=raw,
                file_hash=hashlib.sha256(raw).hexdigest(),
            )
        )
        c = Chunk(
            document_id=d.id,
            chunk_index=0,
            content="RESTORE_EVIDENCE_MARKER " + str(n),
            embedding=[1.0, 0.0, 0.0],
        )
        db.add(c)
        db.flush()
        db.add(ChunkLocation(chunk_id=c.id, page_number=3))
    for identifier, expired in [
        ("11111111-1111-4111-8111-111111111111", False),
        ("22222222-2222-4222-8222-222222222222", True),
    ]:
        db.add(
            RequestTrace(
                id=identifier,
                user_id=1,
                knowledge_base_id=1,
                created_at=now - timedelta(days=8) if expired else now,
                expires_at=now - timedelta(days=1) if expired else now + timedelta(days=7),
                content_consent=True,
                metadata_json=json.dumps(
                    {
                        "contexts": [
                            {"number": 1, "document_id": 2, "chunk_id": 2, "chunk_index": 0}
                        ],
                        "generation_status": "completed",
                    }
                ),
                snapshot_json=json.dumps(
                    {
                        "question": "synthetic recovery question",
                        "answer": "fixture [1]",
                        "contexts": [{"content": "RESTORE_EVIDENCE_MARKER 2"}],
                    }
                ),
            )
        )
    db.flush()
    db.add(
        RequestFeedback(
            trace_id="22222222-2222-4222-8222-222222222222",
            unhelpful=True,
            unsupported_citation=False,
            outdated=False,
        )
    )
    db.commit()
for owner in (None, 1, 2):
    set_limit(engine, owner, Decimal("20"), "synthetic restore fixture")
a = reserve(engine, 1, "generation", "fixture", {}, Decimal("1"))
settle(engine, a, Decimal("0.1"))
b = reserve(engine, 1, "generation", "fixture", {}, Decimal("2"))
retain_unknown(engine, b, "synthetic unknown")
reserve(engine, 1, "generation", "fixture", {}, Decimal("1"))
