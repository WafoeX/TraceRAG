"""Synthetic disposable deployment fixture; executed only by opt-in drill."""

from datetime import UTC, datetime, timedelta

from app.database import SessionLocal
from app.models import IngestionJob

with SessionLocal() as db:
    for i, state in enumerate(("queued", "embedding", "completed"), 1):
        db.add(
            IngestionJob(
                knowledge_base_id=1,
                file_hash="restore-" + str(i),
                dedup_key="kb:1:new:restore-" + str(i),
                title="Synthetic pending",
                filename="fixture.txt",
                payload=None if state == "completed" else b"RESTORE_PENDING_BYTES",
                state=state,
                attempts=1,
                lease_token="old-lease" if state == "embedding" else None,
                lease_until=datetime.now(UTC) + timedelta(hours=1)
                if state == "embedding"
                else None,
                document_id=2 if state == "completed" else None,
            )
        )
    db.commit()
