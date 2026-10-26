"""Independent durable ingestion worker: python -m app.worker.

Provision the schema first. Stop with SIGINT/SIGTERM; an interrupted job is
reclaimed after its lease expires. Never run this against evaluation fixtures.
"""

from __future__ import annotations

import logging
import signal
import threading

from sqlalchemy import select

from app import database
from app.config import get_settings
from app.models import KnowledgeBase
from app.services.application_schema import check_guard, ready
from app.services.ingestion import ClaimedJob, claim, process
from app.services.resilience import application_calls

logger = logging.getLogger(__name__)


def process_owned_job(item: ClaimedJob) -> bool:
    """Resolve the trusted queue library's current owner; never charge a browser hint."""
    with database.SessionLocal() as db:
        check_guard(db)
        owner_id = db.scalar(
            select(KnowledgeBase.owner_id).where(KnowledgeBase.id == item.knowledge_base_id)
        )
    if owner_id is None:
        raise ValueError("Task owner missing")
    with application_calls(user_id=owner_id):
        return process(database.SessionLocal, item)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    settings = get_settings()
    if settings.application_startup_check and not ready(database.engine, settings.embedding_dim):
        raise RuntimeError("Worker database startup failed; inspect readiness/migrations safely")
    stopping = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stopping.set())
    while not stopping.is_set():
        try:
            with database.SessionLocal() as db:
                check_guard(db)
                item = claim(db, require_owned=True)
            if item is None:
                stopping.wait(2)
                continue
            success = process_owned_job(item)
            logger.info("ingestion job_id=%s published=%s", item.id, success)
        except Exception:
            # Avoid logging exception bodies that can contain connection secrets.
            logger.error("Worker operation failed; retrying after backoff")
            stopping.wait(5)


if __name__ == "__main__":
    main()
