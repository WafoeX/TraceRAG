"""Owner-only budget display; no HTTP credit, limit or reconciliation mutations."""

from fastapi import APIRouter, Depends, HTTPException

from app.auth import current_user
from app.models import User
from app.services.application_budget import budget_engine
from app.services.budget_ledger import snapshot

router = APIRouter(prefix="/budget", tags=["budget"])


@router.get("")
def budget(user: User = Depends(current_user)):
    try:
        return snapshot(budget_engine(), user.id)
    except Exception:
        raise HTTPException(503, "Budget ledger unavailable") from None
