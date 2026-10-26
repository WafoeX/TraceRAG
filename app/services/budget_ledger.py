"""Atomic cumulative CNY ledger. Explicit Engine only; safe for offline admin CLIs.

Global -> account -> reservation is the lock order in every monetary transition. Held and
unknown reservations never expire automatically, including after process death.
"""

from __future__ import annotations

import json
from decimal import Decimal
from uuid import uuid4

from sqlalchemy.engine import Connection, Engine

from app.services.budget_policy import BudgetError, money


def _keys(user_id: int) -> dict:
    if type(user_id) is not int or user_id <= 0:
        raise BudgetError("budget_owner_missing")
    return {"global": "global", "user": f"user:{user_id}"}


def _accounts(conn: Connection, user_id: int) -> list:
    rows = (
        conn.exec_driver_sql(
            "SELECT * FROM application_budget_accounts WHERE key IN (%(global)s,%(user)s) "
            "ORDER BY key FOR UPDATE",
            _keys(user_id),
        )
        .mappings()
        .all()
    )
    if len(rows) != 2:
        raise BudgetError("budget_not_configured")
    return list(rows)


def _event(conn: Connection, action: str, scope: str, amount, reason: str, reservation_id=None):
    if not isinstance(reason, str) or not reason.strip() or len(reason) > 512:
        raise BudgetError("budget_audit_reason_required")
    conn.exec_driver_sql(
        "INSERT INTO application_budget_events(id,action,scope,reservation_id,amount_cny,reason) "
        "VALUES (%(id)s,%(action)s,%(scope)s,%(reservation)s,%(amount)s,%(reason)s)",
        {
            "id": str(uuid4()),
            "action": action,
            "scope": scope,
            "reservation": reservation_id,
            "amount": amount,
            "reason": reason.strip(),
        },
    )


def set_limit(engine: Engine, user_id: int | None, limit: Decimal, reason: str) -> None:
    limit = money(limit)
    key = "global" if user_id is None else _keys(user_id)["user"]
    with engine.begin() as conn:
        # Absolute limits, never additive top-ups; repeats cannot mint credit.
        conn.exec_driver_sql(
            "INSERT INTO application_budget_accounts(key,user_id,limit_cny) "
            "VALUES (%(key)s,%(user)s,%(limit)s) ON CONFLICT (key) DO NOTHING",
            {"key": key, "user": user_id, "limit": limit},
        )
        row = (
            conn.exec_driver_sql(
                "SELECT * FROM application_budget_accounts WHERE key=%(key)s FOR UPDATE",
                {"key": key},
            )
            .mappings()
            .one()
        )
        if row["spent_cny"] + row["held_cny"] > limit:
            raise BudgetError("budget_limit_below_committed")
        conn.exec_driver_sql(
            "UPDATE application_budget_accounts SET limit_cny=%(limit)s WHERE key=%(key)s",
            {"limit": limit, "key": key},
        )
        _event(conn, "set_limit", key, limit, reason)


def reserve(
    engine: Engine, user_id: int, role: str, model: str, profile: dict, amount: Decimal
) -> str:
    amount = money(amount)
    identifier = str(uuid4())
    with engine.begin() as conn:
        rows = _accounts(conn, user_id)
        for row in rows:
            if row["blocked"]:
                raise BudgetError("budget_blocked")
            if row["spent_cny"] + row["held_cny"] + amount > row["limit_cny"]:
                raise BudgetError("budget_exhausted")
        conn.exec_driver_sql(
            "UPDATE application_budget_accounts SET held_cny=held_cny+%(amount)s "
            "WHERE key IN (%(global)s,%(user)s)",
            {**_keys(user_id), "amount": amount},
        )
        conn.exec_driver_sql(
            "INSERT INTO application_budget_reservations"
            "(id,user_id,role,model,profile_json,reserved_cny,status) "
            "VALUES (%(id)s,%(user)s,%(role)s,%(model)s,%(profile)s,%(amount)s,'held')",
            {
                "id": identifier,
                "user": user_id,
                "role": role,
                "model": model,
                "profile": json.dumps(profile, sort_keys=True),
                "amount": amount,
            },
        )
    # The provider may only be invoked after this commit is acknowledged.
    return identifier


def _reservation(conn: Connection, identifier: str):
    owner = conn.exec_driver_sql(
        "SELECT user_id FROM application_budget_reservations WHERE id=%(id)s", {"id": identifier}
    ).scalar_one_or_none()
    if owner is None:
        raise BudgetError("budget_reservation_missing")
    rows = _accounts(conn, owner)
    item = (
        conn.exec_driver_sql(
            "SELECT * FROM application_budget_reservations WHERE id=%(id)s FOR UPDATE",
            {"id": identifier},
        )
        .mappings()
        .one()
    )
    return rows, item


def retain_unknown(engine: Engine, identifier: str, reason: str, *, block: bool = False):
    with engine.begin() as conn:
        _, item = _reservation(conn, identifier)
        if item["status"] not in {"held", "unknown"}:
            return  # never resurrect a reservation that was already settled
        if item["reason"] == "budget_bound_violation":
            reason, block = "budget_bound_violation", True
        conn.exec_driver_sql(
            "UPDATE application_budget_reservations SET status='unknown',reason=%(reason)s "
            "WHERE id=%(id)s",
            {"id": identifier, "reason": reason},
        )
        if block:
            conn.exec_driver_sql(
                "UPDATE application_budget_accounts SET blocked=true "
                "WHERE key IN (%(global)s,%(user)s)",
                _keys(item["user_id"]),
            )


def _settle(conn: Connection, identifier: str, estimate: Decimal, *, audit_reason=None) -> bool:
    estimate = money(estimate)
    rows, item = _reservation(conn, identifier)
    if item["status"] in {"settled", "released"}:
        if item["estimated_cny"] != estimate:
            raise BudgetError("budget_settlement_conflict")
        return False
    if item["status"] == "unknown" and audit_reason is None:
        raise BudgetError("budget_audit_required")
    if estimate > item["reserved_cny"] and audit_reason is None:
        raise BudgetError("budget_bound_violation")
    for row in rows:
        if row["spent_cny"] + row["held_cny"] - item["reserved_cny"] + estimate > row["limit_cny"]:
            raise BudgetError("budget_exhausted")
    conn.exec_driver_sql(
        "UPDATE application_budget_accounts SET held_cny=held_cny-%(reserved)s, "
        "spent_cny=spent_cny+%(estimate)s WHERE key IN (%(global)s,%(user)s)",
        {**_keys(item["user_id"]), "reserved": item["reserved_cny"], "estimate": estimate},
    )
    status = "released" if estimate == 0 else "settled"
    conn.exec_driver_sql(
        "UPDATE application_budget_reservations SET status=%(status)s,estimated_cny=%(estimate)s, "
        "reason=%(reason)s WHERE id=%(id)s",
        {
            "status": status,
            "estimate": estimate,
            "id": identifier,
            "reason": "admin_audited" if audit_reason else "usage_reported",
        },
    )
    if audit_reason is not None:
        _event(conn, "resolve", _keys(item["user_id"])["user"], estimate, audit_reason, identifier)
    return True


def settle(engine: Engine, identifier: str, estimate: Decimal) -> bool:
    with engine.begin() as conn:
        return _settle(conn, identifier, estimate)


def resolve(engine: Engine, identifier: str, estimate: Decimal, reason: str) -> bool:
    """Admin-only: stop relevant workers, audit billing, then explicitly reconcile."""
    with engine.begin() as conn:
        return _settle(conn, identifier, estimate, audit_reason=reason)


def unblock(engine: Engine, user_id: int | None, reason: str):
    key = "global" if user_id is None else _keys(user_id)["user"]
    with engine.begin() as conn:
        row = conn.exec_driver_sql(
            "SELECT key FROM application_budget_accounts WHERE key=%(key)s FOR UPDATE", {"key": key}
        ).first()
        if row is None:
            raise BudgetError("budget_not_configured")
        pending = conn.exec_driver_sql(
            "SELECT EXISTS(SELECT 1 FROM application_budget_reservations "
            "WHERE status IN ('held','unknown') AND reason='budget_bound_violation' "
            "AND (CAST(%(user)s AS INTEGER) IS NULL OR user_id=%(user)s))",
            {"user": user_id},
        ).scalar_one()
        if pending:
            raise BudgetError("budget_audit_required")
        conn.exec_driver_sql(
            "UPDATE application_budget_accounts SET blocked=false WHERE key=%(key)s", {"key": key}
        )
        _event(conn, "unblock", key, None, reason)


def snapshot(engine: Engine, user_id: int) -> dict:
    with engine.connect() as conn:
        rows = (
            conn.exec_driver_sql(
                "SELECT * FROM application_budget_accounts WHERE key IN (%(global)s,%(user)s)",
                _keys(user_id),
            )
            .mappings()
            .all()
        )
        indexed = {row["key"]: row for row in rows}
        own, shared = indexed.get(_keys(user_id)["user"]), indexed.get("global")
        entries = (
            conn.exec_driver_sql(
                "SELECT id,role,model,status,reserved_cny,estimated_cny,reason,created_at "
                "FROM application_budget_reservations WHERE user_id=%(user)s "
                "ORDER BY created_at DESC,id DESC LIMIT 20",
                {"user": user_id},
            )
            .mappings()
            .all()
        )

        def view(row):
            if row is None:
                return None
            return {
                "limit_cny": str(row["limit_cny"]),
                "estimated_spent_cny": str(row["spent_cny"]),
                "held_cny": str(row["held_cny"]),
                "available_cny": str(row["limit_cny"] - row["spent_cny"] - row["held_cny"]),
                "blocked": row["blocked"],
            }

        shared_status = (
            "not_configured"
            if shared is None
            else "blocked"
            if shared["blocked"]
            else "exhausted"
            if shared["spent_cny"] + shared["held_cny"] >= shared["limit_cny"]
            else "available"
        )
        return {
            "currency": "CNY",
            "amount_kind": "estimate",
            "actual_cost_cny": None,
            "account": view(own),
            "shared_status": shared_status,
            "reservations": [
                {
                    **dict(r),
                    "reserved_cny": str(r["reserved_cny"]),
                    "estimated_cny": str(r["estimated_cny"])
                    if r["estimated_cny"] is not None
                    else None,
                    "created_at": r["created_at"].isoformat(),
                }
                for r in entries
            ],
        }
