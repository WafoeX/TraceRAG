"""Offline administrator CLI: explicit DB URL, cumulative CNY estimate limits.

No public budget mutations, automatic reset, implicit .env or provider calls.
Audit reconciliation requires stopped relevant workers and explicit confirmation.
"""

from __future__ import annotations

import argparse
import json

from sqlalchemy import create_engine

from app.security import normalize_username
from app.services import budget_ledger as ledger
from app.services.budget_policy import BudgetError, money


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("set-limit", "show", "unblock"):
        sub = commands.add_parser(command)
        scope = sub.add_mutually_exclusive_group(required=True)
        scope.add_argument("--global", dest="global_scope", action="store_true")
        scope.add_argument("--username")
        if command == "set-limit":
            sub.add_argument("--cny", required=True)
        if command != "show":
            sub.add_argument("--reason", required=True)
    reconcile = commands.add_parser("resolve")
    reconcile.add_argument("--reservation", required=True)
    reconcile.add_argument("--cny", required=True)
    reconcile.add_argument("--reason", required=True)
    reconcile.add_argument("--confirm-audited", action="store_true", required=True)
    args = parser.parse_args()
    engine = None
    try:
        engine = create_engine(
            args.database_url.replace("postgresql://", "postgresql+psycopg://", 1)
        )
        if args.command == "resolve":
            changed = ledger.resolve(engine, args.reservation, money(args.cny), args.reason)
            print("Reservation reconciled" if changed else "Already reconciled (same estimate)")
            return
        user_id = None
        if not args.global_scope:
            with engine.connect() as conn:
                user_id = conn.exec_driver_sql(
                    "SELECT id FROM app_users WHERE username=%(name)s",
                    {"name": normalize_username(args.username)},
                ).scalar_one_or_none()
            if user_id is None:
                raise BudgetError("budget_user_missing")
        if args.command == "set-limit":
            ledger.set_limit(engine, user_id, money(args.cny), args.reason)
            print("Cumulative CNY estimate limit set (not billed cost)")
        elif args.command == "unblock":
            ledger.unblock(engine, user_id, args.reason)
            print("Budget unblocked after explicit audit")
        elif user_id is not None:
            print(json.dumps(ledger.snapshot(engine, user_id), ensure_ascii=False))
        else:
            with engine.connect() as conn:
                row = (
                    conn.exec_driver_sql(
                        "SELECT * FROM application_budget_accounts WHERE key='global'"
                    )
                    .mappings()
                    .one_or_none()
                )
            print(
                json.dumps(
                    {
                        "currency": "CNY",
                        "amount_kind": "estimate",
                        "actual_cost_cny": None,
                        "global": dict(row) if row else None,
                    },
                    default=str,
                )
            )
    except BudgetError as exc:
        raise SystemExit(str(exc)) from None
    except Exception:
        raise SystemExit(
            "Budget command failed or outcome unknown; inspect ledger safely before retrying"
        ) from None
    finally:
        if engine is not None:
            engine.dispose()


if __name__ == "__main__":
    main()
