"""Administrator-only account creation; explicit DB URL, no public registration."""

from __future__ import annotations

import argparse
import getpass
import sys

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from app.security import hash_password, normalize_username


def create_user(engine: Engine, username: str, password: str) -> int:
    username, encoded = normalize_username(username), hash_password(password)
    with engine.begin() as conn:
        return conn.exec_driver_sql(
            "INSERT INTO app_users(username,password_hash,created_at) "
            "VALUES (%(u)s,%(p)s,now()) RETURNING id",
            {"u": username, "p": encoded},
        ).scalar_one()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", required=True)
    parser.add_argument("--username", required=True)
    parser.add_argument("--password-stdin", action="store_true")
    args = parser.parse_args()
    engine = None
    try:
        password = (
            sys.stdin.readline().rstrip("\r\n")
            if args.password_stdin
            else getpass.getpass("New password: ")
        )
        engine = create_engine(
            args.database_url.replace("postgresql://", "postgresql+psycopg://", 1)
        )
        create_user(engine, args.username, password)
        print("Account created")
    except Exception:
        raise SystemExit(
            "Account creation failed; check username, password policy and database safely"
        ) from None
    finally:
        if engine is not None:
            engine.dispose()


if __name__ == "__main__":
    main()
