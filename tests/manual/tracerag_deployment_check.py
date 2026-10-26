"""Opt-in real Compose/pg_dump recovery drill. Fixed disposable project only.

Default plan never invokes Docker. --apply uses fake credentials/unverified prices,
no model calls, and cleans only resources created in this dedicated project.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time
import traceback
import urllib.error
import urllib.request
from http.cookiejar import CookieJar
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROJECT = "tracerag-deploy-test"
SOURCE = "tracerag_application"
TARGET = "tracerag_restore_test"
LIVE_TRACE = "11111111-1111-4111-8111-111111111111"
EXPIRED_TRACE = "22222222-2222-4222-8222-222222222222"
PW = "Deployment-demo-only-123"
URL = "http://127.0.0.1:18091"

SEED = (ROOT / "tests/fixtures/deployment_seed.py").read_text()
PENDING = (ROOT / "tests/fixtures/deployment_pending.py").read_text()
FINGERPRINT = (ROOT / "tests/fixtures/deployment_fingerprint.py").read_text()
SAFETY = (ROOT / "tests/fixtures/deployment_safety.py").read_text()


class Drill:
    def __init__(self, folder: Path):
        self.folder = folder
        self.envfile = folder / "application.env"
        self.restored = folder / "restored.env"
        self.backups = folder / "backups"
        self.backups.mkdir(mode=0o700)
        values = (
            "TRACERAG_DB_PASSWORD=synthetic_db_password_only\n"
            "TRACERAG_MODEL_KEY=offline-fixture-no-provider\n"
            "TRACERAG_HTTP_PORT=18091\nTRACERAG_EMBEDDING_DIM=3\n"
            "TRACERAG_COOKIE_SECURE=false\n"
            f"TRACERAG_BACKUP_DIR={self.backups}\n"
        )
        self.envfile.write_text(values + f"TRACERAG_DB_NAME={SOURCE}\n")
        self.restored.write_text(values + f"TRACERAG_DB_NAME={TARGET}\n")
        self.env = {k: v for k, v in os.environ.items() if not k.startswith("TRACERAG_")}
        self.started = False

    def run(
        self, args: list[str], *, restored: bool = False, text: str | None = None, ok: bool = True
    ):
        command = [
            "docker",
            "compose",
            "--env-file",
            str(self.restored if restored else self.envfile),
            "-p",
            PROJECT,
            "-f",
            "compose.application.yml",
            *args,
        ]
        result = subprocess.run(
            command, cwd=ROOT, env=self.env, input=text, text=True, capture_output=True
        )
        if ok and result.returncode:
            raise RuntimeError("Disposable Compose step failed; raw output withheld")
        return result

    def python(self, code: str, *, restored: bool = False):
        result = self.run(
            [
                "run",
                "--rm",
                "--no-deps",
                "-T",
                "--entrypoint",
                "/app/.venv/bin/python",
                "tools",
                "-",
            ],
            text=code,
            restored=restored,
        )
        return result.stdout.strip()

    def sql(self, statement: str):
        return self.run(
            [
                "exec",
                "-T",
                "db",
                "psql",
                "-X",
                "-Atq",
                "-v",
                "ON_ERROR_STOP=1",
                "-U",
                "tracerag",
                "-d",
                "postgres",
                "-c",
                statement,
            ]
        ).stdout.strip()

    def backup(
        self, operation: str, name: str, database: str, *, ok: bool = True, trust: bool = True
    ):
        args = [
            "run",
            "--rm",
            "-T",
            "-e",
            "PGDATABASE=" + database,
            "db-tools",
            operation,
            name,
            "--confirm-services-stopped",
            "--confirm-database",
            database,
        ]
        if operation == "restore" and trust:
            args.append("--trust-local-backup")
        return self.run(args, ok=ok)


def http(path: str, *, client=None, payload=None, method: str = "GET", csrf: str | None = None):
    headers = {"Content-Type": "application/json"}
    if csrf:
        headers["X-CSRF-Token"] = csrf
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(URL + path, data=data, headers=headers, method=method)
    try:
        with (client or urllib.request.build_opener()).open(request, timeout=10) as response:
            return response.status, response.read(), response.headers
    except urllib.error.HTTPError as response:
        return response.code, response.read(), response.headers


def login(name: str):
    client = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(CookieJar()))
    status, body, _ = http(
        "/auth/login", client=client, method="POST", payload={"username": name, "password": PW}
    )
    assert status == 200
    return client, json.loads(body)["csrf_token"]


def wait_ready(expected: int = 200):
    for _ in range(40):
        try:
            if http("/ready")[0] == expected:
                return
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(1)
    raise RuntimeError("Readiness did not reach expected state")


def wait_exited(drill: Drill, services: list[str], *, restored: bool = False):
    # Cold Python/import startup is not guaranteed to finish in two seconds.
    for _ in range(30):
        state = drill.run(["ps", "-a", "--format", "json", *services], restored=restored).stdout
        if state.replace(" ", "").count('"State":"exited"') == len(services):
            return
        time.sleep(1)
    raise RuntimeError("Services did not reject the unsafe database")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if not args.apply:
        print(
            "PLAN ONLY: disposable Compose deployment + empty-target trusted archive recovery; "
            "no real DB/models"
        )
        return
    report = {}
    with tempfile.TemporaryDirectory(prefix="tracerag-p33-") as temporary:
        drill = Drill(Path(temporary))
        # Never remove a pre-existing project/volume, even if it uses this test name.
        for listing in (["volume", "ls", "-q"], ["network", "ls", "-q"], ["ps", "-aq"]):
            existing = subprocess.run(
                ["docker", *listing, "--filter", "label=com.docker.compose.project=" + PROJECT],
                capture_output=True,
                text=True,
            )
            if existing.returncode or existing.stdout.strip():
                raise SystemExit(
                    "Disposable project already has resources; inspect manually, nothing removed"
                )
        try:
            drill.started = True
            drill.run(["up", "-d", "--wait", "db"])
            # Real API cannot start against an empty schema (no startup DDL).
            drill.run(["up", "-d", "api"], ok=False)
            wait_exited(drill, ["api"])
            report["empty_schema_refused"] = True
            drill.run(
                [
                    "run",
                    "--rm",
                    "-T",
                    "tools",
                    "init",
                    "--apply",
                    "--database-url",
                    f"postgresql+psycopg://tracerag@db:5432/{SOURCE}",
                    "--embedding-dim",
                    "3",
                    "--confirm-database",
                    SOURCE,
                    "--confirm-services-stopped",
                ]
            )
            drill.python(SEED)
            drill.run(["up", "-d", "--wait", "api", "worker"])
            wait_ready()
            assert http("/health")[0] == 200
            alice, csrf = login("alice")
            bob, _ = login("bob")
            assert http("/documents/2/chunks/0", client=bob)[0] == 404
            assert http("/documents/2/chunks/0", client=alice)[0] == 200
            original = http("/documents/2/original", client=alice)[1]
            assert original.startswith(b"%PDF")
            assert (
                http(
                    f"/traces/{LIVE_TRACE}/feedback",
                    client=alice,
                    method="PUT",
                    csrf=csrf,
                    payload={"categories": ["unsupported_citation"], "citation_number": 1},
                )[0]
                == 200
            )
            # Actual worker executes a new job but fails at UNVERIFIED pricing before wire.
            status, body, _ = http(
                "/documents/jobs?knowledge_base_id=1",
                client=alice,
                method="POST",
                csrf=csrf,
                payload={
                    "title": "Budget closed",
                    "content": "Synthetic job with no paid embedding",
                },
            )
            assert status == 202
            job_id = json.loads(body)["id"]
            for _ in range(40):
                row = json.loads(http(f"/documents/jobs/{job_id}", client=alice)[1])
                if row["state"] == "failed":
                    assert row["error_code"] == "budget_pricing_unverified"
                    break
                time.sleep(0.5)
            else:
                raise RuntimeError("Worker did not fail closed")
            assert (
                drill.python(
                    "from app.database import engine\nwith engine.connect() as c:\n"
                    " print(c.exec_driver_sql('SELECT count(*) FROM "
                    "application_budget_reservations').scalar_one())"
                )
                == "3"
            )
            report.update(
                api_worker_db=True,
                owner_isolation=True,
                original_pdf=True,
                worker_budget_fail_closed=True,
            )
            drill.run(["stop", "db"])
            assert http("/health")[0] == 200
            wait_ready(503)
            drill.run(["up", "-d", "--wait", "db"])
            wait_ready()
            report["database_outage_readiness_503_liveness_200"] = True
            drill.run(["stop", "api", "worker"])
            drill.python(PENDING)
            before = json.loads(drill.python(FINGERPRINT))
            drill.backup("backup", "good.dump", SOURCE)
            assert drill.backup("backup", "good.dump", SOURCE, ok=False).returncode != 0
            mode = drill.run(
                [
                    "run",
                    "--rm",
                    "-T",
                    "--entrypoint",
                    "bash",
                    "db-tools",
                    "-c",
                    "stat -c %a /backups/good.dump",
                ]
            ).stdout.strip()
            assert mode == "600"
            drill.sql(f"CREATE DATABASE {TARGET}")
            # Hash mismatch and malformed-but-hash-matching archive must not write target.
            drill.run(
                [
                    "run",
                    "--rm",
                    "-T",
                    "--entrypoint",
                    "bash",
                    "db-tools",
                    "-c",
                    "cp /backups/good.dump /backups/tampered.dump; "
                    "cp /backups/good.dump.sha256 /backups/tampered.dump.sha256; "
                    "printf changed >> /backups/tampered.dump; "
                    "printf PRIVATE_INVALID_ARCHIVE > /backups/invalid.dump; "
                    "sha256sum /backups/invalid.dump | cut -d' ' -f1 "
                    "> /backups/invalid.dump.sha256",
                ]
            )
            for name in ("tampered.dump", "invalid.dump"):
                result = drill.backup("restore", name, TARGET, ok=False)
                assert (
                    result.returncode
                    and "PRIVATE_INVALID_ARCHIVE" not in result.stdout + result.stderr
                )
            assert drill.backup("restore", "good.dump", SOURCE, ok=False).returncode != 0
            assert (
                drill.backup("restore", "good.dump", TARGET, ok=False, trust=False).returncode != 0
            )
            assert json.loads(drill.python(FINGERPRINT)) == before
            # Force quarantine SQL failure AFTER archive SQL, proving one-transaction rollback.
            drill.python(
                "from app.database import engine\nwith engine.begin() as c:\n"
                " c.exec_driver_sql('ALTER TABLE ingestion_jobs RENAME state TO broken_state')"
            )
            drill.backup("backup", "broken-schema.dump", SOURCE)
            drill.python(
                "from app.database import engine\nwith engine.begin() as c:\n"
                " c.exec_driver_sql('ALTER TABLE ingestion_jobs RENAME broken_state TO state')"
            )
            assert drill.backup("restore", "broken-schema.dump", TARGET, ok=False).returncode != 0
            count = drill.run(
                [
                    "exec",
                    "-T",
                    "db",
                    "psql",
                    "-X",
                    "-Atq",
                    "-U",
                    "tracerag",
                    "-d",
                    TARGET,
                    "-c",
                    "SELECT count(*) FROM pg_tables WHERE schemaname='public'",
                ]
            ).stdout.strip()
            assert count == "0"
            report.update(
                no_overwrite=True,
                archive_hash_verified=True,
                corrupt_archive_refused=True,
                transactional_restore_rollback=True,
                backup_mode_600=True,
            )
            drill.backup("restore", "good.dump", TARGET)
            assert json.loads(drill.python(FINGERPRINT, restored=True)) == before
            safety = json.loads(
                drill.python(
                    SAFETY,
                    restored=True,
                )
            )
            assert safety == {
                "guard": True,
                "sessions": 0,
                "blocked": True,
                "pending": 0,
                "expired": 0,
                "expired_feedback": 0,
            }
            drill.run(["up", "-d", "--no-deps", "api", "worker"], restored=True, ok=False)
            wait_exited(drill, ["api", "worker"], restored=True)
            report.update(
                data_vector_binary_page_budget_trace_feedback_fingerprints=True,
                sessions_revoked=True,
                budget_holds_preserved=True,
                jobs_paused=True,
                expiry_cascaded=True,
                quarantined_api_worker_refused=True,
            )
            # Synthetic drill review only: never a claim of real provider reconciliation.
            drill.run(
                [
                    "run",
                    "--rm",
                    "-T",
                    "tools",
                    "release",
                    "--apply",
                    "--database-url",
                    f"postgresql+psycopg://tracerag@db:5432/{TARGET}",
                    "--embedding-dim",
                    "3",
                    "--confirm-database",
                    TARGET,
                    "--confirm-services-stopped",
                    "--confirm-privacy-reviewed",
                    "--confirm-billing-reviewed",
                ],
                restored=True,
            )
            drill.run(["up", "-d", "--no-deps", "--wait", "api", "worker"], restored=True)
            wait_ready()
            assert http("/auth/me", client=alice)[0] == 401
            restored_alice, _ = login("alice")
            restored_bob, _ = login("bob")
            assert http("/documents/2/chunks/0", client=restored_bob)[0] == 404
            assert http("/documents/2/original", client=restored_alice)[1] == original
            assert http(f"/traces/{LIVE_TRACE}/feedback", client=restored_alice)[0] == 200
            assert http(f"/traces/{EXPIRED_TRACE}", client=restored_alice)[0] == 404
            budget = json.loads(http("/budget", client=restored_alice)[1])
            assert budget["account"]["blocked"]
            assert json.loads(drill.python(FINGERPRINT, restored=True)) == before
            report.update(
                manual_review_required=True,
                post_review_login_and_owner_isolation=True,
                original_bytes_restored=True,
                review_did_not_unblock_budget_or_retry=True,
            )
            result = {
                "fixture": (
                    "synthetic data, 3D vectors, no provider calls; not real billing reconciliation"
                ),
                "checks": report,
            }
            (ROOT / "docs/assets/tracerag-deployment-validation.json").write_text(
                json.dumps(result, indent=2) + "\n"
            )
            print(json.dumps(result))
        except Exception as exc:
            positions = [
                f.lineno for f in traceback.extract_tb(exc.__traceback__) if f.filename == __file__
            ]
            raise SystemExit(
                f"Disposable deployment drill failed at source lines {positions}; "
                "raw logs/credentials withheld"
            ) from None
        finally:
            if drill.started:
                cleanup = drill.run(["down", "-v", "--remove-orphans"], ok=False)
                if cleanup.returncode:
                    raise SystemExit("Disposable cleanup failed; inspect dedicated test project")


if __name__ == "__main__":
    main()
