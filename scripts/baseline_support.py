"""Safety and artifact helpers for isolated, opt-in baseline work (no model imports)."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from decimal import Decimal
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlsplit

DEFAULT_DATABASE_URL = (
    "postgresql+psycopg://budgetrag_baseline:baseline-local-only@127.0.0.1:55432/budgetrag_baseline"
)
GUARD_MARKER = "budgetrag-isolated-baseline-v1"
ROOT = Path(__file__).resolve().parents[1]


class BaselineError(RuntimeError):
    """Controlled error whose message never contains credentials or provider error bodies."""


def validate_database_url(value: str) -> None:
    try:
        parsed = urlsplit(value)
        valid = (
            parsed.scheme == "postgresql+psycopg"
            and parsed.hostname in {"127.0.0.1", "localhost"}
            and parsed.port == 55432
            and parsed.path == "/budgetrag_baseline"
            and parsed.username == "budgetrag_baseline"
            and not parsed.query
            and not parsed.fragment
        )
    except ValueError as exc:
        raise BaselineError("Invalid isolated database URL") from exc
    if not valid:
        raise BaselineError("Only the dedicated localhost:55432/budgetrag_baseline is allowed")


def prepare_schema(database_url: str, schema: str) -> str:
    """Check an existing DB marker before any DDL; only create a fresh run schema."""
    validate_database_url(database_url)
    if not re.fullmatch(r"(?:baseline|infra)_[0-9a-f]{32}", schema):
        raise BaselineError("Invalid isolated run schema")
    from sqlalchemy import create_engine, text
    from sqlalchemy.engine import make_url
    from sqlalchemy.schema import CreateSchema

    engine = create_engine(database_url, connect_args={"connect_timeout": 5})
    try:
        with engine.begin() as connection:
            marker = connection.execute(
                text("SELECT marker FROM public.budgetrag_baseline_guard WHERE singleton = :flag"),
                {"flag": True},
            ).scalar_one()
            if marker != GUARD_MARKER:
                raise BaselineError("Isolated database guard mismatch; no DDL performed")
            connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public"), {})
            # SQLAlchemy quotes this allowlisted identifier. No IF NOT EXISTS:
            # collision/reuse must fail instead of mixing datasets.
            connection.execute(CreateSchema(schema), {})
    except BaselineError:
        raise
    except Exception:
        raise BaselineError(
            "Cannot prepare isolated PostgreSQL schema; no model calls made"
        ) from None
    finally:
        engine.dispose()
    return (
        make_url(database_url)
        .update_query_dict({"options": f"-csearch_path={schema},public"})
        .render_as_string(hide_password=False)
    )


def existing_schema_url(database_url: str, schema: str) -> str:
    """Read-only reopen path: no DDL, no arbitrary search_path/URL parameters."""
    validate_database_url(database_url)
    if not re.fullmatch(r"infra_[0-9a-f]{32}", schema):
        raise BaselineError("Only an existing infrastructure schema may be reopened")
    return database_url + "?" + urlencode({"options": f"-csearch_path={schema},public"})


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def corpus_manifest(root: Path = ROOT) -> list[dict[str, Any]]:
    frozen = read_json(root / "docs/refactor/baseline/source-manifest.json")
    expected = {
        entry["path"]: entry["sha256"]
        for entry in frozen["files"]
        if entry["path"].startswith("examples/")
    }
    paths = [
        path
        for path in sorted((root / "examples").iterdir())
        if not path.name.startswith("_") and path.suffix.lower() in {".txt", ".md", ".pdf"}
    ]
    if not expected or {p.relative_to(root).as_posix() for p in paths} != set(expected):
        raise BaselineError("Baseline corpus file set changed; create a separate dataset version")
    records = []
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise BaselineError("Baseline corpus must contain regular files, not symlinks")
        relative = path.relative_to(root).as_posix()
        actual = digest(path)
        if actual != expected[relative]:
            raise BaselineError("Baseline corpus hash differs from frozen source snapshot")
        records.append({"path": relative, "sha256": actual, "bytes": path.stat().st_size})
    return records


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise BaselineError("Cannot read valid JSON artifact/configuration") from exc


def runtime_manifest(root: Path = ROOT, *, allow_adapted: bool = False) -> list[dict[str, str]]:
    original = read_json(root / "docs/refactor/baseline/source-manifest.json")
    records = []
    for entry in original["files"]:
        if not entry["path"].startswith(("app/", "api/")):
            continue
        actual = digest(root / entry["path"])
        if actual != entry["sha256"] and not allow_adapted:
            raise BaselineError("Runtime source differs from pre-refactor snapshot")
        record = {"path": entry["path"], "sha256": actual}
        if allow_adapted:
            record["original_sha256"] = entry["sha256"]
        records.append(record)
    return records


def load_cases(path: Path) -> list[dict[str, str]]:
    cases = read_json(path)
    ids: set[str] = set()
    if not isinstance(cases, list) or not cases:
        raise BaselineError("Cases must be a non-empty list")
    for case in cases:
        if not isinstance(case, dict) or set(case) != {"id", "question", "answer"}:
            raise BaselineError("Invalid case fields")
        if any(not isinstance(value, str) or not value.strip() for value in case.values()):
            raise BaselineError("Case values must be non-empty strings")
        if case["id"] in ids:
            raise BaselineError("Duplicate case ID")
        ids.add(case["id"])
    return cases


def summarize(records: list[dict[str, Any]], k: int) -> dict[str, Any]:
    """Phrase-match smoke statistics, explicitly not qrels Recall@K/NDCG."""
    successful = [row for row in records if row["status"] == "ok"]
    ranks = [row["answer_rank"] for row in successful]
    latencies = sorted(row["elapsed_ms"] for row in successful)
    n = len(successful)

    def quantile(q: float) -> float | None:
        if not latencies:
            return None
        position = (len(latencies) - 1) * q
        lo, hi = math.floor(position), math.ceil(position)
        return latencies[lo] + (latencies[hi] - latencies[lo]) * (position - lo)

    return {
        "k": k,
        "queries": len(records),
        "successful_queries": n,
        "failed_queries": len(records) - n,
        "phrase_hit_rate_at_k": sum(rank > 0 for rank in ranks) / n if n else None,
        "phrase_mrr_at_k": sum(1 / rank for rank in ranks if rank) / n if n else None,
        "latency_ms_p50": quantile(0.5),
        "latency_ms_p95": quantile(0.95),
        "quantile_method": "linear (n-1)*q; successful requests only",
        "metric_scope": "16-case phrase-match smoke; not formal qrels recall/ndcg",
    }


class CostLedger:
    """Reserve a conservative per-request cost before a call; unknown usage fails closed.

    Bounds/rates must be supplied and verified for the selected model by its operator.
    This is an API-price estimate, not an independent guarantee of provider billing.
    """

    def __init__(self, limit: Decimal, prices: Mapping[str, Any]):
        if not limit.is_finite() or not Decimal(0) < limit <= Decimal(1):
            raise BaselineError("Explicit budget must be positive and at most USD 1")
        self.limit = limit
        self.spent = Decimal(0)
        self.prices = prices
        self.blocked = False
        self.stop_reason: str | None = None
        self.costs: list[dict[str, Any]] = []
        for kind in ("embedding", "rerank"):
            profile = prices[kind]
            for key in ("input_per_million_usd", "output_per_million_usd"):
                rate = Decimal(str(profile[key]))
                if not rate.is_finite() or rate < 0:
                    raise BaselineError("Prices must be finite and non-negative")
            for key in ("input_token_upper_bound", "output_token_upper_bound"):
                bound = profile[key]
                if type(bound) is not int or bound < 0:
                    raise BaselineError("Request token bounds must be non-negative integers")
            if not profile["input_token_upper_bound"]:
                raise BaselineError("Input token upper bound must be positive")
        self.reserved: Decimal | None = None

    def _price(self, kind: str, input_tokens: int, output_tokens: int) -> Decimal:
        p = self.prices[kind]
        return (
            Decimal(input_tokens) * Decimal(str(p["input_per_million_usd"]))
            + Decimal(output_tokens) * Decimal(str(p["output_per_million_usd"]))
        ) / 1_000_000

    def reserve(self, kind: str) -> Decimal:
        if self.blocked or self.stop_reason is not None or self.reserved is not None:
            raise BaselineError("Unknown usage, stop condition or pending request prevents calls")
        p = self.prices[kind]
        cost = self._price(kind, p["input_token_upper_bound"], p["output_token_upper_bound"])
        if self.spent + cost > self.limit:
            self.stop_reason = "budget_reservation_exceeded"
            raise BaselineError("Request reservation exceeds remaining budget; call not made")
        self.reserved = cost
        return cost

    def settle(self, kind: str, usage: Mapping[str, Any] | None) -> None:
        if self.reserved is None:
            raise BaselineError("Missing request reservation")
        keys = ("prompt_tokens", "completion_tokens") if kind == "rerank" else ("total_tokens",)
        if usage is None or any(type(usage.get(key)) is not int or usage[key] < 0 for key in keys):
            self.blocked = True
            raise BaselineError("Usage unavailable/invalid; no further paid calls allowed")
        input_tokens = usage[keys[0]]
        output_tokens = usage["completion_tokens"] if kind == "rerank" else 0
        p = self.prices[kind]
        if (
            input_tokens > p["input_token_upper_bound"]
            or output_tokens > p["output_token_upper_bound"]
        ):
            self.blocked = True
            raise BaselineError(
                "Actual usage exceeded declared model bounds; stop and audit billing"
            )
        cost = self._price(kind, input_tokens, output_tokens)
        self.spent += cost
        self.costs.append({"kind": kind, "usage": dict(usage), "estimated_cost_usd": str(cost)})
        self.reserved = None
