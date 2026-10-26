"""Pure export schemas and deterministic checks: no settings, DB or provider IO.

Manual review is an attestation, NOT a guarantee that text is anonymous.
"""

from __future__ import annotations

import json
import math
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

MAX_EXPORT_BYTES = 1024 * 1024
CATEGORIES = ("unhelpful", "unsupported_citation", "outdated")
Category = Literal["unhelpful", "unsupported_citation", "outdated"]
GenerationStatus = Literal["completed", "failed", "skipped"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FeedbackInput(StrictModel):
    categories: list[Category] = Field(min_length=1, max_length=3)
    citation_number: int | None = Field(default=None, ge=1, le=20, strict=True)

    @model_validator(mode="after")
    def unique_categories(self):
        if len(set(self.categories)) != len(self.categories):
            raise ValueError("Duplicate feedback categories")
        self.categories = [c for c in CATEGORIES if c in self.categories]
        return self


class Expectations(StrictModel):
    citation_range_valid: bool = Field(strict=True)
    answer_nonempty: bool = Field(strict=True)
    generation_status: GenerationStatus


class ReviewedCase(StrictModel):
    question: str = Field(min_length=1, max_length=4000)
    answer: str = Field(max_length=65536)
    contexts: list[str] = Field(max_length=20)
    observed_generation_status: GenerationStatus
    expected: Expectations
    expected_semantics: str = Field(min_length=1, max_length=2000)
    reviewed: bool = Field(strict=True)

    @model_validator(mode="after")
    def reviewed_and_bounded(self):
        if not self.reviewed or not self.question.strip() or not self.expected_semantics.strip():
            raise ValueError("Manual review required")
        if any(not text.strip() or len(text) > 32768 for text in self.contexts):
            raise ValueError("Invalid context")
        if len(self.model_dump_json().encode()) > MAX_EXPORT_BYTES - 16384:
            raise ValueError("Case too large")
        return self


class ExportInput(StrictModel):
    mode: Literal["metadata", "regression"] = "metadata"
    reviewed_case: ReviewedCase | None = None

    @model_validator(mode="after")
    def matching_mode(self):
        if (self.mode == "regression") != (self.reviewed_case is not None):
            raise ValueError("Invalid export mode")
        return self


class RegressionFile(StrictModel):
    schema_version: Literal["tracerag-regression-v1"]
    feedback: FeedbackInput
    case: ReviewedCase


class CandidateOutput(StrictModel):
    answer: str = Field(max_length=65536)
    generation_status: GenerationStatus


def regression_checks(case: ReviewedCase, candidate: CandidateOutput | None = None) -> dict:
    """Replay structural checks only. Never pretend to judge semantic support."""
    answer = candidate.answer if candidate else case.answer
    status = candidate.generation_status if candidate else case.observed_generation_status
    numbers = re.findall(r"\[([0-9]+)\]", answer)
    valid = all(len(n) <= 100 and 1 <= int(n) <= len(case.contexts) for n in numbers)
    checks = {
        "citation_range_valid": valid == case.expected.citation_range_valid,
        "answer_nonempty": bool(answer.strip()) == case.expected.answer_nonempty,
        "generation_status": status == case.expected.generation_status,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "manual_semantic_review_required": True,
    }


def metadata_export(metadata: dict, feedback: FeedbackInput) -> dict:
    """Reconstruct a small allowlist; never copy source IDs, models or content."""
    config = metadata.get("config")
    config = config if isinstance(config, dict) else {}
    safe_config = {}
    for key in ("use_hybrid", "use_rerank"):
        if type(config.get(key)) is bool:
            safe_config[key] = config[key]
    for key, cap in (("top_k", 20), ("retrieval_candidates", 10000)):
        value = config.get(key)
        if type(value) is int and 1 <= value <= cap:
            safe_config[key] = value
    for key, allowed in (
        ("rerank_prompt_version", {"baseline", "keyed_v1", "relevance_v1"}),
        ("rerank_passage_mode", {"prefix_600", "full_chunk"}),
    ):
        value = config.get(key)
        if isinstance(value, str) and value in allowed:
            safe_config[key] = value
    value = config.get("min_relevance")
    if (
        isinstance(value, float | int)
        and not isinstance(value, bool)
        and 0 <= value <= 1
        and math.isfinite(value)
    ):
        safe_config["min_relevance"] = value
    result = {
        "schema_version": "tracerag-feedback-metadata-v1",
        "feedback": feedback.model_dump(),
        "run": {"config": safe_config},
    }
    status = metadata.get("generation_status")
    if isinstance(status, str) and status in {"completed", "failed", "skipped"}:
        result["run"]["generation_status"] = status
    code = metadata.get("http_status")
    if type(code) is int and 100 <= code <= 599:
        result["run"]["http_status"] = code
    contexts = metadata.get("contexts")
    if isinstance(contexts, list) and len(contexts) <= 20:
        result["run"]["context_count"] = len(contexts)
    return result


def parse_metadata(raw: str) -> dict:
    try:
        value = json.loads(raw)
    except (ValueError, TypeError):
        raise ValueError("Trace metadata unavailable") from None
    if not isinstance(value, dict):
        raise ValueError("Trace metadata unavailable")
    return value
