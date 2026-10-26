"""Replay manually reviewed structural checks offline. No .env, DB or model calls.

This checks answer emptiness, citation number range and generation state only;
it is NOT a semantic-support, scientific-correctness or anonymization evaluator.
Optional candidate JSON contains only answer and generation_status.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.services.feedback_cases import (
    MAX_EXPORT_BYTES,
    CandidateOutput,
    RegressionFile,
    regression_checks,
)


def read_bounded(path: str) -> str:
    with Path(path).open("rb") as file:
        raw = file.read(MAX_EXPORT_BYTES + 1)
    if len(raw) > MAX_EXPORT_BYTES:
        raise ValueError("Input too large")
    return raw.decode("utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--candidate")
    args = parser.parse_args()
    try:
        case = RegressionFile.model_validate_json(read_bounded(args.case))
        candidate = (
            CandidateOutput.model_validate_json(read_bounded(args.candidate))
            if args.candidate
            else None
        )
        result = regression_checks(case.case, candidate)
    except Exception:
        print("Invalid regression input; no content or paths logged", file=sys.stderr)
        raise SystemExit(2) from None
    print(json.dumps(result))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
