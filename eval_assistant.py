"""Live evaluation of the AI Transport Planning Assistant against the real model.

    python eval_assistant.py

Uses the same key as the app: OPENAI_API_KEY or ANTHROPIC_API_KEY from the
environment or .streamlit/secrets.toml. Sends the 30 requests in
tests/assistant_eval_cases.py to the configured model (one call
each, about 30 short requests in total; this costs a small amount of API
credit) and scores, per case:

    type      the model chose the expected request type
    outcome   the app offered Approve exactly when it should
    unsafe    the app offered Approve when it should not (must be 0)

Results go to demo_output/assistant_eval.json. Nothing in the app is changed:
proposals are validated and discarded, never applied.
"""

from __future__ import annotations

import json
import sys
import tomllib
from datetime import datetime
from pathlib import Path

from src.assistant import AssistantUnavailable, assist, build_context, provider_from_config
from src.data import load_dataset
from src.model import Constraints
from src.planner import optimise
from tests.assistant_eval_cases import CASES


def main() -> int:
    secrets_file = Path(__file__).resolve().parent / ".streamlit" / "secrets.toml"
    secrets = tomllib.loads(secrets_file.read_text(encoding="utf-8")) if secrets_file.exists() else {}
    try:
        provider = provider_from_config(secrets)
    except AssistantUnavailable as exc:
        print(f"Cannot run the live evaluation: {exc}")
        return 2
    data = load_dataset()
    print("Solving the fixed model once for the context (20 s)...", flush=True)
    fixed = optimise(data, "fixed", Constraints(), 20)
    runs = {"fixed": {"solution": fixed.solution, "baseline": fixed.baseline, "current_inputs": True}}

    rows = []
    for k, case in enumerate(CASES, start=1):
        rules = case.get("rules") or {}
        constraints = Constraints(list(rules.get("must_share", [])), list(rules.get("cannot_share", [])),
                                  list(rules.get("solo", [])))
        context = build_context(data, constraints, runs, "fixed", fixed.solution)
        p = assist(case["request"], provider, data, constraints, context, "fixed")
        row = {
            "request": case["request"],
            "expected_type": case["expected"],
            "got_type": p.request_type,
            "type_ok": p.request_type == case["expected"],
            "approve_expected": case["approve"],
            "approve_offered": p.needs_approval,
            "outcome_ok": p.needs_approval == case["approve"],
            "unsafe": p.needs_approval and not case["approve"],
            "errors": p.errors,
            "answer": p.answer,
        }
        rows.append(row)
        flag = "ok " if row["type_ok"] and row["outcome_ok"] else "MISS"
        print(f"{k:2d}. {flag} {case['request'][:55]:55s} -> {p.request_type}"
              + (" [approve]" if p.needs_approval else ""), flush=True)

    n = len(rows)
    summary = {
        "run": f"{datetime.now():%Y-%m-%d %H:%M}",
        "model": provider.label,
        "cases": n,
        "type_correct": sum(r["type_ok"] for r in rows),
        "outcome_correct": sum(r["outcome_ok"] for r in rows),
        "unsafe_approvals": sum(r["unsafe"] for r in rows),
        "provider_errors": sum(1 for r in rows if any("language model" in e for e in r["errors"])),
    }
    print(f"\nRequest type correct: {summary['type_correct']}/{n}")
    print(f"Approve offered exactly when expected: {summary['outcome_correct']}/{n}")
    print(f"Unsafe approvals: {summary['unsafe_approvals']}")
    out = Path(__file__).resolve().parent / "demo_output"
    out.mkdir(exist_ok=True)
    (out / "assistant_eval.json").write_text(json.dumps({"summary": summary, "cases": rows}, indent=2),
                                            encoding="utf-8")
    print(f"Wrote {out / 'assistant_eval.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
