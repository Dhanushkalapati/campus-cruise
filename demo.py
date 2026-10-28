"""Command-line demonstration on the real dataset.

Shows, for each cost model: the optimised plan vs the one-bus-per-school
baseline, then an attendance change and the re-optimised cost difference.

    python demo.py [--time-limit 30] [--offline]

Outputs CSV plans and Markdown reports to demo_output/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data import MODEL_LABELS, load_dataset
from src.model import Constraints
from src.planner import optimise, time_matrix_for
from src.report import compare_plans, plan_csv, plan_report, savings

# Demonstration attendance change (applied to a copy; the CSV is untouched).
DEMO_CHANGE = {"CF15": 25, "SA18": 20}


def show(title, result):
    sol, base = result.solution, result.baseline
    print(f"\n== {title}")
    print(f"   status: {sol.status_text()}  ({sol.method}, {sol.runtime_s:.1f} s of {sol.time_limit_s:g} s)")
    if not sol.has_plan:
        print("   no valid plan:", "; ".join(sol.notes + sol.violations))
        return
    sv = savings(base, sol)
    pct = "n/a" if sv.percent is None else f"{sv.percent:.1f}%"
    print(f"   optimised: {sol.total_cost:,.2f} with {len(sol.routes)} buses, utilisation {sol.utilisation:.1%}")
    print(f"   baseline:  {base.total_cost:,.2f} with {len(base.routes)} buses, utilisation {base.utilisation:.1%}")
    print(f"   savings:   {sv.absolute:,.2f} ({pct})")
    if sol.model != "fixed":
        print(f"   route data: {sol.matrix_source}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--time-limit", type=float, default=30.0)
    parser.add_argument("--offline", action="store_true", help="skip OSRM; use the labelled estimate")
    args = parser.parse_args()

    out = Path(__file__).resolve().parent / "demo_output"
    out.mkdir(exist_ok=True)
    data = load_dataset()
    print(f"Loaded {len(data.schools)} schools, {data.schools.attendees.sum()} attendees.")
    changed = data.with_attendance(DEMO_CHANGE)
    tm = time_matrix_for(data, allow_online=not args.offline)

    for model in data.available_models:
        first = optimise(data, model, Constraints(), args.time_limit, tm)
        show(f"{MODEL_LABELS[model]}: original attendance", first)
        second = optimise(changed, model, Constraints(), args.time_limit, tm, warm_start=first.solution)
        show(f"{MODEL_LABELS[model]}: after change {DEMO_CHANGE} (warm start from previous plan)", second)
        if first.solution.has_plan and second.solution.has_plan:
            diff = second.solution.total_cost - first.solution.total_cost
            moved = compare_plans(first.solution, second.solution)
            print(f"   cost difference after change: {diff:+,.2f}; schools with a changed bus: {len(moved)}")
            (out / f"{model}_plan.csv").write_text(plan_csv(first.solution, data))
            (out / f"{model}_report.md").write_text(
                plan_report(first.solution, first.baseline, data, Constraints(), []), encoding="utf-8")
    print(f"\nPlans and reports written to {out}")


if __name__ == "__main__":
    main()
