"""Reproducible benchmark on the supplied data: one solve per cost model.

    python benchmark.py                      # default limits (fixed 20 s, routing 300 s)
    python benchmark.py --time-limit 60      # same limit for every model
    python benchmark.py --models distance    # subset
    python benchmark.py --offline            # no OSRM request (cached or labelled estimate)

Writes demo_output/benchmark.json and demo_output/benchmark.md with the
status, runtime, costs and bus counts of every run. The figures are the
results of that run on that machine: routing is heuristic, so another run
can give a slightly different plan.
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime
from pathlib import Path

from src.data import COST_MODELS, DEFAULT_TIME_LIMITS, MODEL_LABELS, load_dataset
from src.model import Constraints
from src.distances import straight_line_matrix
from src.explain import distance_impact
from src.planner import optimise, points_of, time_matrix_for
from src.report import impact_metrics


def run(models: list[str], time_limit: float | None, offline: bool) -> list[dict]:
    data = load_dataset()
    # Travel times: used for the time model's cost, and for every model's driving-hours figure.
    tm = time_matrix_for(data, allow_online=not offline)
    miles = straight_line_matrix(points_of(data))
    rows = []
    for model in models:
        if model not in data.available_models:
            rows.append({"model": model, "status": "UNAVAILABLE", "notes": ["cost file missing or invalid"]})
            continue
        limit = time_limit if time_limit is not None else DEFAULT_TIME_LIMITS[model]
        print(f"Running {MODEL_LABELS[model]} with a {limit:g} s limit...", flush=True)
        result = optimise(data, model, Constraints(), limit, tm)
        sol, base = result.solution, result.baseline
        row = {
            "model": model,
            "label": MODEL_LABELS[model],
            "status": sol.status,
            "status_text": sol.status_text(),
            "time_limit_s": sol.time_limit_s,
            "runtime_s": round(sol.runtime_s, 1),
            "route_data": sol.matrix_source if model != "fixed" else "not used for cost",
            "notes": sol.notes + (tm.notes if model == "time" and tm else []),
        }
        if sol.has_plan:
            im = impact_metrics(base, sol)
            row.update({
                "attendees": im.attendees,
                "schools_served": im.schools_served,
                "optimised_cost": round(im.optimised_cost, 2),
                "baseline_cost": round(im.baseline_cost, 2),
                "saving": round(im.saving, 2),
                "saving_percent": None if im.saving_percent is None else round(im.saving_percent, 1),
                "optimised_buses": im.optimised_buses,
                "baseline_buses": im.baseline_buses,
                "cost_per_student_before": round(im.cost_per_student_before, 2),
                "cost_per_student_after": round(im.cost_per_student_after, 2),
                "optimised_utilisation_percent": round(im.optimised_utilisation * 100, 1),
                "baseline_utilisation_percent": round(im.baseline_utilisation * 100, 1),
            })
            di = distance_impact(sol, base, data, miles, tm.hours, tm.source)
            row.update({
                "baseline_bus_miles": round(di.baseline_miles, 1),
                "plan_bus_miles": round(di.plan_miles, 1),
                "bus_miles_saved": round(di.miles_saved, 1),
                "bus_miles_saved_percent": None if di.miles_saved_percent is None
                else round(di.miles_saved_percent, 1),
                "baseline_bus_hours": round(di.baseline_hours, 2),
                "plan_bus_hours": round(di.plan_hours, 2),
                "bus_hours_saved": round(di.hours_saved, 2),
                "hours_source": di.hours_source,
            })
        rows.append(row)
        print(f"  {row['status_text']}: {row.get('optimised_cost', '-')} vs baseline "
              f"{row.get('baseline_cost', '-')} in {row['runtime_s']} s", flush=True)
    return rows


def markdown(rows: list[dict], started: str) -> str:
    lines = [
        "# Campus Cruise benchmark",
        "",
        f"Run started {started} on {platform.system()} {platform.release()}, Python {platform.python_version()}.",
        "Supplied data, no attendance edits, no organiser constraints. Costs in cost units.",
        "Baseline: every school rents its own cheapest bus, priced with the same model's rates.",
        "",
        "| Model | Status | Limit (s) | Runtime (s) | Optimised | Buses | Baseline | Buses | Saving | Cost/attendee before -> after |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if "optimised_cost" not in r:
            lines.append(f"| {r.get('label', r['model'])} | {r.get('status_text', r['status'])} | "
                         f"{r.get('time_limit_s', '-')} | {r.get('runtime_s', '-')} | - | - | - | - | - | - |")
            continue
        pct = "n/a" if r["saving_percent"] is None else f"{r['saving_percent']}%"
        lines.append(
            f"| {r['label']} | {r['status_text']} | {r['time_limit_s']:g} | {r['runtime_s']} | "
            f"{r['optimised_cost']:,.2f} | {r['optimised_buses']} | {r['baseline_cost']:,.2f} | "
            f"{r['baseline_buses']} | {r['saving']:,.2f} ({pct}) | "
            f"{r['cost_per_student_before']:.2f} -> {r['cost_per_student_after']:.2f} |"
        )
    lines += ["", "Bus distance and driving time (summed over buses, first pickup to stadium; miles are",
              "straight-line, not road mileage; hours from the source named; no fuel or emissions estimate):", "",
              "| Model | Bus-miles baseline -> plan | Saved | Bus-hours baseline -> plan | Saved | Hours source |",
              "|---|---|---|---|---|---|"]
    for r in rows:
        if "plan_bus_miles" in r:
            mp = "n/a" if r["bus_miles_saved_percent"] is None else f"{r['bus_miles_saved_percent']}%"
            lines.append(f"| {r['label']} | {r['baseline_bus_miles']:,.1f} -> {r['plan_bus_miles']:,.1f} | "
                         f"{r['bus_miles_saved']:,.1f} ({mp}) | {r['baseline_bus_hours']:,.2f} -> "
                         f"{r['plan_bus_hours']:,.2f} | {r['bus_hours_saved']:,.2f} | {r['hours_source']} |")
    lines += ["", "Route data per model:"]
    lines += [f"- {r.get('label', r['model'])}: {r.get('route_data', '-')}" for r in rows]
    lines += ["", "Routing results are heuristic (valid, not proven cheapest). A rerun can differ slightly.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--time-limit", type=float, default=None,
                        help="seconds per model; default: DEFAULT_TIME_LIMITS in src/data.py")
    parser.add_argument("--models", nargs="+", choices=COST_MODELS, default=list(COST_MODELS))
    parser.add_argument("--offline", action="store_true", help="no OSRM request")
    args = parser.parse_args()

    started = f"{datetime.now():%Y-%m-%d %H:%M}"
    rows = run(args.models, args.time_limit, args.offline)
    out = Path(__file__).resolve().parent / "demo_output"
    out.mkdir(exist_ok=True)
    (out / "benchmark.json").write_text(json.dumps({"started": started, "runs": rows}, indent=2), encoding="utf-8")
    (out / "benchmark.md").write_text(markdown(rows, started), encoding="utf-8")
    print(f"\nWrote {out / 'benchmark.md'} and {out / 'benchmark.json'}")


if __name__ == "__main__":
    main()
