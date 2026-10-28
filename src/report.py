"""Reporting: savings, school cost shares, plan comparison and exports.

Cost sharing rule (estimate, not an invoice): each bus's cost is split
between its schools in proportion to their attendees,

    share = bus cost x school attendees / bus attendees.

For per-mile and per-hour buses the whole route cost is split this way too,
so a school does not pay more just because it was picked up first.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from .data import CURRENCY_LABEL, MODEL_LABELS, Dataset
from .model import Constraints, Solution


@dataclass
class Savings:
    baseline_cost: float
    optimised_cost: float
    baseline_buses: int
    optimised_buses: int

    @property
    def absolute(self) -> float:
        return self.baseline_cost - self.optimised_cost

    @property
    def percent(self) -> float | None:
        """Savings percentage; None when the baseline cost is zero."""
        if self.baseline_cost == 0:
            return None
        return self.absolute / self.baseline_cost * 100.0


def savings(baseline: Solution, optimised: Solution) -> Savings:
    return Savings(baseline.total_cost, optimised.total_cost, len(baseline.routes), len(optimised.routes))


def bus_table(solution: Solution) -> pd.DataFrame:
    unit = solution.metric_unit
    rows = []
    for r in solution.routes:
        rows.append({
            "Bus": r.bus_id,
            "Bus size": r.bus_size,
            "Schools (pickup order)": " -> ".join(r.stops) + " -> Stadium",
            "Schools": len(r.stops),
            "Attendees": r.attendees,
            "Empty seats": r.empty_seats,
            "Utilisation %": round(r.utilisation * 100, 1),
            f"Route ({unit})": round(r.metric, 2),
            "Cost": round(r.cost, 2),
        })
    return pd.DataFrame(rows)


def cost_shares(solution: Solution, dataset: Dataset) -> pd.DataFrame:
    """One row per transported school. Shares sum to the bus cost exactly
    before rounding; displayed values are rounded to 2 decimals."""
    attendees = dict(zip(dataset.schools["school_id"], dataset.schools["attendees"]))
    names = dict(zip(dataset.schools["school_id"], dataset.schools["name"]))
    rows = []
    for r in solution.routes:
        for stop_no, sid in enumerate(r.stops, start=1):
            a = int(attendees[sid])
            share = r.cost * a / r.attendees if r.attendees else 0.0
            rows.append({
                "School": names[sid],
                "Attendees": a,
                "Bus": r.bus_id,
                "Pickup #": stop_no,
                "Shared with": ", ".join(s for s in r.stops if s != sid) or "-",
                "Bus cost": r.cost,
                "Cost share": share,
                "Cost per attendee": share / a if a else 0.0,
            })
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values("School").reset_index(drop=True)
    return frame


def groupings(solution: Solution) -> dict[str, frozenset[str]]:
    """school -> set of schools on the same bus (including itself)."""
    return {s: frozenset(r.stops) for r in solution.routes for s in r.stops}


def compare_plans(before: Solution, after: Solution) -> pd.DataFrame:
    """Schools whose bus-mates, bus size or pickup order changed."""
    g1, g2 = groupings(before), groupings(after)
    route1 = {s: r for r in before.routes for s in r.stops}
    route2 = {s: r for r in after.routes for s in r.stops}
    rows = []
    for sid in sorted(set(g1) | set(g2)):
        r1, r2 = route1.get(sid), route2.get(sid)
        if r1 and r2 and g1[sid] == g2[sid] and r1.bus_size == r2.bus_size and r1.stops == r2.stops:
            continue
        reasons = []
        if not r1:
            reasons.append("newly transported")
        elif not r2:
            reasons.append("no longer transported")
        else:
            if g1[sid] != g2[sid]:
                reasons.append("different bus-mates")
            if r1.bus_size != r2.bus_size:
                reasons.append(f"bus size {r1.bus_size} -> {r2.bus_size}")
            if g1[sid] == g2[sid] and r1.stops != r2.stops:
                reasons.append("pickup order changed")
        rows.append({
            "School": sid,
            "Before": " -> ".join(r1.stops) + f" ({r1.bus_size}-seat)" if r1 else "-",
            "After": " -> ".join(r2.stops) + f" ({r2.bus_size}-seat)" if r2 else "-",
            "Change": ", ".join(reasons),
        })
    return pd.DataFrame(rows, columns=["School", "Before", "After", "Change"])


def attendance_changes(before: Dataset, after: Dataset) -> list[str]:
    a = dict(zip(before.schools["school_id"], before.schools["attendees"]))
    b = dict(zip(after.schools["school_id"], after.schools["attendees"]))
    return [f"{s}: {a[s]} -> {b[s]}" for s in a if a[s] != b.get(s)]


def plan_csv(solution: Solution, dataset: Dataset) -> str:
    """CSV export, one row per school. Refuses invalid plans."""
    if not solution.has_plan:
        raise ValueError("Only a validated plan can be exported.")
    frame = cost_shares(solution, dataset)
    sizes = {r.bus_id: r.bus_size for r in solution.routes}
    frame.insert(3, "Bus size", frame["Bus"].map(sizes))
    for col in ("Bus cost", "Cost share", "Cost per attendee"):
        frame[col] = frame[col].round(2)
    return frame.to_csv(index=False)


def plan_report(
    solution: Solution,
    baseline: Solution,
    dataset: Dataset,
    constraints: Constraints,
    overrides: list[str],
) -> str:
    """Markdown report with assumptions, status, allocation, savings and shares."""
    if not solution.has_plan:
        raise ValueError("Only a validated plan can be exported.")
    s = savings(baseline, solution)
    pct = "n/a (baseline cost is zero)" if s.percent is None else f"{s.percent:.1f}%"
    model_rates = dataset.rates[solution.model]
    lines = [
        "# Campus Cruise transport plan",
        "",
        f"Generated {datetime.now():%Y-%m-%d %H:%M}. Costs are in {CURRENCY_LABEL} (the data does not state a currency).",
        "",
        "## Inputs and assumptions",
        f"- Event: {dataset.event.name} ({dataset.event.latitude}, {dataset.event.longitude})",
        f"- Cost model: {MODEL_LABELS[solution.model]}",
        "- Bus rates: " + ", ".join(f"{k}-seat = {v:g}" for k, v in sorted(model_rates.items())),
        f"- Route data: {solution.matrix_source}" if solution.model != "fixed" else "- Route data: not used for cost (fixed model)",
        "- A bus starts at its first pickup and ends at the stadium; no return trip; no dwell time.",
        "- A school is never split between buses.",
        f"- Schools transported: {len({x for r in solution.routes for x in r.stops})}; attendees: {solution.total_attendees}",
        "- Attendance changes: " + ("; ".join(overrides) if overrides else "none"),
        "- Organiser constraints: " + ("; ".join(constraints.describe()) if not constraints.is_empty() else "none"),
        "",
        "## Solver",
        f"- Method: {solution.method}",
        f"- Status: {solution.status_text()}",
        f"- Runtime: {solution.runtime_s:.1f} s (limit {solution.time_limit_s:g} s)",
        *[f"- {n}" for n in solution.notes],
        "- Independent validation: passed (capacity, no split schools, all schools served, costs recalculated)",
        "",
        "## Result",
        f"- Optimised cost: {s.optimised_cost:.2f} with {s.optimised_buses} buses",
        f"- Baseline (one bus per school): {s.baseline_cost:.2f} with {s.baseline_buses} buses",
        f"- Savings: {s.absolute:.2f} ({pct})",
        f"- Seat utilisation: {solution.utilisation * 100:.1f}%",
        "",
        "## Buses",
        bus_table(solution).to_markdown(index=False),
        "",
        "## School cost shares (estimate, proportional to attendees)",
        cost_shares(solution, dataset).round(2).to_markdown(index=False),
        "",
    ]
    return "\n".join(lines)
