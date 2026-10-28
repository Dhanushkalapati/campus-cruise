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


@dataclass
class ImpactMetrics:
    """Impact figures for one validated plan and its baseline (same model, same inputs).

    Only quantities the optimiser actually computes are included. Fuel,
    carbon and planning time are not modelled, so they are not reported.
    """

    model: str
    status_text: str
    proven_optimal: bool
    schools_served: int
    attendees: int
    baseline_cost: float
    optimised_cost: float
    baseline_buses: int
    optimised_buses: int
    baseline_utilisation: float
    optimised_utilisation: float

    @property
    def saving(self) -> float:
        return self.baseline_cost - self.optimised_cost

    @property
    def saving_percent(self) -> float | None:
        """None when the baseline cost is zero."""
        return None if self.baseline_cost == 0 else self.saving / self.baseline_cost * 100.0

    @property
    def cost_per_student_before(self) -> float | None:
        """None when nobody travels."""
        return None if self.attendees == 0 else self.baseline_cost / self.attendees

    @property
    def cost_per_student_after(self) -> float | None:
        return None if self.attendees == 0 else self.optimised_cost / self.attendees

    @property
    def buses_removed(self) -> int:
        return self.baseline_buses - self.optimised_buses


def impact_metrics(baseline: Solution, optimised: Solution) -> ImpactMetrics:
    """Impact figures; refuses invalid plans and mismatched baselines."""
    if not optimised.has_plan:
        raise ValueError("Impact figures need a validated plan.")
    if not baseline.has_plan:
        raise ValueError("The baseline plan is not valid.")
    if baseline.model != optimised.model:
        raise ValueError("The baseline must use the same cost model as the plan.")
    if baseline.total_attendees != optimised.total_attendees:
        raise ValueError(
            f"Baseline carries {baseline.total_attendees} attendees but the plan carries "
            f"{optimised.total_attendees}: they were built from different inputs."
        )
    return ImpactMetrics(
        model=optimised.model,
        status_text=optimised.status_text(),
        proven_optimal=optimised.proven_optimal,
        schools_served=sum(len(r.stops) for r in optimised.routes),
        attendees=optimised.total_attendees,
        baseline_cost=baseline.total_cost,
        optimised_cost=optimised.total_cost,
        baseline_buses=len(baseline.routes),
        optimised_buses=len(optimised.routes),
        baseline_utilisation=baseline.utilisation,
        optimised_utilisation=optimised.utilisation,
    )


def bus_table(solution: Solution, end_label: str = "Stadium") -> pd.DataFrame:
    """One row per bus. ``end_label`` names the destination at the end of each route."""
    unit = solution.metric_unit
    rows = []
    for r in solution.routes:
        rows.append({
            "Bus": r.bus_id,
            "Bus size": r.bus_size,
            "Schools (pickup order)": " -> ".join(r.stops) + f" -> {end_label}",
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
    e = dataset.event
    frame["Destination"] = e.name
    frame["Destination latitude"] = e.latitude
    frame["Destination longitude"] = e.longitude
    return frame.to_csv(index=False)


def plan_report(
    solution: Solution,
    baseline: Solution,
    dataset: Dataset,
    constraints: Constraints,
    overrides: list[str],
    distance_lines: list[str] | None = None,
) -> str:
    """Markdown report: summary, assumptions, status, results, distance, warnings,
    Responsible AI notes, allocation and shares.

    ``distance_lines``: optional Markdown bullets with bus-distance figures
    (``explain.distance_lines``); kept as text so this module needs no matrices.
    """
    if not solution.has_plan:
        raise ValueError("Only a validated plan can be exported.")
    s = savings(baseline, solution)
    pct = "n/a (baseline cost is zero)" if s.percent is None else f"{s.percent:.1f}%"
    model_rates = dataset.rates[solution.model]
    status_meaning = ("optimality proven: no valid plan costs less under this pricing" if solution.proven_optimal
                      else "valid plan, best found within the time limit; not proven to be the cheapest")
    warnings = ["Costs are in the challenge's cost units, not a real currency or a quote."]
    if solution.model != "fixed":
        warnings.append("Route search is heuristic: a longer or repeated search may find a cheaper plan.")
    if solution.model == "time" and str(solution.matrix_source).startswith(("APPROX", "ML")):
        warnings.append(f"Travel times are estimates ({solution.matrix_source}), not road routing.")
    warnings.append("Distances are straight-line (haversine) miles, not road mileage.")
    lines = [
        "# Campus Cruise transport plan",
        "",
        f"Generated {datetime.now():%Y-%m-%d %H:%M}. Costs are in {CURRENCY_LABEL} (the data does not state a currency).",
        "",
        "## Summary",
        f"- {MODEL_LABELS[solution.model]}: {s.optimised_buses} shared buses cost {s.optimised_cost:,.2f}, against "
        f"{s.baseline_buses} buses and {s.baseline_cost:,.2f} if every school rented its own bus: "
        f"{s.absolute:,.2f} saved ({pct}).",
        f"- Solver status: {solution.status_text()} ({status_meaning}).",
        "- Independent validation: passed. No school is split and no bus is over capacity.",
        "",
        "## Inputs and assumptions",
        f"- Event: {dataset.event.name} ({dataset.event.latitude}, {dataset.event.longitude})",
        f"- Cost model: {MODEL_LABELS[solution.model]}",
        "- Bus rates: " + ", ".join(f"{k}-seat = {v:g}" for k, v in sorted(model_rates.items())),
        f"- Route data: {solution.matrix_source}" if solution.model != "fixed" else "- Route data: not used for cost (fixed model)",
        "- A bus starts at its first pickup and ends at the event venue; no return trip; no dwell time.",
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
        f"- Seat utilisation: {solution.utilisation * 100:.1f}% (baseline {baseline.utilisation * 100:.1f}%)",
        *(distance_lines or []),
        "",
        "## Approximations and warnings",
        *[f"- {w}" for w in warnings],
        "",
        "## Responsible AI and privacy",
        "- The plan comes from the OR-Tools optimiser and passed separate validation code. Any change an AI "
        "assistant proposed was applied only after an organiser approved it.",
        "- Data used: postcode outcodes, one coordinate per outcode and head counts. No names or student records.",
        "- An organiser must review this plan and confirm it with schools and the bus company before booking.",
        "",
        "## Buses",
        bus_table(solution, dataset.event.short_name).to_markdown(index=False),
        "",
        "## School cost shares (estimate, proportional to attendees)",
        cost_shares(solution, dataset).round(2).to_markdown(index=False),
        "",
    ]
    return "\n".join(lines)
