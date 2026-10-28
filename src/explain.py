"""Explainability: plan summaries, data-grounded explanations, audit report.

Every statement is computed from a validated solution, its baseline, the
input data and the rates. Each explanation line is tagged:

    "fact"            computed from the plan and data (re-checkable)
    "interpretation"  a reading of the facts, not a proven cause

The solver does not report why it chose a plan, so causal claims about
individual decisions are never made as facts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from .data import CURRENCY_LABEL, MODEL_LABELS, Dataset
from .model import Constraints, Solution
from .report import bus_table, cost_shares, impact_metrics, savings
from .solvers import seat_cover_lower_bound
from .validation import route_metric, validate_solution

OBJECTIVES = {
    "fixed": "Minimise total rental cost (fixed price per bus; location does not affect cost)",
    "distance": "Minimise total cost = rate per mile x straight-line miles of each route",
    "time": "Minimise total cost = rate per hour x driving hours of each route",
}


def _index(ds: Dataset) -> dict[str, int]:
    return {sid: i for i, sid in enumerate(ds.schools["school_id"])}


def plan_totals(sol: Solution, ds: Dataset, miles: np.ndarray, hours: np.ndarray | None) -> dict:
    """Straight-line miles and (if a time matrix is given) driving hours of every route, summed.

    ``miles`` and ``hours`` cover the dataset's schools in order, then the event.
    """
    idx = _index(ds)
    total_miles = sum(route_metric([idx[s] for s in r.stops], miles) for r in sol.routes)
    total_hours = None if hours is None else sum(route_metric([idx[s] for s in r.stops], hours)
                                                 for r in sol.routes)
    return {"miles": total_miles, "hours": total_hours}


@dataclass
class DistanceImpact:
    """Bus-distance and driving-time totals of a plan and its baseline.

    Both totals are summed over the same route rule (first pickup, each stop,
    event venue) from the same matrices, so they are comparable. Miles are
    straight-line (haversine), not road miles. Hours come from ``hours_source``
    and are None when no travel-time data is available.
    """

    baseline_miles: float
    plan_miles: float
    baseline_hours: float | None = None
    plan_hours: float | None = None
    hours_source: str | None = None

    @property
    def miles_saved(self) -> float:
        return self.baseline_miles - self.plan_miles

    @property
    def miles_saved_percent(self) -> float | None:
        return safe_percent(self.miles_saved, self.baseline_miles)

    @property
    def hours_saved(self) -> float | None:
        if self.baseline_hours is None or self.plan_hours is None:
            return None
        return self.baseline_hours - self.plan_hours

    @property
    def hours_saved_percent(self) -> float | None:
        saved = self.hours_saved
        return None if saved is None else safe_percent(saved, self.baseline_hours)


def distance_impact(sol: Solution, base: Solution, ds: Dataset, miles: np.ndarray,
                    hours: np.ndarray | None = None, hours_source: str | None = None) -> DistanceImpact:
    """Bus-miles (and driving hours, if ``hours`` is given) of the plan against its baseline.

    Bus-distance saved = baseline total bus-miles - plan total bus-miles.
    Refuses plans without a validated result or with a baseline from other inputs.
    """
    impact_metrics(base, sol)  # same checks: validated plan, same model, same attendees
    a, b = plan_totals(base, ds, miles, hours), plan_totals(sol, ds, miles, hours)
    return DistanceImpact(a["miles"], b["miles"], a["hours"], b["hours"],
                          hours_source if hours is not None else None)


def summary_row(entry: dict, current: bool, miles: np.ndarray, hours: np.ndarray | None,
                hours_source: str | None) -> dict:
    """One row of the all-models plan summary. ``entry`` is a stored run."""
    res, ds = entry["result"], entry["dataset"]
    sol, base = res.solution, res.baseline
    row = {
        "Model": MODEL_LABELS[entry["model"]],
        "Current inputs": "yes" if current else "no (out of date)",
        "Status": sol.status_text(),
        "Proven optimal": "yes" if sol.proven_optimal else "no",
        "Runtime (s)": round(sol.runtime_s, 1),
        "Limit (s)": sol.time_limit_s,
        "Travel-time data": "not used for cost" if entry["model"] == "fixed" else sol.matrix_source,
        "Approximate data": "yes" if entry["model"] != "fixed" and sol.matrix_source.startswith(
            ("APPROX", "ML")) else "no",
    }
    if not sol.has_plan or not base.has_plan:
        return row
    im = impact_metrics(base, sol)
    totals = plan_totals(sol, ds, miles, hours)
    di = distance_impact(sol, base, ds, miles, hours, hours_source)
    row.update({
        "Baseline cost": round(im.baseline_cost, 2),
        "Optimised cost": round(im.optimised_cost, 2),
        "Change": round(-im.saving, 2),
        "Change %": None if im.saving_percent is None else round(-im.saving_percent, 1),
        "Buses before": im.baseline_buses,
        "Buses after": im.optimised_buses,
        "Attendees": im.attendees,
        "Seats": sol.total_seats,
        "Utilisation %": round(im.optimised_utilisation * 100, 1),
        "Straight-line miles": round(totals["miles"], 1),
        "Baseline miles": round(di.baseline_miles, 1),
        "Bus-miles saved": round(di.miles_saved, 1),
        "Driving hours": None if totals["hours"] is None else round(totals["hours"], 2),
        "Baseline hours": None if di.baseline_hours is None else round(di.baseline_hours, 2),
        "Hours source": hours_source if totals["hours"] is not None else "not available",
    })
    return row


def explain_plan(sol: Solution, base: Solution, ds: Dataset, constraints: Constraints,
                 matrix: np.ndarray) -> list[tuple[str, str]]:
    """(kind, text) lines explaining a validated plan. ``matrix``: the model's own route metric
    (miles, or hours for the time model; miles for fixed, display only)."""
    if not sol.has_plan:
        return [("fact", f"No validated plan to explain: {sol.status_text()}.")]
    model = sol.model
    rates = ds.rates[model]
    sizes = sorted(rates)
    idx = _index(ds)
    att = dict(zip(ds.schools["school_id"], ds.schools["attendees"]))
    out: list[tuple[str, str]] = []

    out.append(("fact", f"Objective: {OBJECTIVES[model]}."))
    # Bus sizes: is each bus the cheapest size that seats its schools, for this route?
    def unit(size, metric):
        return rates[size] if model == "fixed" else rates[size] * metric

    not_cheapest = []
    for r in sol.routes:
        fits = [s for s in sizes if s >= r.attendees]
        best = min(fits, key=lambda s: (unit(s, r.metric), s))
        if unit(r.bus_size, r.metric) > unit(best, r.metric) + 1e-9:
            not_cheapest.append(f"{r.bus_id} ({r.bus_size}-seat, {best}-seat would fit)")
    if not_cheapest:
        out.append(("fact", "Buses not on the cheapest size that fits their load: "
                    + "; ".join(not_cheapest) + "."))
    else:
        out.append(("fact", "Every bus uses the cheapest available size that seats its schools."))

    mix = pd.Series([r.bus_size for r in sol.routes]).value_counts().sort_index()
    out.append(("fact", "Bus mix: " + ", ".join(f"{n} x {s}-seat" for s, n in mix.items())
                + f"; {sol.total_attendees} attendees on {sol.total_seats} seats "
                f"({sol.utilisation * 100:.1f}% full, {sol.total_seats - sol.total_attendees} empty seats)."))
    if model == "fixed":
        per_seat = ", ".join(f"{s}-seat {rates[s] / s:.2f}" for s in sizes)
        out.append(("fact", f"Price per seat: {per_seat} {CURRENCY_LABEL}. Larger buses are cheaper per seat "
                    "here, so filling large buses lowers cost."))
        lb = seat_cover_lower_bound(sol.total_attendees, rates)
        if sol.total_cost <= lb + 1e-6:
            out.append(("fact", f"Cost {sol.total_cost:,.2f} equals the seat-cover lower bound {lb:,.2f} (the "
                        "cheapest set of buses with enough seats for everyone). No plan can cost less."))
        else:
            out.append(("fact", f"Seat-cover lower bound: {lb:,.2f}. The plan costs {sol.total_cost:,.2f}; "
                        + ("CP-SAT proved no cheaper plan exists (keeping schools whole forces the gap)."
                           if sol.proven_optimal else "optimality is not proven, so a cheaper plan may exist.")))
        out.append(("interpretation", "Location does not change the fixed price, so groupings are driven by "
                    "how school sizes fill buses. Among equally cheap plans the solver's second stage "
                    "prefers nearby schools on the same bus."))
    else:
        # Sharing saving per bus versus each of its schools on its own cheapest direct bus.
        event = matrix.shape[0] - 1
        alone_total = 0.0
        shared = [r for r in sol.routes if len(r.stops) > 1]
        for r in shared:
            for s in r.stops:
                m = float(matrix[idx[s], event])
                alone_total += min(rates[t] * m for t in sizes if t >= att[s])
        shared_cost = sum(r.cost for r in shared)
        unit_name = "hours" if model == "time" else "miles"
        out.append(("fact", f"{len(shared)} buses carry more than one school. Those schools would cost "
                    f"{alone_total:,.2f} on their own direct buses; sharing costs {shared_cost:,.2f}."))
        out.append(("fact", f"Route {unit_name} are summed from first pickup to the event venue; "
                    "no cost before the first pickup and no return trip (challenge rules)."))
        out.append(("interpretation", "Cost grows with every extra leg, so the search tends to group schools "
                    "that are close to each other or on the way to the event venue, and can use more, smaller "
                    "buses than the fixed model when detours would cost more than the seats saved."))
        out.append(("fact", f"Status: {sol.status_text()}. The routing search is heuristic; "
                    "a different or longer search may find a cheaper plan."))
        if sol.matrix_source.startswith(("APPROX", "ML")):
            out.append(("fact", f"Travel times used: {sol.matrix_source}. These are estimates, not road routing."))

    if not constraints.is_empty():
        out.append(("fact", "Organiser rules applied as hard constraints and re-checked by the validator: "
                    + "; ".join(constraints.describe()) + "."))
        out.append(("interpretation", "Whether these rules raised the cost is not measured here. "
                    "Run once without them and compare to see their effect."))
    sv = savings(base, sol)
    pct = "n/a (zero baseline)" if sv.percent is None else f"{sv.percent:.1f}%"
    out.append(("fact", f"Against one bus per school ({sv.baseline_buses} buses, {sv.baseline_cost:,.2f}): "
                f"{sv.absolute:,.2f} saved ({pct}) with {sv.optimised_buses} buses."))
    return out


def model_difference(a: Solution, b: Solution) -> list[str]:
    """Plain facts on how two plans (possibly different models) group the same schools."""
    ga = {frozenset(r.stops) for r in a.routes}
    gb = {frozenset(r.stops) for r in b.routes}
    same = len(ga & gb)
    return [
        f"{MODEL_LABELS[a.model]}: {len(a.routes)} buses; {MODEL_LABELS[b.model]}: {len(b.routes)} buses.",
        f"{same} bus groupings are identical in both plans; {len(ga - gb)} differ.",
        "Costs are in different units of the challenge's price tables and are not comparable.",
    ]


# --------------------------------------------------------------------------- #
# Audit report
# --------------------------------------------------------------------------- #


def distance_lines(di: DistanceImpact | None) -> list[str]:
    """Markdown bullet lines for bus-distance and driving-time figures (shared by both reports)."""
    if di is None:
        return ["- Bus distance: not available for this run."]
    pct = "n/a" if di.miles_saved_percent is None else f"{di.miles_saved_percent:.1f}%"
    out = [f"- Bus distance (straight-line miles, first pickup to venue, summed over buses): "
           f"{di.baseline_miles:,.1f} (one bus per school) -> {di.plan_miles:,.1f} (plan); "
           f"{di.miles_saved:,.1f} bus-miles saved ({pct}). Straight-line, not road mileage."]
    if di.hours_saved is not None:
        hp = "n/a" if di.hours_saved_percent is None else f"{di.hours_saved_percent:.1f}%"
        out.append(f"- Driving time ({di.hours_source}): {di.baseline_hours:,.2f} h -> {di.plan_hours:,.2f} h "
                   f"summed over buses; {di.hours_saved:,.2f} bus-hours saved ({hp}). No dwell time at stops.")
    else:
        out.append("- Driving time: no travel-time data available for this run.")
    out.append("- Fuel use and emissions are not estimated: the data has no vehicle or fuel information.")
    return out


def audit_report(entry: dict, base_dataset: Dataset, run_log: list[dict], summary: pd.DataFrame,
                 matrix: np.ndarray, current: bool, distance: DistanceImpact | None = None) -> str:
    """Markdown audit report for one stored run. Refuses runs without a validated plan.

    The plan is re-validated here, so the report states the result of a fresh check.
    """
    res, ds = entry["result"], entry["dataset"]
    sol, base = res.solution, res.baseline
    if not sol.has_plan:
        raise ValueError("Only a validated plan can be reported.")
    check_matrix = None if sol.model == "fixed" else matrix
    violations = validate_solution(sol, ds.schools, ds.rates[sol.model], entry["constraints"], check_matrix)
    im = impact_metrics(base, sol)
    pct = "n/a (zero baseline)" if im.saving_percent is None else f"{im.saving_percent:.1f}%"
    edits = [f"{s}: {a} -> {b}" for s, a, b in zip(
        base_dataset.schools["school_id"], base_dataset.schools["attendees"], ds.schools["attendees"]) if a != b]
    rates = ds.rates[sol.model]
    lines = [
        "# Campus Cruise audit report",
        "",
        f"Generated {datetime.now():%Y-%m-%d %H:%M}. Costs in {CURRENCY_LABEL} (the data states no currency).",
        "" if current else "\n> **Out of date:** the app's inputs changed after this run. "
                           "These figures describe the run below, not the current inputs.\n",
        "## 1. Executive summary",
        f"- {MODEL_LABELS[sol.model]}: {im.optimised_buses} buses costing {im.optimised_cost:,.2f}, against "
        f"{im.baseline_buses} buses costing {im.baseline_cost:,.2f} with one bus per school: "
        f"{im.saving:,.2f} saved ({pct}).",
        f"- Solver status: **{sol.status_text()}**"
        + (" (optimality proven)." if sol.proven_optimal else " (no optimality proof)."),
        f"- Independent validation: {'passed' if not violations else 'FAILED: ' + '; '.join(violations)}.",
        "",
        "## 2. Dataset and scenario",
        f"- Event: {ds.event.name} ({ds.event.latitude}, {ds.event.longitude}).",
        f"- {len(ds.schools)} pickup points identified by postcode outcode; {int(ds.schools['attendees'].sum())} "
        "attendees; no student-level records.",
        "- Attendance edits in this run: " + ("; ".join(edits) if edits else "none") + ".",
        "- Organiser rules: " + ("; ".join(entry["constraints"].describe()) or "none") + ".",
        f"- Run #{entry['run_no']} ({entry['trigger']}), finished {entry['finished_at']}; time limit "
        f"{entry['settings']['time_limit']:g} s; warm start {'on' if entry['settings']['warm_start'] else 'off'}.",
        "",
        "## 3. Baseline methodology",
        "- Every school with attendees rents its own cheapest bus that fits and drives straight to the event venue, "
        "priced with the same model's rates and the same route data.",
        "",
        "## 4. Optimisation approach",
        f"- Objective: {OBJECTIVES[sol.model]}.",
        f"- Method: {sol.method}.",
        "- Bus rates: " + ", ".join(f"{k}-seat = {v:g}" for k, v in sorted(rates.items())) + ".",
        "- Route data: " + ("not used for cost" if sol.model == "fixed" else sol.matrix_source) + ".",
        *[f"- Solver note: {n}" for n in sol.notes],
        "",
        "## 5. Results and comparisons",
        f"- Seat utilisation: {im.baseline_utilisation * 100:.1f}% (baseline) -> "
        f"{im.optimised_utilisation * 100:.1f}% (plan).",
        f"- Cost per attendee: {im.cost_per_student_before:.2f} -> {im.cost_per_student_after:.2f}.",
        f"- Buses: {im.baseline_buses} -> {im.optimised_buses} ({im.buses_removed} fewer).",
        *distance_lines(distance),
        "",
        "### Why this plan",
        *[f"- {'' if k == 'fact' else '*Interpretation:* '}{t}" for k, t in
          explain_plan(sol, base, ds, entry["constraints"], matrix)],
        "",
        "### All models run this session",
        summary.to_markdown(index=False) if not summary.empty else "Only this model was run.",
        "",
        "### Buses",
        bus_table(sol, ds.event.short_name).to_markdown(index=False),
        "",
        "### School cost shares (estimate, proportional to attendees)",
        cost_shares(sol, ds).round(2).to_markdown(index=False),
        "",
        "## 6. Constraint validation",
        "Checked independently of the solver (`src/validation.py`): every school with attendees on exactly one "
        "bus, no school split, no bus over capacity, only available bus sizes, every organiser rule, every route "
        "length and cost recalculated, total recalculated.",
        f"Result: {'no violations' if not violations else str(len(violations)) + ' violation(s)'}.",
        "",
        "## 7. Solver status and optimality",
        f"- Status: {sol.status_text()}; runtime {sol.runtime_s:.1f} s of {sol.time_limit_s:g} s.",
        "- OPTIMAL is reported only when CP-SAT proves it or the cost equals the seat-cover lower bound. "
        "Routing results are always FEASIBLE: valid, best found within the limit, not proven minimal.",
        "",
        "## 8. Limitations and assumptions",
        "- Unlimited buses of each size; a bus starts at its first pickup and ends at the event venue; no return "
        "trip; no dwell time; costs in the challenge's units.",
        "- Distance is straight-line (haversine), not road distance.",
        "- Travel times are OSRM road-network estimates, a labelled fixed-speed estimate, or a labelled "
        "experimental ML estimate of OSRM; none is an observed journey time.",
        "- Costs from different cost models are not comparable.",
        "",
        "## 9. Responsible AI and data handling",
        "- Plans come from OR-Tools and pass the independent validator. A language model (if configured) only "
        "proposes changes; an organiser approved every applied change.",
        "- Data: outcodes, one coordinate per outcode and head counts. No names or student records.",
        "- External services: OSRM (coordinates only, time model and road paths), the configured LLM API "
        "(outcodes, head counts, plan figures; only when the assistant is used).",
        "- An organiser must review this plan and confirm it with schools and the bus company before booking.",
        "",
        "## Audit trail (this session)",
        pd.DataFrame(run_log).to_markdown(index=False) if run_log else "No runs recorded.",
        "",
    ]
    return "\n".join(lines)


def run_summary(entry: dict, proposal, totals: dict | None, hours_source: str | None) -> tuple[str, str]:
    """(kind, markdown) chat summary of a finished assistant-approved run.

    Built only from the stored result and the approved proposal. ``totals`` are
    plan_totals() for a valid plan (or None).
    """
    sol, base = entry["result"].solution, entry["result"].baseline
    lines = [f"**Optimiser run #{entry['run_no']} finished at {entry['finished_at']}.**",
             f"- Objective: {OBJECTIVES[sol.model]}.",
             f"- Optimiser executed: {sol.method} (limit {sol.time_limit_s:g} s, ran {sol.runtime_s:.1f} s)."]
    if proposal is not None:
        if proposal.attendance:
            lines.append("- Attendance applied: " + ", ".join(
                f"{s} {proposal.current_attendance[s]} -> {v}" for s, v in proposal.attendance.items()) + ".")
        if proposal.rules:
            lines.append("- Hard rules applied: " + "; ".join(
                f"{k.replace('_', ' ')} {' & '.join(s)}" for k, s in proposal.rules) + ".")
        for n in proposal.notes:
            if n.startswith(("Keep nearby", "Fewer buses")):
                lines.append(f"- {n}")
    lines.append(f"- Status: {sol.status_text()}. Proven optimal: {'yes' if sol.proven_optimal else 'no'}.")
    if sol.status == "INVALID":
        lines.append("- The solver's plan failed independent validation: "
                     + "; ".join(sol.violations) + ". It is not shown or exported.")
        kind = "error"
    elif not sol.has_plan:
        lines.append("- No valid plan: " + " ".join(sol.notes) + " Nothing is shown as a new plan.")
        kind = "error"
    else:
        sv = savings(base, sol)
        pct = "n/a" if sv.percent is None else f"{sv.percent:.1f}%"
        lines += [
            f"- {len(sol.routes)} buses (baseline {len(base.routes)}), {sol.total_attendees} attendees, "
            f"seat utilisation {sol.utilisation * 100:.1f}%.",
            f"- Total cost {sol.total_cost:,.2f} vs baseline {base.total_cost:,.2f}: {sv.absolute:,.2f} saved "
            f"({pct}). Cost units; not comparable across models.",
            f"- Independent validation: passed.",
        ]
        if totals:
            lines.append(f"- Total straight-line miles: {totals['miles']:,.1f}."
                         + (f" Total driving hours: {totals['hours']:,.2f} ({hours_source})."
                            if totals.get("hours") is not None else " Driving hours: not available."))
        if sol.model != "fixed" and sol.matrix_source.startswith(("APPROX", "ML")):
            lines.append(f"- Travel times are estimates: {sol.matrix_source}.")
        kind = "success"
    lines.append("- Review it in **Overview** and **Routes**. *Generated by the app from the solver "
                 "result, not by the language model.*")
    return kind, "\n".join(lines)


def metrics_csv(summary: pd.DataFrame) -> str:
    return summary.to_csv(index=False)


def safe_percent(part: float, whole: float) -> float | None:
    """part / whole x 100, or None for a zero or non-finite base."""
    if not whole or not math.isfinite(whole) or not math.isfinite(part):
        return None
    return part / whole * 100.0
