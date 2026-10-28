"""Scenarios tab: test a change without touching the current plan.

A scenario starts from the inputs of the current plan (its attendance and
organiser rules) and changes only what the organiser enters here:

* attendance: an optional percentage change for every school, then optional
  per-school values;
* bus prices of the plan's cost model: an optional percentage change.

The scenario is optimised by the same optimiser and validator as any plan.
Its result is stored separately and is never shown as the current plan unless
the organiser adopts it explicitly. Prices are supplied data, so a scenario
with a price change can be compared but not adopted.

Pure functions, no Streamlit, so the rules are unit-tested directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .data import Dataset, whole_attendance
from .model import Solution
from .report import impact_metrics

MIN_SCALE, MAX_SCALE = -90, 200          # percent change of every school's attendance
MIN_PRICE, MAX_PRICE = -90, 300          # percent change of every bus price
MAX_SCENARIO_ATTENDANCE = 500            # same upper bound as the attendance editor


@dataclass(frozen=True)
class ScenarioInputs:
    """What the organiser entered. ``edits``: school -> attendance, applied after scaling."""

    scale_percent: int = 0
    edits: tuple[tuple[str, int], ...] = ()
    price_percent: int = 0

    @property
    def changes_prices(self) -> bool:
        return self.price_percent != 0


@dataclass
class ScenarioCheck:
    attendance: dict[str, int] = field(default_factory=dict)  # full scenario attendance per school
    errors: list[str] = field(default_factory=list)


def scaled(count: int, percent: int) -> int:
    """``count`` changed by ``percent``, rounded up to a whole person (exact integer arithmetic).

    Rounding up keeps enough seats for a growing group.
    """
    return -(-count * (100 + percent) // 100)


def scenario_attendance(base: dict[str, int], inputs: ScenarioInputs) -> ScenarioCheck:
    """Attendance for every school in the scenario, or the reasons it is invalid."""
    out = ScenarioCheck()
    p = inputs.scale_percent
    if isinstance(p, bool) or not isinstance(p, int) or not MIN_SCALE <= p <= MAX_SCALE:
        out.errors.append(f"The attendance change must be a whole number of percent from {MIN_SCALE} "
                          f"to {MAX_SCALE}.")
        return out
    q = inputs.price_percent
    if isinstance(q, bool) or not isinstance(q, int) or not MIN_PRICE <= q <= MAX_PRICE:
        out.errors.append(f"The price change must be a whole number of percent from {MIN_PRICE} to {MAX_PRICE}.")
    att = {s: scaled(int(a), p) for s, a in base.items()}
    for school, value in inputs.edits:
        if school not in att:
            out.errors.append(f"Unknown school {school!r}.")
            continue
        count = whole_attendance(value)
        if count is None:
            out.errors.append(f"{school}: {value!r} is not a whole number >= 0.")
        elif count > MAX_SCENARIO_ATTENDANCE:
            out.errors.append(f"{school}: {count} is above the allowed maximum of {MAX_SCENARIO_ATTENDANCE}.")
        else:
            att[school] = count
    out.attendance = att
    return out


def scenario_dataset(dataset: Dataset, attendance: dict[str, int], model: str, price_percent: int) -> Dataset:
    """The source dataset with the scenario's attendance and prices. ``dataset`` is not changed."""
    source = dict(zip(dataset.schools["school_id"], dataset.schools["attendees"]))
    overrides = {s: v for s, v in attendance.items() if v != int(source[s])}
    working = dataset.with_attendance(overrides)
    if price_percent:
        working = working.with_price_change(model, price_percent)
    return working


def overrides_vs_source(dataset: Dataset, attendance: dict[str, int]) -> dict[str, int]:
    """Only the values that differ from the source data (the app's ``overrides`` format)."""
    source = dict(zip(dataset.schools["school_id"], dataset.schools["attendees"]))
    return {s: v for s, v in attendance.items() if v != int(source[s])}


def attendance_diff(before: dict[str, int], after: dict[str, int]) -> list[str]:
    return [f"{s}: {before[s]} -> {after[s]}" for s in before if before[s] != after.get(s)]


def _fmt(x, digits=2):
    return None if x is None else round(float(x), digits)


def comparison_table(current: tuple[Solution, Solution], scenario: tuple[Solution, Solution],
                     current_distance=None, scenario_distance=None) -> pd.DataFrame:
    """Measure | Current plan | Scenario | Difference, from two validated (plan, baseline) pairs.

    ``*_distance``: optional ``explain.DistanceImpact`` for each side. Costs are
    compared only because both sides use the same cost model.
    """
    (a, a_base), (b, b_base) = current, scenario
    if a.model != b.model:
        raise ValueError("A scenario is compared with a plan of the same cost model only.")
    ia, ib = impact_metrics(a_base, a), impact_metrics(b_base, b)
    rows = [
        ("Status", a.status_text(), b.status_text(), None),
        ("Attendees", ia.attendees, ib.attendees, ib.attendees - ia.attendees),
        ("Buses", ia.optimised_buses, ib.optimised_buses, ib.optimised_buses - ia.optimised_buses),
        ("Total cost", _fmt(ia.optimised_cost), _fmt(ib.optimised_cost), _fmt(ib.optimised_cost - ia.optimised_cost)),
        ("Baseline cost (one bus per school)", _fmt(ia.baseline_cost), _fmt(ib.baseline_cost),
         _fmt(ib.baseline_cost - ia.baseline_cost)),
        ("Cost per attendee", _fmt(ia.cost_per_student_after), _fmt(ib.cost_per_student_after),
         None if None in (ia.cost_per_student_after, ib.cost_per_student_after)
         else _fmt(ib.cost_per_student_after - ia.cost_per_student_after)),
        ("Seat utilisation %", _fmt(ia.optimised_utilisation * 100, 1), _fmt(ib.optimised_utilisation * 100, 1),
         _fmt((ib.optimised_utilisation - ia.optimised_utilisation) * 100, 1)),
    ]
    if current_distance is not None and scenario_distance is not None:
        rows.append(("Bus-miles (straight line)", _fmt(current_distance.plan_miles, 1),
                     _fmt(scenario_distance.plan_miles, 1),
                     _fmt(scenario_distance.plan_miles - current_distance.plan_miles, 1)))
        if current_distance.plan_hours is not None and scenario_distance.plan_hours is not None:
            rows.append(("Driving hours (summed over buses)", _fmt(current_distance.plan_hours),
                         _fmt(scenario_distance.plan_hours),
                         _fmt(scenario_distance.plan_hours - current_distance.plan_hours)))
    frame = pd.DataFrame(rows, columns=["Measure", "Current plan", "Scenario", "Difference"])
    return frame.astype({"Current plan": object, "Scenario": object, "Difference": object})

