"""Input validation and independent solution checking.

Input checks treat every CSV as untrusted: values are parsed as text first and
converted explicitly, so malformed rows are reported rather than crashing.

``validate_solution`` re-checks a solver result without trusting the solver:
it re-counts passengers, re-applies every constraint and recomputes every
route cost from the route legs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    from .model import Constraints, Solution

SCHOOL_COLUMNS = {"outcode", "latitude", "longitude", "attendants"}
COST_TOLERANCE = 0.01


@dataclass
class ValidationReport:
    """Result of input validation.

    errors:   problems that block optimisation for every model.
    warnings: problems that were handled (for example, a row excluded).
    excluded: human-readable descriptions of excluded rows.
    capacity_errors: model -> schools that fit on no bus of that model.
    """

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    excluded: list[str] = field(default_factory=list)
    capacity_errors: dict[str, list[str]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def model_ok(self, model: str) -> bool:
        return self.ok and not self.capacity_errors.get(model)

    def check_capacity(self, schools: pd.DataFrame, model: str, rates: dict[int, float]) -> None:
        """Record schools that are larger than every bus of ``model``."""
        largest = max(rates) if rates else 0
        too_big = schools[schools["attendees"] > largest]
        self.capacity_errors[model] = [
            f"{row.name} has {row.attendees} attendees but the largest bus seats {largest}; "
            "a school cannot be split, so no feasible plan exists."
            for row in too_big.itertuples()
        ]


def _to_number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype("string").str.strip(), errors="coerce")


def validate_schools(raw: pd.DataFrame | None, report: ValidationReport) -> pd.DataFrame:
    """Return clean schools (school_id, name, latitude, longitude, attendees)."""
    empty = pd.DataFrame(
        {
            "school_id": pd.Series(dtype=str),
            "name": pd.Series(dtype=str),
            "latitude": pd.Series(dtype=float),
            "longitude": pd.Series(dtype=float),
            "attendees": pd.Series(dtype=int),
        }
    )
    if raw is None:
        return empty
    missing = SCHOOL_COLUMNS - set(raw.columns)
    if missing:
        report.errors.append(f"School file is missing columns: {', '.join(sorted(missing))}")
        return empty
    if raw.empty:
        report.errors.append("School file contains no rows.")
        return empty

    ids = raw["outcode"].astype("string").str.strip()
    lat = _to_number(raw["latitude"])
    lon = _to_number(raw["longitude"])
    att = _to_number(raw["attendants"])

    keep = []
    for i in range(len(raw)):
        label = ids.iloc[i] if pd.notna(ids.iloc[i]) and ids.iloc[i] else f"row {i + 2}"
        problems = []
        if pd.isna(ids.iloc[i]) or not ids.iloc[i]:
            problems.append("missing outcode")
        if pd.isna(lat.iloc[i]) or not -90 <= lat.iloc[i] <= 90:
            problems.append("missing or invalid latitude")
        if pd.isna(lon.iloc[i]) or not -180 <= lon.iloc[i] <= 180:
            problems.append("missing or invalid longitude")
        if pd.isna(att.iloc[i]):
            problems.append("missing attendance")
        elif att.iloc[i] < 0:
            problems.append("negative attendance")
        elif att.iloc[i] != int(att.iloc[i]):
            problems.append("non-integer attendance")
        if problems:
            report.excluded.append(f"{label}: {', '.join(problems)}")
        else:
            keep.append(i)

    if report.excluded:
        report.warnings.append(
            f"{len(report.excluded)} school row(s) excluded because of invalid data: "
            + "; ".join(report.excluded)
        )

    schools = pd.DataFrame(
        {
            "school_id": ids.iloc[keep].astype(str).to_numpy(),
            "latitude": lat.iloc[keep].astype(float).to_numpy(),
            "longitude": lon.iloc[keep].astype(float).to_numpy(),
            "attendees": att.iloc[keep].astype(int).to_numpy(),
        }
    )
    # The dataset identifies schools by postcode outcode only; no names are supplied.
    schools.insert(1, "name", schools["school_id"])

    dupes = schools["school_id"][schools["school_id"].duplicated()].unique().tolist()
    if dupes:
        report.errors.append(
            f"Duplicate school identifiers: {', '.join(dupes)}. Fix the source file; "
            "the optimiser cannot tell these schools apart."
        )
    if schools.empty:
        report.errors.append("No valid school rows remain after validation.")
    zero = schools.loc[schools["attendees"] == 0, "school_id"].tolist()
    if zero:
        report.warnings.append(
            f"Schools with zero attendees need no bus and are left out of the plan: {', '.join(zero)}"
        )
    return schools.reset_index(drop=True)


def validate_bus_table(
    raw: pd.DataFrame | None, price_column: str, file_name: str, report: ValidationReport
) -> dict[int, float]:
    """Return {bus size: rate} for one cost file. Invalid rows are excluded."""
    if raw is None:
        return {}
    needed = {"Bus size", price_column}
    missing = needed - set(raw.columns)
    if missing:
        report.warnings.append(f"{file_name} is missing columns {', '.join(sorted(missing))}; model unavailable.")
        return {}
    sizes = _to_number(raw["Bus size"])
    prices = _to_number(raw[price_column])
    table: dict[int, float] = {}
    for i in range(len(raw)):
        size, price = sizes.iloc[i], prices.iloc[i]
        if pd.isna(size) or size <= 0 or size != int(size):
            report.warnings.append(f"{file_name} row {i + 2}: invalid bus size {raw['Bus size'].iloc[i]!r}; row excluded.")
            continue
        if pd.isna(price) or price < 0:
            report.warnings.append(f"{file_name} row {i + 2}: invalid price {raw[price_column].iloc[i]!r}; row excluded.")
            continue
        if int(size) in table:
            report.warnings.append(f"{file_name}: duplicate bus size {int(size)}; first value kept.")
            continue
        table[int(size)] = float(price)
    if not table:
        report.warnings.append(f"{file_name} has no valid rows; model unavailable.")
    return table


# --------------------------------------------------------------------------- #
# Independent solution validation
# --------------------------------------------------------------------------- #


def route_metric(stops: list[int], matrix: np.ndarray) -> float:
    """Sum of legs first stop -> ... -> last stop -> event (index -1 of matrix).

    The bus starts at its first school (challenge rule), so no leg is charged
    before the first pickup.
    """
    if not stops:
        return 0.0
    event = matrix.shape[0] - 1
    path = list(stops) + [event]
    return float(sum(matrix[a, b] for a, b in zip(path[:-1], path[1:])))


def validate_solution(
    solution: "Solution",
    schools: pd.DataFrame,
    rates: dict[int, float],
    constraints: "Constraints | None" = None,
    matrix: np.ndarray | None = None,
) -> list[str]:
    """Return a list of violations; an empty list means the plan is valid.

    ``matrix`` (miles or hours, schools then event as the last row) is
    required for the distance and time models.
    """
    problems: list[str] = []
    index = {sid: i for i, sid in enumerate(schools["school_id"])}
    attendees = dict(zip(schools["school_id"], schools["attendees"]))
    required = {sid for sid, a in attendees.items() if a > 0}

    seen: dict[str, str] = {}
    total = 0.0
    for route in solution.routes:
        if route.bus_size not in rates:
            problems.append(f"{route.bus_id}: bus size {route.bus_size} is not an available type.")
            continue
        if not route.stops:
            problems.append(f"{route.bus_id}: bus has no schools.")
        load = 0
        for sid in route.stops:
            if sid not in index:
                problems.append(f"{route.bus_id}: unknown school {sid}.")
                continue
            if sid in seen:
                problems.append(f"{sid} is on both {seen[sid]} and {route.bus_id} (school split or duplicated).")
            seen[sid] = route.bus_id
            load += int(attendees[sid])
        if load != route.attendees:
            problems.append(f"{route.bus_id}: reported {route.attendees} attendees, recount gives {load}.")
        if load > route.bus_size:
            problems.append(f"{route.bus_id}: {load} attendees exceed capacity {route.bus_size}.")

        rate = rates[route.bus_size]
        if solution.model == "fixed":
            cost = rate
        else:
            if matrix is None:
                problems.append("No distance/time matrix supplied for route cost check.")
                break
            metric = route_metric([index[s] for s in route.stops if s in index], matrix)
            if abs(metric - route.metric) > 1e-6:
                problems.append(f"{route.bus_id}: reported route length {route.metric:.4f}, recalculated {metric:.4f}.")
            cost = rate * metric
        if abs(cost - route.cost) > COST_TOLERANCE:
            problems.append(f"{route.bus_id}: reported cost {route.cost:.2f}, recalculated {cost:.2f}.")
        total += cost

    missing = required - set(seen)
    if missing:
        problems.append(f"Schools not assigned to any bus: {', '.join(sorted(missing))}")
    extra = set(seen) - required
    if extra:
        problems.append(f"Schools with zero attendees were given a bus: {', '.join(sorted(extra))}")
    if abs(total - solution.total_cost) > COST_TOLERANCE:
        problems.append(f"Reported total {solution.total_cost:.2f}, recalculated {total:.2f}.")

    if constraints is not None:
        for a, b in constraints.must_share:
            if a in seen and b in seen and seen[a] != seen[b]:
                problems.append(f"{a} and {b} must share a bus but are on {seen[a]} and {seen[b]}.")
        for a, b in constraints.cannot_share:
            if a in seen and b in seen and seen[a] == seen[b]:
                problems.append(f"{a} and {b} must not share a bus but are both on {seen[a]}.")
        for sid in constraints.solo:
            bus = next((r for r in solution.routes if sid in r.stops), None)
            if bus is not None and len(bus.stops) > 1:
                problems.append(f"{sid} must travel alone but shares {bus.bus_id}.")
    return problems
