"""One entry point used by the app and the demo: build matrices, solve, baseline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import Dataset
from .distances import SOURCE_APPROX, TimeMatrix, road_time_matrix, straight_line_matrix
from .model import INFEASIBLE, Constraints, Solution
from .solvers import event_points, solve_baseline, solve_fixed, solve_routing


@dataclass
class PlanResult:
    solution: Solution
    baseline: Solution
    matrix: np.ndarray


def points_of(dataset: Dataset) -> list[tuple[float, float]]:
    return event_points(dataset.schools, dataset.event.latitude, dataset.event.longitude)


def time_matrix_for(dataset: Dataset, allow_online: bool = True, allow_approximate: bool = True) -> TimeMatrix:
    return road_time_matrix(points_of(dataset), allow_online=allow_online, allow_approximate=allow_approximate)


def optimise(
    dataset: Dataset,
    model: str,
    constraints: Constraints | None = None,
    time_limit: float = 30.0,
    time_matrix: TimeMatrix | None = None,
    warm_start: Solution | None = None,
) -> PlanResult:
    """Solve one cost model. ``time_matrix`` is required for the time model.

    ``warm_start``: a previous valid plan to start the search from. It is
    repaired for the current attendance and constraints before use.
    """
    start = [list(r.stops) for r in warm_start.routes] if warm_start is not None and warm_start.has_plan else None
    constraints = constraints or Constraints()
    if model not in dataset.available_models:
        raise ValueError(f"Cost model '{model}' is not available in the supplied data.")
    rates = dataset.rates[model]
    miles = straight_line_matrix(points_of(dataset))

    if not dataset.report.model_ok(model):
        problems = dataset.report.errors + dataset.report.capacity_errors.get(model, [])
        blocked = Solution(model=model, status=INFEASIBLE, notes=problems)
        return PlanResult(blocked, blocked, miles)

    if model == "fixed":
        e = dataset.event
        sol = solve_fixed(dataset.schools, rates, e.latitude, e.longitude, constraints, time_limit,
                          warm_start=start)
        return PlanResult(sol, solve_baseline(dataset.schools, rates, model, miles), miles)

    if model == "distance":
        sol = solve_routing(dataset.schools, rates, miles, model, constraints, time_limit, warm_start=start)
        return PlanResult(sol, solve_baseline(dataset.schools, rates, model, miles), miles)

    if time_matrix is None:
        raise ValueError("The time model needs a travel-time matrix.")
    hours = time_matrix.hours
    sol = solve_routing(dataset.schools, rates, hours, model, constraints, time_limit,
                        metric_unit="hours", matrix_source=time_matrix.source, warm_start=start)
    if time_matrix.source == SOURCE_APPROX:
        sol.notes.append("Travel times are an offline approximation, not real road routing.")
    base = solve_baseline(dataset.schools, rates, model, hours, "hours")
    base.matrix_source = time_matrix.source
    return PlanResult(sol, base, hours)
