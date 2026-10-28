"""Shared result and constraint types."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Constraints:
    """Organiser preferences, applied as hard constraints.

    must_share:   pairs of school ids that must ride the same bus.
    cannot_share: pairs of school ids that must ride different buses.
    solo:         school ids that must have a bus to themselves.
    """

    must_share: list[tuple[str, str]] = field(default_factory=list)
    cannot_share: list[tuple[str, str]] = field(default_factory=list)
    solo: list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not (self.must_share or self.cannot_share or self.solo)

    def describe(self) -> list[str]:
        lines = [f"{a} and {b} must share a bus" for a, b in self.must_share]
        lines += [f"{a} and {b} must not share a bus" for a, b in self.cannot_share]
        lines += [f"{s} travels on its own bus" for s in self.solo]
        return lines


@dataclass
class BusRoute:
    """One rented bus.

    stops:  school ids in pickup order; the bus then drives to the event.
    metric: route length in miles (distance model and fixed-model display)
            or hours (time model), first pickup to event.
    cost:   cost of this bus under the solution's model.
    """

    bus_id: str
    bus_size: int
    stops: list[str]
    attendees: int
    metric: float
    cost: float

    @property
    def empty_seats(self) -> int:
        return self.bus_size - self.attendees

    @property
    def utilisation(self) -> float:
        return self.attendees / self.bus_size if self.bus_size else 0.0


# Solver status values.
OPTIMAL = "OPTIMAL"  # proven minimum cost
FEASIBLE = "FEASIBLE"  # valid plan, optimality not proven
INFEASIBLE = "INFEASIBLE"  # proven that no valid plan exists
NO_SOLUTION = "NO_SOLUTION"  # solver stopped without finding a plan
INVALID = "INVALID"  # plan failed independent validation


@dataclass
class Solution:
    model: str
    status: str
    routes: list[BusRoute] = field(default_factory=list)
    total_cost: float = 0.0
    runtime_s: float = 0.0
    time_limit_s: float = 0.0
    method: str = ""
    metric_unit: str = "miles"
    matrix_source: str = "straight line (haversine)"
    notes: list[str] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)
    excluded_schools: list[str] = field(default_factory=list)

    @property
    def has_plan(self) -> bool:
        return self.status in (OPTIMAL, FEASIBLE) and bool(self.routes) and not self.violations

    @property
    def proven_optimal(self) -> bool:
        return self.status == OPTIMAL

    @property
    def total_attendees(self) -> int:
        return sum(r.attendees for r in self.routes)

    @property
    def total_seats(self) -> int:
        return sum(r.bus_size for r in self.routes)

    @property
    def utilisation(self) -> float:
        return self.total_attendees / self.total_seats if self.total_seats else 0.0

    def bus_of(self) -> dict[str, str]:
        return {sid: r.bus_id for r in self.routes for sid in r.stops}

    def status_text(self) -> str:
        return {
            OPTIMAL: "Proven optimal",
            FEASIBLE: "Feasible (optimality not proven)",
            INFEASIBLE: "Infeasible: no valid plan exists",
            NO_SOLUTION: "No plan found within the time limit",
            INVALID: "Plan failed validation",
        }.get(self.status, self.status)
