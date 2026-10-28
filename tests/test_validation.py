import numpy as np

from conftest import FIXED, make_schools
from src.model import BusRoute, Constraints, Solution, OPTIMAL
from src.solvers import check_constraints
from src.validation import validate_solution

SCHOOLS = make_schools([("A", 51.6, -3.3, 10), ("B", 51.6, -3.4, 20), ("C", 51.7, -3.3, 30)])


def plan(routes, total=None):
    total = sum(r.cost for r in routes) if total is None else total
    return Solution(model="fixed", status=OPTIMAL, routes=routes, total_cost=total)


def test_valid_plan_passes():
    p = plan([BusRoute("1", 35, ["A", "B"], 30, 0, 60.0), BusRoute("2", 35, ["C"], 30, 0, 60.0)])
    assert validate_solution(p, SCHOOLS, FIXED) == []


def test_split_school_detected():
    p = plan([BusRoute("1", 35, ["A", "B"], 30, 0, 60.0), BusRoute("2", 50, ["A", "C"], 40, 0, 75.0)])
    assert any("both" in v for v in validate_solution(p, SCHOOLS, FIXED))


def test_capacity_violation_detected():
    p = plan([BusRoute("1", 35, ["A", "B", "C"], 60, 0, 60.0)])
    assert any("exceed capacity" in v for v in validate_solution(p, SCHOOLS, FIXED))


def test_missing_school_detected():
    p = plan([BusRoute("1", 35, ["A", "B"], 30, 0, 60.0)])
    assert any("not assigned" in v for v in validate_solution(p, SCHOOLS, FIXED))


def test_wrong_cost_detected():
    p = plan([BusRoute("1", 50, ["A", "B"], 30, 0, 60.0), BusRoute("2", 35, ["C"], 30, 0, 60.0)])
    assert any("recalculated" in v for v in validate_solution(p, SCHOOLS, FIXED))


def test_wrong_total_detected():
    p = plan([BusRoute("1", 35, ["A", "B"], 30, 0, 60.0), BusRoute("2", 35, ["C"], 30, 0, 60.0)], total=100)
    assert any("Reported total" in v for v in validate_solution(p, SCHOOLS, FIXED))


def test_unknown_bus_size_detected():
    p = plan([BusRoute("1", 40, ["A", "B", "C"], 60, 0, 60.0)])
    assert any("not an available type" in v for v in validate_solution(p, SCHOOLS, FIXED))


def test_route_cost_recomputed_from_legs():
    m = np.array([[0, 1, 9], [1, 0, 4], [9, 4, 0]], dtype=float)  # A, B, event
    schools = make_schools([("A", 0, 0, 5), ("B", 0, 0, 5)])
    good = Solution(model="distance", status=OPTIMAL,
                    routes=[BusRoute("1", 15, ["A", "B"], 10, 5.0, 2.25 * 5)], total_cost=11.25)
    assert validate_solution(good, schools, {15: 2.25}, matrix=m) == []
    bad = Solution(model="distance", status=OPTIMAL,
                   routes=[BusRoute("1", 15, ["A", "B"], 10, 4.0, 9.0)], total_cost=9.0)
    assert validate_solution(bad, schools, {15: 2.25}, matrix=m)


def test_constraint_violations_detected():
    p = plan([BusRoute("1", 35, ["A", "B"], 30, 0, 60.0), BusRoute("2", 35, ["C"], 30, 0, 60.0)])
    c = Constraints(must_share=[("A", "C")], cannot_share=[("A", "B")], solo=["B"])
    v = validate_solution(p, SCHOOLS, FIXED, c)
    assert len(v) == 3


def test_conflicting_constraints_explained():
    c = Constraints(must_share=[("A", "B"), ("B", "C")], cannot_share=[("A", "C")])
    problems = check_constraints(SCHOOLS, FIXED, c)
    assert any("forbidden" in p for p in problems)


def test_must_share_over_capacity_explained():
    schools = make_schools([("A", 51.6, -3.3, 30), ("B", 51.6, -3.4, 30)])
    problems = check_constraints(schools, FIXED, Constraints(must_share=[("A", "B")]))
    assert any("largest bus" in p for p in problems)


def test_solo_and_must_share_conflict():
    problems = check_constraints(SCHOOLS, FIXED, Constraints(must_share=[("A", "B")], solo=["A"]))
    assert any("alone" in p for p in problems)


def test_unknown_school_in_constraint():
    problems = check_constraints(SCHOOLS, FIXED, Constraints(solo=["ZZ"]))
    assert any("unknown" in p for p in problems)
