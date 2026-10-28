"""Cost accounting, routing, baseline, savings, cost shares and export."""

import itertools

import numpy as np
import pytest

from conftest import EVENT, FIXED, PER_MILE, make_schools
from src.data import Dataset
from src.distances import haversine_miles, straight_line_matrix
from src.model import FEASIBLE, INVALID, OPTIMAL, BusRoute, Constraints, Solution
from src.report import compare_plans, cost_shares, plan_csv, plan_report, savings
from src.solvers import event_points, solve_baseline, solve_routing
from src.validation import route_metric


def test_haversine_one_degree_latitude():
    assert haversine_miles(51.0, -3.0, 52.0, -3.0) == pytest.approx(69.09, abs=0.05)
    assert haversine_miles(51.5, -3.2, 51.5, -3.2) == 0


def test_route_legs_start_at_first_school():
    # points: A, B, event. Bus A -> B -> event = 3 + 4; nothing before A.
    m = np.array([[0, 3, 10], [3, 0, 4], [10, 4, 0]], dtype=float)
    assert route_metric([0, 1], m) == 7
    assert route_metric([1, 0], m) == 3 + 10
    assert route_metric([0], m) == 10
    assert route_metric([], m) == 0


def test_asymmetric_legs_use_direction():
    m = np.array([[0, 1, 5], [9, 0, 2], [0, 0, 0]], dtype=float)
    assert route_metric([0, 1], m) == 1 + 2
    assert route_metric([1, 0], m) == 9 + 5


def brute_force_routing(schools, rates, matrix):
    """Exact minimum over all partitions, orders and bus sizes (tiny instances)."""
    from test_fixed_solver import partitions

    att = schools["attendees"].tolist()
    best = float("inf")
    for part in partitions(list(range(len(att)))):
        total = 0.0
        for block in part:
            load = sum(att[i] for i in block)
            fits = [t for t in rates if t >= load]
            if not fits:
                total = float("inf")
                break
            dist = min(route_metric(list(p), matrix) for p in itertools.permutations(block))
            total += min(rates[t] for t in fits) * dist
        best = min(best, total)
    return best


SYN = make_schools([
    ("A", 51.70, -3.60, 10), ("B", 51.72, -3.58, 12),
    ("C", 51.65, -3.00, 20), ("D", 51.80, -3.30, 30), ("E", 51.62, -3.02, 9),
])


def test_routing_matches_brute_force_on_small_instance():
    m = straight_line_matrix(event_points(SYN, *EVENT))
    sol = solve_routing(SYN, PER_MILE, m, "distance", time_limit=3)
    assert sol.status == FEASIBLE and sol.violations == []
    assert sol.total_cost == pytest.approx(brute_force_routing(SYN, PER_MILE, m), rel=1e-6)


def test_routing_cost_equals_independent_recalculation():
    m = straight_line_matrix(event_points(SYN, *EVENT))
    sol = solve_routing(SYN, PER_MILE, m, "distance", time_limit=2)
    index = {s: i for i, s in enumerate(SYN["school_id"])}
    recalculated = sum(PER_MILE[r.bus_size] * route_metric([index[s] for s in r.stops], m) for r in sol.routes)
    assert sol.total_cost == pytest.approx(recalculated)


def test_routing_respects_constraints():
    m = straight_line_matrix(event_points(SYN, *EVENT))
    c = Constraints(must_share=[("A", "D")], cannot_share=[("C", "E")], solo=["B"])
    sol = solve_routing(SYN, PER_MILE, m, "distance", constraints=c, time_limit=2)
    bus = sol.bus_of()
    assert sol.has_plan
    assert bus["A"] == bus["D"] and bus["C"] != bus["E"]
    assert next(r for r in sol.routes if "B" in r.stops).stops == ["B"]


def test_routing_never_claims_optimal():
    m = straight_line_matrix(event_points(SYN, *EVENT))
    assert solve_routing(SYN, PER_MILE, m, "distance", time_limit=1).status == FEASIBLE


def test_baseline_uses_cheapest_fitting_bus():
    m = straight_line_matrix(event_points(SYN, *EVENT))
    fixed = solve_baseline(SYN, FIXED, "fixed", m)
    assert [r.bus_size for r in sorted(fixed.routes, key=lambda r: r.stops)] == [15, 15, 35, 35, 15]
    assert fixed.total_cost == 40 + 40 + 60 + 60 + 40
    dist = solve_baseline(SYN, PER_MILE, "distance", m)
    expected = sum(PER_MILE[15 if a <= 15 else 35] * m[i, -1] for i, a in enumerate(SYN["attendees"]))
    assert dist.total_cost == pytest.approx(expected)


def test_savings_math_and_negative_savings():
    base = Solution("fixed", OPTIMAL, total_cost=200.0)
    better = Solution("fixed", OPTIMAL, total_cost=150.0)
    worse = Solution("fixed", OPTIMAL, total_cost=250.0)
    assert savings(base, better).percent == pytest.approx(25.0)
    assert savings(base, worse).absolute == -50.0
    assert savings(base, worse).percent == pytest.approx(-25.0)
    assert savings(Solution("fixed", OPTIMAL, total_cost=0.0), better).percent is None


def dataset_for(schools):
    return Dataset(schools=schools, rates={"fixed": FIXED, "distance": PER_MILE})


def test_cost_shares_sum_to_bus_and_total():
    m = straight_line_matrix(event_points(SYN, *EVENT))
    sol = solve_routing(SYN, PER_MILE, m, "distance", time_limit=2)
    shares = cost_shares(sol, dataset_for(SYN))
    assert shares["Cost share"].sum() == pytest.approx(sol.total_cost)
    for r in sol.routes:
        assert shares.loc[shares["Bus"] == r.bus_id, "Cost share"].sum() == pytest.approx(r.cost)
    assert len(shares) == len(SYN)


def test_cost_share_proportional():
    sol = Solution("fixed", OPTIMAL, routes=[BusRoute("Bus 01", 35, ["A", "B"], 22, 0, 60.0)], total_cost=60)
    shares = cost_shares(sol, dataset_for(SYN)).set_index("School")
    assert shares.loc["A", "Cost share"] == pytest.approx(60 * 10 / 22)
    assert shares.loc["B", "Cost per attendee"] == pytest.approx(60 / 22)


def test_export_valid_plan():
    m = straight_line_matrix(event_points(SYN, *EVENT))
    sol = solve_routing(SYN, PER_MILE, m, "distance", time_limit=1)
    base = solve_baseline(SYN, PER_MILE, "distance", m)
    csv = plan_csv(sol, dataset_for(SYN))
    assert csv.splitlines()[0].startswith("School,Attendees,Bus,Bus size")
    assert len(csv.strip().splitlines()) == len(SYN) + 1
    report = plan_report(sol, base, dataset_for(SYN), Constraints(), [])
    assert "Savings" in report and "not proven" in report


def test_export_refuses_invalid_plan():
    bad = Solution("fixed", INVALID, routes=[BusRoute("Bus 01", 15, ["A"], 10, 0, 40.0)], total_cost=40)
    with pytest.raises(ValueError):
        plan_csv(bad, dataset_for(SYN))


def test_compare_plans_lists_changed_schools():
    a = Solution("fixed", OPTIMAL, routes=[BusRoute("1", 35, ["A", "B"], 22, 0, 60), BusRoute("2", 35, ["C"], 20, 0, 60)])
    b = Solution("fixed", OPTIMAL, routes=[BusRoute("1", 50, ["A", "B", "C"], 42, 0, 75)])
    diff = compare_plans(a, b)
    assert set(diff["School"]) == {"A", "B", "C"}
    assert compare_plans(a, a).empty
