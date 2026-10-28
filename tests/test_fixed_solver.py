"""Fixed-cost solver tests. Small instances are checked against brute force."""

import random

import pytest

from conftest import EVENT, FIXED, make_schools
from src.model import FEASIBLE, INFEASIBLE, NO_SOLUTION, OPTIMAL, Constraints
from src.solvers import solve_fixed
from src.validation import validate_solution


def solve(schools, rates=FIXED, constraints=None, limit=10):
    return solve_fixed(schools, rates, *EVENT, constraints=constraints, time_limit=limit)


def partitions(items):
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for part in partitions(rest):
        for k in range(len(part)):
            yield part[:k] + [[first] + part[k]] + part[k + 1:]
        yield [[first]] + part


def brute_force_cost(sizes, rates, constraints=Constraints(), ids=None):
    ids = ids or list(range(len(sizes)))
    best = float("inf")
    for part in partitions(list(range(len(sizes)))):
        cost = 0.0
        group = {i: k for k, block in enumerate(part) for i in block}
        ok = all(group[ids.index(a)] == group[ids.index(b)] for a, b in constraints.must_share)
        ok &= all(group[ids.index(a)] != group[ids.index(b)] for a, b in constraints.cannot_share)
        ok &= all(len(part[group[ids.index(s)]]) == 1 for s in constraints.solo)
        if not ok:
            continue
        for block in part:
            load = sum(sizes[i] for i in block)
            fits = [rates[t] for t in rates if t >= load]
            if not fits:
                break
            cost += min(fits)
        else:
            best = min(best, cost)
    return best


def schools_from_sizes(sizes):
    return make_schools([(f"S{i}", 51.5 + i * 0.01, -3.3 - i * 0.01, a) for i, a in enumerate(sizes)])


def test_single_school():
    sol = solve(schools_from_sizes([12]))
    assert sol.status == OPTIMAL and sol.total_cost == 40 and len(sol.routes) == 1


def test_schools_share_one_bus():
    sol = solve(schools_from_sizes([10, 12, 20]))
    assert sol.status == OPTIMAL
    assert sol.total_cost == 75 and len(sol.routes) == 1


def test_schools_that_cannot_all_fit_on_one_bus():
    sol = solve(schools_from_sizes([30, 30, 30]))
    assert sol.status == OPTIMAL
    assert len(sol.routes) >= 2
    assert sol.total_cost == brute_force_cost([30, 30, 30], FIXED)


def test_mixed_sizes_cheaper_than_one_large():
    # 15-seat (40) + 35-seat (60) = 100 < two 50-seat (150) for loads 15 and 35.
    sol = solve(schools_from_sizes([15, 35]), rates={15: 40.0, 35: 60.0, 50: 200.0})
    assert sol.total_cost == 100
    assert sorted(r.bus_size for r in sol.routes) == [15, 35]


def test_attendance_equal_to_largest_bus():
    sol = solve(schools_from_sizes([50]))
    assert sol.status == OPTIMAL and sol.routes[0].bus_size == 50 and sol.routes[0].empty_seats == 0


def test_attendance_above_largest_bus_is_infeasible():
    sol = solve(schools_from_sizes([51, 10]))
    assert sol.status == INFEASIBLE and not sol.has_plan
    assert any("exceeds the largest bus" in n for n in sol.notes)


def test_zero_attendance_school_gets_no_bus():
    sol = solve(schools_from_sizes([0, 10]))
    assert sol.status == OPTIMAL
    assert sol.excluded_schools == ["S0"]
    assert all("S0" not in r.stops for r in sol.routes)


@pytest.mark.parametrize("seed", range(12))
def test_matches_brute_force_on_random_instances(seed):
    rng = random.Random(seed)
    sizes = [rng.randint(1, 45) for _ in range(rng.randint(2, 8))]
    sol = solve(schools_from_sizes(sizes))
    assert sol.status == OPTIMAL
    assert sol.total_cost == pytest.approx(brute_force_cost(sizes, FIXED))
    assert sol.violations == []


@pytest.mark.parametrize("seed", range(6))
def test_matches_brute_force_with_constraints(seed):
    rng = random.Random(100 + seed)
    sizes = [rng.randint(1, 20) for _ in range(6)]
    ids = [f"S{i}" for i in range(6)]
    c = Constraints(must_share=[("S0", "S5")], cannot_share=[("S1", "S2")], solo=["S3"])
    sol = solve(schools_from_sizes(sizes), constraints=c)
    assert sol.status == OPTIMAL
    assert sol.total_cost == pytest.approx(brute_force_cost(sizes, FIXED, c, ids))


def test_cannot_share_is_respected():
    schools = schools_from_sizes([10, 10])
    assert len(solve(schools).routes) == 1
    sol = solve(schools, constraints=Constraints(cannot_share=[("S0", "S1")]))
    assert len(sol.routes) == 2 and sol.total_cost == 80


def test_must_share_is_respected():
    schools = schools_from_sizes([30, 30, 15])
    sol = solve(schools, constraints=Constraints(must_share=[("S0", "S2")]))
    assert sol.bus_of()["S0"] == sol.bus_of()["S2"]


def test_solo_is_respected():
    schools = schools_from_sizes([5, 5, 5])
    sol = solve(schools, constraints=Constraints(solo=["S1"]))
    bus = next(r for r in sol.routes if "S1" in r.stops)
    assert bus.stops == ["S1"]


def test_infeasible_must_share_explained():
    schools = schools_from_sizes([30, 30])
    sol = solve(schools, constraints=Constraints(must_share=[("S0", "S1")]))
    assert sol.status == INFEASIBLE and sol.notes


def test_reoptimise_after_attendance_change(real_dataset):
    e = real_dataset.event
    before = solve_fixed(real_dataset.schools, FIXED, e.latitude, e.longitude, time_limit=20)
    changed = real_dataset.with_attendance({"CF15": 30, "SA18": 30})  # +54 attendees
    after = solve_fixed(changed.schools, FIXED, e.latitude, e.longitude, time_limit=20)
    assert before.has_plan and after.has_plan
    assert after.total_attendees == before.total_attendees + 54
    assert after.total_cost >= before.total_cost


def test_real_dataset_optimum_is_proven(real_dataset):
    e = real_dataset.event
    sol = solve_fixed(real_dataset.schools, FIXED, e.latitude, e.longitude, time_limit=30)
    # 1244 attendees need >= 25 buses; 25 buses seat 1244 only if all are 50-seat.
    assert sol.status == OPTIMAL
    assert sol.total_cost == 1875
    assert validate_solution(sol, real_dataset.schools, FIXED) == []


def test_tiny_time_limit_never_overclaims(real_dataset):
    e = real_dataset.event
    sol = solve_fixed(real_dataset.schools, FIXED, e.latitude, e.longitude,
                      time_limit=0.01, prefer_nearby=False)
    assert sol.status in (OPTIMAL, FEASIBLE, NO_SOLUTION)
    if sol.status == OPTIMAL:
        assert sol.total_cost == 1875  # a claimed optimum must be the true one
    if sol.status == NO_SOLUTION:
        assert not sol.has_plan


def test_seat_cover_lower_bound_matches_enumeration():
    from src.solvers import seat_cover_lower_bound

    for total in range(0, 160):
        best = min(
            40 * a + 60 * b + 75 * c
            for a in range(12) for b in range(6) for c in range(5)
            if 15 * a + 35 * b + 50 * c >= total
        )
        assert seat_cover_lower_bound(total, FIXED) == best
    assert seat_cover_lower_bound(1244, FIXED) == 1875
