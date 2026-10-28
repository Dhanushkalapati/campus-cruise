"""Attendance changes must reach the optimiser and refresh every derived result.

Tests named test_real_* use the supplied CSVs (copies in memory; the files are
never written). All other data here is SYNTHETIC and labelled as such.
"""

import math

import pytest

from conftest import EVENT, FIXED, PER_MILE, make_schools
from src.data import Dataset, attendance_edits, whole_attendance
from src.model import FEASIBLE, INFEASIBLE, OPTIMAL, Constraints
from src.planner import optimise
from src.report import attendance_changes, cost_shares, plan_csv, plan_report, savings
from src.solvers import solve_fixed, solve_routing, straight_line_matrix
from src.validation import validate_solution

REAL_FIXED_LIMIT = 15  # enough for CP-SAT to prove 1,875 on the real data on a laptop


def att_of(dataset, sid):
    return int(dataset.schools.loc[dataset.schools.school_id == sid, "attendees"].iloc[0])


def route_with(solution, sid):
    return next(r for r in solution.routes if sid in r.stops)


def assert_derived_from(result, dataset):
    """Every result figure is consistent with ``dataset`` (the edited copy)."""
    sol, base = result.solution, result.baseline
    total = int(dataset.schools["attendees"].sum())
    assert sol.has_plan
    assert sol.total_attendees == total
    assert base.total_attendees == total
    matrix = None if sol.model == "fixed" else result.matrix
    assert validate_solution(sol, dataset.schools, dataset.rates[sol.model], None, matrix) == []
    shares = cost_shares(sol, dataset)
    assert math.isclose(shares["Cost share"].sum(), sol.total_cost, abs_tol=1e-6)
    for row in shares.itertuples():
        assert row.Attendees == att_of(dataset, row.School)
    sv = savings(base, sol)
    assert math.isclose(sv.absolute, base.total_cost - sol.total_cost)
    assert math.isclose(sol.utilisation, total / sum(r.bus_size for r in sol.routes))


def synthetic_dataset(rows, model="fixed", rates=None):
    rates = rates or (FIXED if model == "fixed" else PER_MILE)
    return Dataset(schools=make_schools(rows), rates={model: dict(rates)})


# --------------------------------------------------------------------------- #
# Test A: real data, one school 18 -> 15
# --------------------------------------------------------------------------- #


def test_real_A_reduction_18_to_15_reaches_solver_and_outputs(real_dataset):
    assert att_of(real_dataset, "NP11") == 18  # documented original value
    edited = real_dataset.with_attendance({"NP11": 15})
    assert int(edited.schools["attendees"].sum()) == int(real_dataset.schools["attendees"].sum()) - 3
    result = optimise(edited, "fixed", Constraints(), REAL_FIXED_LIMIT)
    assert_derived_from(result, edited)
    sol, base = result.solution, result.baseline
    # The plan was built on 15, not 18: it fails a recount against the original data.
    assert validate_solution(sol, real_dataset.schools, FIXED) != []
    # Baseline recalculated: NP11 now fits a 15-seat bus (40) instead of a 35-seat bus (60).
    assert route_with(base, "NP11").bus_size == 15
    assert base.total_cost == pytest.approx(3330 - 20)
    assert sol.status in (OPTIMAL, FEASIBLE)
    if sol.status == OPTIMAL:
        assert sol.total_cost == 1875  # 1241 attendees still need 25 x 50 seats
    # Exports use the edited figures.
    assert "NP11,15," in plan_csv(sol, edited).replace(" ", "")
    report = plan_report(sol, base, edited, Constraints(), attendance_changes(real_dataset, edited))
    assert "NP11: 18 -> 15" in report


# --------------------------------------------------------------------------- #
# Test B: increase -> capacity violation detected, plan recalculated (SYNTHETIC)
# --------------------------------------------------------------------------- #


def test_B_increase_breaks_old_plan_and_is_reoptimised():
    rows = [("A", 51.60, -3.20, 10), ("B", 51.61, -3.21, 4)]
    before = synthetic_dataset(rows)
    first = optimise(before, "fixed", Constraints(), 5)
    assert [r.bus_size for r in first.solution.routes] == [15]
    after = before.with_attendance({"B": 6})  # 16 > 15
    # The old plan is no longer valid for the new attendance.
    problems = validate_solution(first.solution, after.schools, FIXED)
    assert any("exceed capacity" in p for p in problems)
    second = optimise(after, "fixed", Constraints(), 5, warm_start=first.solution)
    assert_derived_from(second, after)
    assert [r.bus_size for r in second.solution.routes] == [35]
    assert second.solution.total_cost == 60


# --------------------------------------------------------------------------- #
# Test C: reduction makes a different allocation cheaper (SYNTHETIC)
# --------------------------------------------------------------------------- #


def test_C_fixed_reduction_allows_fewer_buses():
    rows = [("A", 51.60, -3.20, 30), ("B", 51.61, -3.21, 21)]  # 51 > 50: two buses
    before = synthetic_dataset(rows)
    first = optimise(before, "fixed", Constraints(), 5)
    assert len(first.solution.routes) == 2 and first.solution.total_cost == 120
    after = before.with_attendance({"B": 20})  # 50: one 50-seat bus
    second = optimise(after, "fixed", Constraints(), 5, warm_start=first.solution)
    assert_derived_from(second, after)
    assert [r.bus_size for r in second.solution.routes] == [50]
    assert second.solution.total_cost == 75


def test_C_fixed_reduction_allows_smaller_bus():
    before = synthetic_dataset([("A", 51.60, -3.20, 20), ("B", 51.61, -3.21, 16)])  # 36 -> 50-seat
    after = before.with_attendance({"B": 15})  # 35 -> 35-seat
    a = optimise(before, "fixed", Constraints(), 5).solution
    b = optimise(after, "fixed", Constraints(), 5, warm_start=a).solution
    assert [r.bus_size for r in a.routes] == [50]
    assert [r.bus_size for r in b.routes] == [35]


def test_C_distance_reduction_changes_bus_size():
    # Two neighbouring schools about 25 miles from the event.
    rows = [("A", 51.80, -3.20, 10), ("B", 51.801, -3.201, 6)]
    before = synthetic_dataset(rows, "distance")
    first = optimise(before, "distance", Constraints(), 2)
    assert [r.bus_size for r in first.solution.routes] == [35]  # 16 > 15
    after = before.with_attendance({"B": 5})
    second = optimise(after, "distance", Constraints(), 2, warm_start=first.solution)
    assert_derived_from(second, after)
    assert [r.bus_size for r in second.solution.routes] == [15]
    assert second.solution.total_cost < first.solution.total_cost


# --------------------------------------------------------------------------- #
# Test D: same allocation, figures still refresh (SYNTHETIC)
# --------------------------------------------------------------------------- #


def test_D_same_allocation_refreshes_totals_and_shares():
    before = synthetic_dataset([("A", 51.60, -3.20, 10), ("B", 51.61, -3.21, 10)])
    after = before.with_attendance({"A": 9})
    r1 = optimise(before, "fixed", Constraints(), 5)
    r2 = optimise(after, "fixed", Constraints(), 5, warm_start=r1.solution)
    assert [sorted(r.stops) for r in r1.solution.routes] == [sorted(r.stops) for r in r2.solution.routes]
    assert r1.solution.total_cost == r2.solution.total_cost == 60
    assert_derived_from(r2, after)
    assert r2.solution.routes[0].attendees == 19
    assert r2.solution.utilisation == pytest.approx(19 / 35)
    assert r2.baseline.total_attendees == 19
    shares = cost_shares(r2.solution, after).set_index("School")["Cost share"]
    assert shares["A"] == pytest.approx(60 * 9 / 19)
    assert shares["B"] == pytest.approx(60 * 10 / 19)


# --------------------------------------------------------------------------- #
# Test E: several schools changed before one run (real data)
# --------------------------------------------------------------------------- #


def test_real_E_multiple_changes_all_reach_solver(real_dataset):
    changes = {"NP11": 15, "SA8": 10, "CF15": 25}
    edited = real_dataset.with_attendance(changes)
    delta = sum(v - att_of(real_dataset, k) for k, v in changes.items())
    assert int(edited.schools["attendees"].sum()) == 1244 + delta
    result = optimise(edited, "distance", Constraints(), 3)
    assert_derived_from(result, edited)
    for sid, value in changes.items():
        bus = route_with(result.solution, sid)
        assert bus.attendees == sum(att_of(edited, s) for s in bus.stops)
        assert att_of(edited, sid) == value


# --------------------------------------------------------------------------- #
# Test F: invalid attendance values
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("bad", [-1, -0.5, 2.5, "abc", "15", None, float("nan"), float("inf"), True, [3]])
def test_F_with_attendance_rejects_bad_values(real_dataset, bad):
    with pytest.raises(ValueError):
        real_dataset.with_attendance({"NP11": bad})


@pytest.mark.parametrize("good,expected", [(0, 0), (15, 15), (15.0, 15)])
def test_F_whole_attendance_accepts(good, expected):
    assert whole_attendance(good) == expected


def test_F_editor_edits_parsed_with_errors():
    ids, originals = ["A", "B", "C", "D", "E"], [18, 10, 5, 7, 9]
    edited_rows = {0: {"Attendees": 15}, 1: {"Attendees": None}, 2: {"Attendees": 4.5},
                   3: {"Attendees": -2}, 4: {"Attendees": 9}}
    overrides, errors = attendance_edits(edited_rows, ids, originals)
    assert overrides == {"A": 15}  # E unchanged, so not an override
    assert len(errors) == 3
    assert any("B" in e and "empty" in e for e in errors)
    assert any(e.startswith("C:") for e in errors)
    assert any(e.startswith("D:") for e in errors)


def test_F_editor_string_row_keys_and_all_rows_kept():
    overrides, errors = attendance_edits({"0": {"Attendees": 15}, "2": {"Attendees": 0}},
                                         ["A", "B", "C"], [18, 10, 5])
    assert overrides == {"A": 15, "C": 0} and errors == []


def test_F_zero_attendance_needs_no_bus_and_no_division_error(real_dataset):
    edited = real_dataset.with_attendance({"NP11": 0})
    result = optimise(edited, "distance", Constraints(), 2)
    assert_derived_from(result, edited)
    assert "NP11" not in result.solution.bus_of()
    assert "NP11" in result.solution.excluded_schools
    assert "NP11" not in cost_shares(result.solution, edited)["School"].tolist()


# --------------------------------------------------------------------------- #
# Test G: infeasible update shows no plan
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("model", ["fixed", "distance"])
def test_real_G_school_larger_than_every_bus(real_dataset, model):
    edited = real_dataset.with_attendance({"NP11": 60})
    result = optimise(edited, model, Constraints(), 2)
    assert result.solution.status == INFEASIBLE
    assert not result.solution.has_plan and not result.solution.routes
    assert any("NP11" in n for n in result.solution.notes)
    with pytest.raises(ValueError):
        plan_csv(result.solution, edited)


def test_real_G_conflicting_constraints_after_edit(real_dataset):
    c = Constraints(must_share=[("NP11", "SA8")], cannot_share=[("NP11", "SA8")])
    result = optimise(real_dataset, "fixed", c, 2)
    assert result.solution.status == INFEASIBLE and not result.solution.has_plan


# --------------------------------------------------------------------------- #
# Test H: repeated changes use the latest values
# --------------------------------------------------------------------------- #


def test_real_H_repeated_changes_use_latest_values(real_dataset):
    first_ds = real_dataset.with_attendance({"NP11": 15})
    first = optimise(first_ds, "distance", Constraints(), 2)
    second_ds = real_dataset.with_attendance({"NP11": 12, "SA8": 14})
    second = optimise(second_ds, "distance", Constraints(), 2, warm_start=first.solution)
    assert_derived_from(second, second_ds)
    assert second.solution.total_attendees == 1244 - 6 - 4
    assert first.solution.total_attendees == 1244 - 3
    # The first plan's figures are not valid for the second inputs.
    assert validate_solution(first.solution, second_ds.schools, PER_MILE, None, second.matrix) != []


# --------------------------------------------------------------------------- #
# Time limits
# --------------------------------------------------------------------------- #


def test_real_fixed_respects_time_budget(real_dataset):
    e = real_dataset.event
    sol = solve_fixed(real_dataset.schools, FIXED, e.latitude, e.longitude, time_limit=3)
    assert sol.runtime_s <= 3 + 0.75  # model build and validation overhead only
    assert sol.status in (OPTIMAL, FEASIBLE)
    if sol.status == FEASIBLE:
        assert any("lower bound" in n for n in sol.notes)


def test_real_routing_respects_time_limit(real_dataset):
    miles = straight_line_matrix(list(zip(real_dataset.schools.latitude, real_dataset.schools.longitude)) + [EVENT])
    sol = solve_routing(real_dataset.schools, PER_MILE, miles, "distance", time_limit=2)
    assert sol.status == FEASIBLE and sol.has_plan
    assert sol.runtime_s <= 2 + 1.5
