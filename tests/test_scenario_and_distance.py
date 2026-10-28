"""Bus-distance impact (src/explain.py), price what-ifs (src/data.py) and the
Scenario simulator (src/scenario.py, Scenarios tab). Synthetic data is SYNTHETIC."""

from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.data import Dataset
from src.distances import straight_line_matrix
from src.explain import DistanceImpact, distance_impact, distance_lines, plan_totals
from src.model import Constraints
from src.planner import optimise, points_of
from src.report import plan_report
from src.scenario import (
    ScenarioInputs,
    attendance_diff,
    comparison_table,
    overrides_vs_source,
    scaled,
    scenario_attendance,
    scenario_dataset,
)
from tests.conftest import FIXED, PER_MILE, make_schools

APP = str(Path(__file__).resolve().parent.parent / "app.py")
SCHOOLS = make_schools([("A", 51.60, -3.30, 30), ("B", 51.61, -3.31, 10), ("C", 51.70, -3.60, 12)])


def ds(rates):
    return Dataset(schools=SCHOOLS, rates=rates)


@pytest.fixture(scope="module")
def fixed():
    data = ds({"fixed": FIXED, "distance": PER_MILE})
    return data, optimise(data, "fixed", Constraints(), 5)


# --------------------------------------------------------------------------- #
# Bus-distance impact
# --------------------------------------------------------------------------- #


def test_distance_impact_matches_route_sums(fixed):
    data, res = fixed
    miles = straight_line_matrix(points_of(data))
    di = distance_impact(res.solution, res.baseline, data, miles)
    assert di.baseline_miles == pytest.approx(plan_totals(res.baseline, data, miles, None)["miles"])
    assert di.plan_miles == pytest.approx(plan_totals(res.solution, data, miles, None)["miles"])
    assert di.miles_saved == pytest.approx(di.baseline_miles - di.plan_miles)
    assert di.miles_saved_percent == pytest.approx(di.miles_saved / di.baseline_miles * 100)
    assert di.hours_saved is None and di.hours_source is None  # no hours given: none invented


def test_distance_impact_with_hours(fixed):
    data, res = fixed
    miles = straight_line_matrix(points_of(data))
    hours = miles / 30.0
    di = distance_impact(res.solution, res.baseline, data, miles, hours, "test hours")
    assert di.hours_saved == pytest.approx(di.miles_saved / 30.0)
    assert di.hours_source == "test hours"


def test_distance_percent_safe_with_zero_baseline():
    di = DistanceImpact(baseline_miles=0.0, plan_miles=0.0, baseline_hours=0.0, plan_hours=0.0)
    assert di.miles_saved_percent is None and di.hours_saved_percent is None


def test_distance_impact_refuses_mismatched_baseline(fixed):
    data, res = fixed
    miles = straight_line_matrix(points_of(data))
    other = optimise(data.with_attendance({"A": 20}), "fixed", Constraints(), 5)
    with pytest.raises(ValueError):
        distance_impact(res.solution, other.baseline, data, miles)


def test_distance_lines_state_units_and_no_emissions_claim(fixed):
    data, res = fixed
    di = distance_impact(res.solution, res.baseline, data, straight_line_matrix(points_of(data)))
    text = "\n".join(distance_lines(di))
    assert "straight-line" in text.lower() and "not road mileage" in text
    assert "not estimated" in text and "no travel-time data" in text
    assert distance_lines(None) == ["- Bus distance: not available for this run."]


def test_plan_report_has_required_sections(fixed):
    data, res = fixed
    di = distance_impact(res.solution, res.baseline, data, straight_line_matrix(points_of(data)))
    md = plan_report(res.solution, res.baseline, data, Constraints(), [], distance_lines(di))
    for heading in ("## Summary", "## Inputs and assumptions", "## Solver", "## Result",
                    "## Approximations and warnings", "## Responsible AI and privacy", "## Buses"):
        assert heading in md
    assert "bus-miles saved" in md and "Independent validation: passed" in md


# --------------------------------------------------------------------------- #
# Price what-if
# --------------------------------------------------------------------------- #


def test_price_change_copies_rates():
    data = ds({"fixed": FIXED})
    up = data.with_price_change("fixed", 10)
    assert up.rates["fixed"][50] == pytest.approx(82.5)
    assert data.rates["fixed"][50] == 75.0  # original untouched


@pytest.mark.parametrize("bad", [-91, 301, 2.5, True, "10"])
def test_price_change_rejects_bad_values(bad):
    with pytest.raises(ValueError):
        ds({"fixed": FIXED}).with_price_change("fixed", bad)


def test_price_change_reprices_plan_and_baseline():
    data = ds({"fixed": FIXED})
    a = optimise(data, "fixed", Constraints(), 5)
    b = optimise(data.with_price_change("fixed", 20), "fixed", Constraints(), 5)
    assert b.solution.total_cost == pytest.approx(a.solution.total_cost * 1.2)
    assert b.baseline.total_cost == pytest.approx(a.baseline.total_cost * 1.2)


# --------------------------------------------------------------------------- #
# Scenario attendance
# --------------------------------------------------------------------------- #


def test_scaled_rounds_up_exactly():
    assert scaled(10, 10) == 11          # float maths would give 11.000000000000002 -> 12
    assert scaled(15, 10) == 17          # 16.5 rounds up
    assert scaled(15, 0) == 15
    assert scaled(15, -50) == 8          # 7.5 rounds up
    assert scaled(0, 50) == 0


def test_scenario_attendance_scale_then_edits():
    check = scenario_attendance({"A": 30, "B": 10}, ScenarioInputs(10, (("B", 4.0),), 0))
    assert not check.errors and check.attendance == {"A": 33, "B": 4}


@pytest.mark.parametrize("inputs, fragment", [
    (ScenarioInputs(-91), "attendance change"),
    (ScenarioInputs(0, (), 400), "price change"),
    (ScenarioInputs(0, (("Z", 3),)), "Unknown school"),
    (ScenarioInputs(0, (("A", 2.5),)), "not a whole number"),
    (ScenarioInputs(0, (("A", 900),)), "maximum"),
])
def test_scenario_attendance_rejects_bad_inputs(inputs, fragment):
    check = scenario_attendance({"A": 30}, inputs)
    assert any(fragment in e for e in check.errors)


def test_scenario_dataset_leaves_source_untouched():
    data = ds({"fixed": FIXED})
    working = scenario_dataset(data, {"A": 40, "B": 10, "C": 12}, "fixed", 5)
    assert int(working.schools.set_index("school_id").loc["A", "attendees"]) == 40
    assert int(data.schools.set_index("school_id").loc["A", "attendees"]) == 30
    assert working.rates["fixed"][15] == pytest.approx(42.0) and data.rates["fixed"][15] == 40.0
    assert overrides_vs_source(data, {"A": 40, "B": 10, "C": 12}) == {"A": 40}
    assert attendance_diff({"A": 30, "B": 10}, {"A": 40, "B": 10}) == ["A: 30 -> 40"]


def test_scenario_over_capacity_has_no_plan():
    data = ds({"fixed": FIXED})
    res = optimise(scenario_dataset(data, {"A": 60, "B": 10, "C": 12}, "fixed", 0), "fixed", Constraints(), 5)
    assert not res.solution.has_plan  # 60 > 50 seats and a school is never split


def test_comparison_table(fixed):
    data, res = fixed
    more = optimise(data.with_attendance({"B": 25}), "fixed", Constraints(), 5)
    table = comparison_table((res.solution, res.baseline), (more.solution, more.baseline))
    row = table.set_index("Measure")
    assert row.loc["Attendees", "Difference"] == 15
    assert row.loc["Total cost", "Difference"] == pytest.approx(more.solution.total_cost - res.solution.total_cost)
    other = optimise(data, "distance", Constraints(), 1)
    with pytest.raises(ValueError):
        comparison_table((res.solution, res.baseline), (other.solution, other.baseline))


# --------------------------------------------------------------------------- #
# Scenarios tab (AppTest, real data, fixed model)
# --------------------------------------------------------------------------- #


def fixed_app():
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    at.sidebar.radio[0].set_value("fixed").run()
    at.sidebar.slider[0].set_value(5)
    at.run()
    return at.button(key="run_optimiser").click().run()


def test_what_if_tab_isolated_then_adopted():
    at = fixed_app()
    assert not at.exception
    plan = at.session_state["current"]
    assert plan["result"].solution.has_plan
    runs_before = at.session_state["solver_runs"]
    assert any("Bus-miles" in str(df.value.to_dict()) for df in at.dataframe)  # Overview distance row

    at.number_input(key="scn_scale").set_value(10)
    at.button(key="scn_run").click().run()
    assert not at.exception
    sc = at.session_state["scenario"]
    s_sol = sc["entry"]["result"].solution
    table = next(df.value for df in at.dataframe if {"Measure", "Scenario", "Difference"} <= set(df.value.columns))
    assert not table.astype(str).apply(lambda c: c.str.contains("nan")).any().any()  # no "+nan" cells
    assert table.set_index("Measure").loc["Status", "Difference"] == ""
    assert s_sol.has_plan and s_sol.total_attendees > plan["result"].solution.total_attendees
    # The current plan, its inputs and the solver counter are untouched.
    assert at.session_state["current"] is plan or at.session_state["current"]["key"] == plan["key"]
    assert at.session_state["overrides"] == {} and at.session_state["solver_runs"] == runs_before

    # Adopting needs the confirmation tick.
    assert at.button(key="scn_adopt").disabled
    at.checkbox(key="scn_confirm").check().run()
    at.button(key="scn_adopt").click().run()
    assert not at.exception
    cur = at.session_state["current"]
    assert cur["trigger"] == "scenario" and cur["result"].solution.total_attendees == s_sol.total_attendees
    assert at.session_state["overrides"] and at.session_state["scenario"] is None
    assert at.session_state["solver_runs"] == runs_before  # shown from the stored result, no new search
    assert at.session_state["previous"]["key"] == plan["key"]


def test_price_scenario_cannot_be_adopted():
    at = fixed_app()
    at.number_input(key="scn_price").set_value(20)
    at.button(key="scn_run").click().run()
    assert not at.exception
    sc = at.session_state["scenario"]
    assert sc["entry"]["key"] is None and sc["inputs"].changes_prices
    assert not [b for b in at.button if b.key == "scn_adopt"]
    assert any("for testing only" in i.value and "not used as the plan" in i.value for i in at.info)
    base = sc["base"]["result"].solution.total_cost
    assert sc["entry"]["result"].solution.total_cost == pytest.approx(base * 1.2)
