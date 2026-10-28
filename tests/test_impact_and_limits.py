"""Solver time limits, impact metrics and the Impact & Responsible AI tab.

Solvers are replaced by stubs where only the passed time limit matters, so
no test waits for a real 300-second search. Synthetic plans are labelled.
"""

from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

import benchmark
import src.planner as planner
from src.data import DEFAULT_TIME_LIMITS, MAX_TIME_LIMIT, MODEL_LABELS
from src.distances import SOURCE_APPROX, TimeMatrix
from src.model import FEASIBLE, INFEASIBLE, INVALID, OPTIMAL, BusRoute, Constraints, Solution
from src.report import ImpactMetrics, impact_metrics

APP = str(Path(__file__).resolve().parent.parent / "app.py")


# --------------------------------------------------------------------------- #
# Time limits
# --------------------------------------------------------------------------- #


def test_default_limits_are_300_s_for_routing_and_unchanged_for_fixed():
    assert DEFAULT_TIME_LIMITS == {"fixed": 20.0, "distance": 300.0, "time": 300.0}
    assert MAX_TIME_LIMIT >= 300


@pytest.fixture
def captured_limits(monkeypatch):
    """Replace both solvers with stubs that record the time limit they receive."""
    seen = {}

    def fake_routing(schools, rates, matrix, model, constraints=None, time_limit=30.0, **kw):
        seen[model] = time_limit
        return Solution(model=model, status=FEASIBLE, time_limit_s=time_limit)

    def fake_fixed(schools, rates, lat, lon, constraints=None, time_limit=30.0, **kw):
        seen["fixed"] = time_limit
        return Solution(model="fixed", status=OPTIMAL, time_limit_s=time_limit)

    monkeypatch.setattr(planner, "solve_routing", fake_routing)
    monkeypatch.setattr(planner, "solve_fixed", fake_fixed)
    return seen


def offline_matrix(dataset):
    n = len(dataset.schools) + 1
    return TimeMatrix(np.ones((n, n)), SOURCE_APPROX, [])


def test_default_limit_reaches_each_solver(real_dataset, captured_limits):
    for model in ("fixed", "distance", "time"):
        planner.optimise(real_dataset, model, time_matrix=offline_matrix(real_dataset))
    assert captured_limits == {"fixed": 20.0, "distance": 300.0, "time": 300.0}


def test_explicit_limit_overrides_default(real_dataset, captured_limits):
    planner.optimise(real_dataset, "distance", Constraints(), 45)
    planner.optimise(real_dataset, "time", Constraints(), 7, offline_matrix(real_dataset))
    assert captured_limits == {"distance": 45, "time": 7}


def test_non_positive_limit_rejected(real_dataset):
    with pytest.raises(ValueError, match="positive"):
        planner.optimise(real_dataset, "distance", Constraints(), 0)


def test_routing_stops_at_limit_with_best_valid_plan(real_dataset):
    """Real solver, short limit: it returns the best plan found, labelled heuristic."""
    sol = planner.optimise(real_dataset, "distance", Constraints(), 3).solution
    assert sol.status == FEASIBLE and sol.has_plan and not sol.proven_optimal
    assert sol.time_limit_s == 3 and sol.runtime_s < 3 + 10


# --------------------------------------------------------------------------- #
# Impact metrics (synthetic plans)
# --------------------------------------------------------------------------- #


def plan(model, buses, status=FEASIBLE):
    """SYNTHETIC plan. buses: list of (size, stops, attendees, cost)."""
    routes = [BusRoute(f"Bus {k + 1}", size, stops, att, 1.0, cost)
              for k, (size, stops, att, cost) in enumerate(buses)]
    return Solution(model=model, status=status, routes=routes, total_cost=sum(r.cost for r in routes))


BASE = plan("fixed", [(15, ["A"], 10, 40.0), (15, ["B"], 12, 40.0), (35, ["C"], 20, 60.0)], OPTIMAL)
OPT = plan("fixed", [(35, ["A", "B"], 22, 60.0), (35, ["C"], 20, 60.0)], OPTIMAL)


def test_impact_figures():
    im = impact_metrics(BASE, OPT)
    assert (im.baseline_cost, im.optimised_cost, im.saving) == (140.0, 120.0, 20.0)
    assert im.saving_percent == pytest.approx(20 / 140 * 100)
    assert im.attendees == 42 and im.schools_served == 3
    assert im.cost_per_student_before == pytest.approx(140 / 42)
    assert im.cost_per_student_after == pytest.approx(120 / 42)
    assert (im.baseline_buses, im.optimised_buses, im.buses_removed) == (3, 2, 1)
    assert im.optimised_utilisation == pytest.approx(42 / 70)
    assert im.baseline_utilisation == pytest.approx(42 / 65)
    assert im.proven_optimal and im.status_text == "Proven optimal"


def test_feasible_plan_is_not_reported_as_optimal():
    feasible = plan("fixed", [(35, ["A", "B"], 22, 60.0), (35, ["C"], 20, 60.0)], FEASIBLE)
    im = impact_metrics(BASE, feasible)
    assert not im.proven_optimal and im.status_text == "Feasible (optimality not proven)"


def test_zero_baseline_and_zero_attendees_are_safe():
    im = ImpactMetrics("fixed", "", False, 0, 0, 0.0, 0.0, 0, 0, 0.0, 0.0)
    assert im.saving_percent is None
    assert im.cost_per_student_before is None and im.cost_per_student_after is None


@pytest.mark.parametrize("status", [INFEASIBLE, INVALID])
def test_impact_refuses_plans_without_a_valid_result(status):
    bad = plan("fixed", [(35, ["A", "B"], 22, 60.0)], status)
    with pytest.raises(ValueError):
        impact_metrics(BASE, bad)


def test_impact_refuses_plan_with_violations():
    bad = plan("fixed", [(35, ["A", "B"], 22, 60.0), (35, ["C"], 20, 60.0)])
    bad.violations = ["Bus 1: 22 attendees exceed capacity 15."]
    with pytest.raises(ValueError):
        impact_metrics(BASE, bad)


def test_impact_refuses_mismatched_inputs():
    other_attendance = plan("fixed", [(35, ["A", "B"], 25, 60.0), (35, ["C"], 20, 60.0)])
    with pytest.raises(ValueError, match="different inputs"):
        impact_metrics(BASE, other_attendance)
    other_model = plan("distance", [(35, ["A", "B"], 22, 60.0), (35, ["C"], 20, 60.0)])
    with pytest.raises(ValueError, match="same cost model"):
        impact_metrics(BASE, other_model)


def test_real_fixed_impact_matches_plan(real_dataset):
    """Bus counts come from the plans, not from an assumed 60-school baseline."""
    res = planner.optimise(real_dataset, "fixed", Constraints(), 20)
    im = impact_metrics(res.baseline, res.solution)
    transported = int((real_dataset.schools["attendees"] > 0).sum())
    assert im.baseline_buses == transported == im.schools_served
    assert im.optimised_buses == len(res.solution.routes)
    assert im.buses_removed == transported - len(res.solution.routes)
    assert im.attendees == int(real_dataset.schools["attendees"].sum())
    assert im.cost_per_student_after == pytest.approx(res.solution.total_cost / im.attendees)


def test_impact_after_attendance_change_uses_new_inputs(real_dataset):
    edited = real_dataset.with_attendance({"NP11": 15})  # source value 18
    res = planner.optimise(edited, "fixed", Constraints(), 20)
    im = impact_metrics(res.baseline, res.solution)
    assert im.attendees == int(real_dataset.schools["attendees"].sum()) - 3
    assert im.cost_per_student_after == pytest.approx(res.solution.total_cost / im.attendees)


# --------------------------------------------------------------------------- #
# Benchmark report
# --------------------------------------------------------------------------- #


def test_benchmark_markdown_handles_missing_plans():
    rows = [
        {"model": "fixed", "label": "Fixed", "status": OPTIMAL, "status_text": "Proven optimal",
         "time_limit_s": 20.0, "runtime_s": 1.2, "route_data": "not used for cost", "notes": [],
         "optimised_cost": 1875.0, "optimised_buses": 25, "baseline_cost": 3330.0, "baseline_buses": 60,
         "saving": 1455.0, "saving_percent": 43.7, "cost_per_student_before": 2.68,
         "cost_per_student_after": 1.51},
        {"model": "time", "status": "UNAVAILABLE", "notes": ["cost file missing or invalid"]},
    ]
    md = benchmark.markdown(rows, "2026-01-01 00:00")
    assert "| Fixed | Proven optimal | 20 | 1.2 | 1,875.00 | 25 | 3,330.00 | 60 | 1,455.00 (43.7%)" in md
    assert "| time | UNAVAILABLE |" in md


# --------------------------------------------------------------------------- #
# Dashboard: Impact tab and time-limit control
# --------------------------------------------------------------------------- #


def metrics(at):
    return {m.label: m for m in at.metric}


def run(at):
    return at.button(key="run_optimiser").click().run()


def test_slider_default_matches_model_and_value_reaches_solver():
    at = AppTest.from_file(APP, default_timeout=180).run()
    assert at.sidebar.slider[0].value == 20  # fixed
    at.sidebar.radio[0].set_value("distance").run()
    assert at.sidebar.slider[0].value == 300
    assert any("Limit sent to the solver: 300 s" in c.value for c in at.sidebar.caption)
    at.sidebar.slider[0].set_value(5).run()
    run(at)
    assert at.session_state["current"]["result"].solution.time_limit_s == 5


def impact_table(at):
    """The Overview's baseline-vs-optimised table, indexed by measure."""
    frame = next(d.value for d in at.dataframe if "Measure" in d.value.columns)
    return frame.set_index("Measure")


def models_table(at):
    return next(d.value for d in at.dataframe if "Cost model" in d.value.columns)


def check_impact_figures(at, im):
    """Every impact figure on the Overview (cards and table) and the Compare table matches ``im``."""
    m, t = metrics(at), impact_table(at)
    assert m["Total cost"].value == f"{im.optimised_cost:,.2f}"
    assert m["Savings"].value == f"{im.saving:,.2f}"
    assert m["Buses"].value == str(im.optimised_buses)
    assert m["Attendees"].value == str(im.attendees)
    assert t.loc["Total cost", "One bus per school"] == f"{im.baseline_cost:,.2f}"
    assert t.loc["Total cost", "Optimised plan"] == f"{im.optimised_cost:,.2f}"
    assert t.loc["Buses", "One bus per school"] == str(im.baseline_buses)
    assert int(t.loc["Buses", "One bus per school"]) - int(t.loc["Buses", "Optimised plan"]) == im.buses_removed
    assert t.loc["Cost per attendee", "Optimised plan"] == f"{im.cost_per_student_after:,.2f}"
    assert t.loc["Cost per attendee", "One bus per school"] == f"{im.cost_per_student_before:,.2f}"
    row = models_table(at).set_index("Cost model").loc[MODEL_LABELS[im.model]]
    assert row["Optimised cost"] == pytest.approx(im.optimised_cost)
    assert row["Baseline cost"] == pytest.approx(im.baseline_cost)


def test_impact_figures_match_results_and_refresh_after_edit():
    at = AppTest.from_file(APP, default_timeout=180).run()
    assert not at.exception
    run(at)  # fixed model, 20 s
    res = at.session_state["current"]["result"]
    im = impact_metrics(res.baseline, res.solution)
    check_impact_figures(at, im)
    assert im.attendees == 1244

    # Pending edit (automatic re-optimisation off): figures are flagged as old,
    # and the all-models row is marked stale.
    at.sidebar.checkbox(key="auto_run_fixed").uncheck().run()
    at.session_state["overrides"] = {"NP11": 15}
    at.run()
    assert any("do not include your changes to: attendance" in w.value for w in at.warning)
    fixed_row = models_table(at)[models_table(at)["Cost model"].str.startswith("Fixed")].iloc[0]
    assert fixed_row["Status"].startswith("Not run with current inputs")

    # Rerun: every impact figure comes from the new plan.
    run(at)
    res2 = at.session_state["current"]["result"]
    im2 = impact_metrics(res2.baseline, res2.solution)
    assert im2.attendees == 1241
    check_impact_figures(at, im2)
    assert not any("do not include your changes" in w.value for w in at.warning)


def test_impact_tab_states_unimplemented_features_honestly():
    at = AppTest.from_file(APP, default_timeout=180).run()
    text = " ".join(x.value for x in list(at.info) + list(at.warning) + list(at.caption))
    assert "Proposed future enhancement, not implemented" in text
    assert "Final sign-off is a recommended procedure, not a built-in feature" in text
    assert "No pilot results exist" in text
    assert "not endorsed by the Welsh Government" in text
