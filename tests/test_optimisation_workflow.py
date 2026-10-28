"""Optimisation workflow in the dashboard (Streamlit AppTest, real data).

Covers: when the solver runs, result reuse, stale-result labels, automatic
runs, duplicate-run prevention, and failure handling. ``solver_runs`` in
session state counts every optimise() call. Tabs and expanders are
client-side in Streamlit and never rerun the script; the tests below drive the
widgets that do rerun it.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import src.planner
from src.model import NO_SOLUTION, Solution
from src.planner import PlanResult

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def start(model="distance", limit=5):
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    at.sidebar.radio[0].set_value(model).run()  # the model's own slider appears after this run
    at.sidebar.slider[0].set_value(limit)
    return at.run()


def runs(at):
    return at.session_state["solver_runs"]


def warnings(at):
    return [w.value for w in at.warning]


def errors(at):
    return [e.value for e in at.error]


def test_page_load_never_runs_solver():
    at = start("fixed")  # fixed has auto-run on, but only after a first deliberate run
    at.run()
    assert runs(at) == 0 and at.session_state["current"] is None
    assert at.button(key="run_optimiser_main")  # explicit action offered


def test_unrelated_widgets_do_not_rerun_solver():
    at = start("distance")
    at.button(key="run_optimiser").click().run()
    first = at.session_state["current"]
    assert runs(at) == 1
    at.selectbox(key="must_share_a").set_value("CF15").run()   # picking, not adding, a pair
    at.button(key="map_clear").click().run()                   # map filter reset
    at.run().run()                                             # plain rerenders
    assert runs(at) == 1 and at.session_state["current"] is first
    assert any("up to date" in s.value for s in at.sidebar.success)


def test_routing_model_waits_for_explicit_run():
    at = start("distance")
    at.button(key="run_optimiser").click().run()
    first = at.session_state["current"]
    at.session_state["overrides"] = {"NP11": 15}
    at.run().run()
    assert runs(at) == 1 and at.session_state["current"] is first
    assert any("Optimisation required" in w for w in warnings(at))
    assert any("Out of date" in md.value for md in at.markdown)  # pill on Overview
    at.button(key="run_optimiser_main").click().run()
    cur = at.session_state["current"]
    assert runs(at) == 2 and cur["overrides"] == {"NP11": 15} and cur["trigger"] == "manual"
    assert not any("Optimisation required" in w for w in warnings(at))


def test_fixed_model_auto_runs_once_per_change():
    at = start("fixed")
    at.button(key="run_optimiser").click().run()
    assert runs(at) == 1
    at.session_state["overrides"] = {"NP11": 15}
    at.run()
    cur = at.session_state["current"]
    assert runs(at) == 2 and cur["trigger"] == "auto" and cur["overrides"] == {"NP11": 15}
    # Same inputs again: no duplicate run.
    at.run().run()
    assert runs(at) == 2 and at.session_state["current"] is cur
    # Display of the new run: figures come from it.
    m = {x.label: x.value for x in at.metric}
    assert m["Attendees"] == "1241"
    assert m["Total cost"] == f"{cur['result'].solution.total_cost:,.2f}"
    # The runtime is no longer a headline card; it is in the run caption and the run log.
    assert "Solver runtime" not in m
    sol = cur["result"].solution
    assert any(f"run #{cur['run_no']}" in c.value and "time limit 5 s" in c.value
               and f"finished in {sol.runtime_s:.1f} s" in c.value for c in at.caption)
    assert at.session_state["run_log"][-1]["Runtime (s)"] == round(sol.runtime_s, 1)


def test_auto_run_can_be_switched_off():
    at = start("fixed")
    at.button(key="run_optimiser").click().run()
    at.sidebar.checkbox(key="auto_run_fixed").uncheck().run()
    at.session_state["solo"] = ["NP11"]
    at.run()
    assert runs(at) == 1
    assert any("constraints" in w and "Optimisation required" in w for w in warnings(at))


def test_cached_result_reused_when_inputs_return():
    at = start("distance")
    at.button(key="run_optimiser").click().run()
    first = at.session_state["current"]
    at.session_state["overrides"] = {"NP11": 15}
    at.button(key="run_optimiser").click().run()
    second = at.session_state["current"]
    assert runs(at) == 2
    at.session_state["overrides"] = {}
    at.run()
    assert runs(at) == 2 and at.session_state["current"] is first
    assert at.session_state["previous"] is second
    assert any("Reused the stored result" in i.value for i in at.info)


def test_cache_invalidated_by_changed_inputs_and_model():
    at = start("distance")
    at.button(key="run_optimiser").click().run()
    dist = at.session_state["current"]
    at.sidebar.radio[0].set_value("fixed").run()  # auto: fixed is new for these inputs
    assert runs(at) == 2 and at.session_state["current"]["model"] == "fixed"
    at.sidebar.radio[0].set_value("distance").run()  # back: cached, no run
    assert runs(at) == 2 and at.session_state["current"] is dist
    at.session_state["cannot_share"] = [("CF15", "CF38")]
    at.run()  # distance: auto off, so stale
    assert runs(at) == 2 and at.session_state["current"] is dist
    assert any("constraints" in w for w in warnings(at))


def test_manual_rerun_on_current_inputs_searches_again():
    at = start("distance")
    at.button(key="run_optimiser").click().run()
    first = at.session_state["current"]
    at.button(key="run_optimiser").click().run()
    cur = at.session_state["current"]
    assert runs(at) == 2 and cur is not first and cur["key"] == first["key"]
    assert at.session_state["result_cache"][cur["key"]] is cur


def test_solver_exception_keeps_last_result_and_blocks_auto_retry(monkeypatch):
    at = start("fixed")
    at.button(key="run_optimiser").click().run()
    good = at.session_state["current"]
    calls = []

    def boom(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("solver crashed")

    monkeypatch.setattr(src.planner, "optimise", boom)
    at.session_state["overrides"] = {"NP11": 15}
    at.run()
    assert len(calls) == 1
    assert at.session_state["current"] is good and at.session_state["last_valid"] is good
    assert any("solver crashed" in e and "No result was replaced" in e for e in errors(at))
    assert any("Optimisation required" in w for w in warnings(at))  # old plan labelled stale
    at.run().run()
    assert len(calls) == 1  # failed auto-run is not repeated for the same inputs
    assert any("already tried" in c.value for c in at.caption)


def test_timeout_on_same_inputs_keeps_valid_plan(monkeypatch):
    at = start("distance")
    at.button(key="run_optimiser").click().run()
    good = at.session_state["current"]

    def timeout(dataset, model, *args, **kwargs):
        sol = Solution(model=model, status=NO_SOLUTION, time_limit_s=5, runtime_s=5.0)
        return PlanResult(sol, sol, None)

    monkeypatch.setattr(src.planner, "optimise", timeout)
    at.button(key="run_optimiser").click().run()
    assert at.session_state["current"] is good
    assert any("existing plan for these inputs is kept" in e for e in errors(at))
    m = {x.label: x.value for x in at.metric}
    assert m["Total cost"] == f"{good['result'].solution.total_cost:,.2f}"


def test_infeasible_new_inputs_show_last_valid_separately():
    at = start("distance")
    at.button(key="run_optimiser").click().run()
    good = at.session_state["current"]
    at.session_state["overrides"] = {"NP11": 60}  # bigger than any bus
    at.button(key="run_optimiser").click().run()
    assert not at.session_state["current"]["result"].solution.has_plan
    assert at.session_state["last_valid"] is good
    assert any("Last valid plan kept for comparison" in i.value for i in at.info)
    assert "Total cost" not in {x.label for x in at.metric}


@pytest.mark.parametrize("model", ["distance"])
def test_invalid_attendance_blocks_auto_and_manual_runs(model):
    at = start(model)
    at.button(key="run_optimiser").click().run()
    at.session_state["attendance_errors"] = ["NP11: attendance is empty."]
    at.button(key="run_optimiser").click().run()
    assert runs(at) == 1
    assert any("Not optimised. Invalid attendance" in e for e in errors(at))
