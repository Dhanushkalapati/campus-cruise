"""Assistant -> approval -> real optimiser -> validator -> factual summary.

LLM calls go to a scripted FakeProvider (no key, no network). The optimiser,
validator and data are real. The post-run summary is built by the app from
the stored result, never by the language model.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import src.assistant as assistant
import src.planner
from src.assistant import validate_proposal
from src.data import load_dataset
from src.model import INVALID, Constraints, Solution
from src.planner import PlanResult
from tests.assistant_eval_cases import llm
from tests.test_assistant import FakeProvider

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def with_prefs(raw, prefs):
    raw = dict(raw)
    raw["preferences"] = prefs
    return raw


def app_with(monkeypatch, provider):
    monkeypatch.setattr(assistant, "provider_from_config", lambda secrets=None: provider)
    at = AppTest.from_file(APP, default_timeout=240).run()
    for key in ("time_limit_fixed", "time_limit_distance", "time_limit_time"):
        at.session_state[key] = 5  # every model short; set before its slider is drawn
    return at.run()


def send(at, text):
    return at.chat_input(key="chat_input").set_value(text).run()


def last_summary(at):
    return next(m for m in reversed(at.session_state["chat_history"]) if m.get("summary"))


# --- validation only (no app) -------------------------------------------------


@pytest.fixture(scope="module")
def data():
    return load_dataset()


@pytest.mark.parametrize("model", ["fixed", "distance", "time"])
def test_objective_requests_map_to_the_right_model(data, model):
    p = validate_proposal(llm("run_optimiser", "Run.", model=model), "x", data, Constraints(), "fixed")
    assert p.needs_approval and p.cost_model == model


def test_group_nearby_applied_only_where_the_optimiser_supports_it(data):
    fixed = validate_proposal(with_prefs(llm("run_optimiser", model="fixed"), ["group_nearby"]),
                              "Keep nearby schools together and minimise cost.", data, Constraints(), "fixed")
    assert fixed.preferences_applied == ["group_nearby"] and any("never raises the cost" in n for n in fixed.notes)
    dist = validate_proposal(with_prefs(llm("run_optimiser", model="distance"), ["group_nearby"]),
                             "x", data, Constraints(), "fixed")
    assert dist.preferences_not_applied == ["group_nearby"] and dist.preferences_applied == []


def test_fewer_buses_is_reported_not_applied(data):
    p = validate_proposal(with_prefs(llm("run_optimiser", model="fixed"), ["fewer_buses"]),
                          "Reduce the number of buses.", data, Constraints(), "fixed")
    assert p.needs_approval and p.preferences_not_applied == ["fewer_buses"]
    assert any("At least 25 buses are needed (1244 attendees" in n for n in p.notes)


def test_bypassing_capacity_is_refused(data):
    p = validate_proposal(llm("attendance_change", att=[("NP11", 80)]), "Put 80 from NP11 on one bus.",
                          data, Constraints(), "fixed")
    assert not p.needs_approval and any("never split" in e for e in p.errors)


# --- full pipeline in the app ---------------------------------------------------


def test_approved_cost_request_runs_real_optimiser_and_reports_real_figures(monkeypatch):
    raw = with_prefs(llm("run_optimiser", "Cheapest plan, nearby schools together.", model="fixed"),
                     ["group_nearby"])
    at = app_with(monkeypatch, FakeProvider(raw))
    send(at, "Keep nearby schools together and minimise cost.")
    assert at.session_state["solver_runs"] == 0          # nothing runs before approval
    assert at.session_state["assistant_proposal"].preferences_applied == ["group_nearby"]
    at.button(key="chat_approve").click().run()
    assert not at.exception
    cur = at.session_state["current"]
    sol, base = cur["result"].solution, cur["result"].baseline
    assert at.session_state["solver_runs"] == 1 and cur["trigger"] == "assistant"
    text = last_summary(at)["text"]
    assert sol.method in text and sol.status_text() in text
    assert f"{sol.total_cost:,.2f}" in text and f"{base.total_cost:,.2f}" in text
    assert f"{len(sol.routes)} buses" in text
    assert "Keep nearby schools together: applied" in text
    assert "not by the language model" in text
    log = at.session_state["run_log"][-1]
    assert log["Trigger"] == "assistant" and log["Validation"] == "passed"


@pytest.mark.parametrize("model", ["distance", "time"])
def test_approved_routing_request_runs_that_model(monkeypatch, model):
    at = app_with(monkeypatch, FakeProvider(llm("run_optimiser", "Run.", model=model)))
    if model == "time":
        at.session_state["time_source_choice"] = "estimate"  # no network in tests
        at.run()
    send(at, "Prioritise the shortest overall journey." if model == "distance" else "Prioritise travel time.")
    at.button(key="chat_approve").click().run()
    cur = at.session_state["current"]
    assert cur["model"] == model and cur["result"].solution.status == "FEASIBLE"
    text = last_summary(at)["text"]
    assert "Proven optimal: no" in text
    if model == "time":
        assert "APPROXIMATE" in text  # the estimate is labelled in the summary


def test_result_failing_validation_is_reported_as_failed(monkeypatch):
    at = app_with(monkeypatch, FakeProvider(llm("run_optimiser", "Run.", model="fixed")))
    real = src.planner.optimise

    def invalid(dataset, model, *a, **k):
        res = real(dataset, model, *a, **k)
        bad = Solution(model=model, status=INVALID, routes=res.solution.routes,
                       violations=["Bus 01: 60 attendees exceed capacity 50."])
        return PlanResult(bad, res.baseline, res.matrix)

    monkeypatch.setattr(src.planner, "optimise", invalid)
    send(at, "Find the cheapest plan.")
    at.button(key="chat_approve").click().run()
    s = last_summary(at)
    assert s["kind"] == "error" and "failed independent validation" in s["text"]
    assert "Total cost" not in {m.label for m in at.metric}
    assert at.session_state["run_log"][-1]["Validation"].startswith("failed")


def test_optimiser_exception_is_reported_and_never_claimed_as_run(monkeypatch):
    at = app_with(monkeypatch, FakeProvider(llm("run_optimiser", "Run.", model="fixed")))

    def boom(*a, **k):
        raise RuntimeError("solver crashed")

    monkeypatch.setattr(src.planner, "optimise", boom)
    send(at, "Find the cheapest plan.")
    at.button(key="chat_approve").click().run()
    s = last_summary(at)
    assert s["kind"] == "error" and "solver crashed" in s["text"] and "No result was replaced" in s["text"]
    assert at.session_state["current"] is None
    # The approval message never says the optimiser already ran.
    approvals = [m["text"] for m in at.session_state["chat_history"] if m.get("kind") == "success"]
    assert approvals and all("ran on" not in t for t in approvals)
