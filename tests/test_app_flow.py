"""Dashboard flow with Streamlit's AppTest (headless; no browser).

Uses the real data files. Attendance edits are injected into session state
(``overrides``), which is where the editor's on_change callback writes them;
AppTest cannot type into st.data_editor cells. The callback's parsing is
covered by test_attendance_reoptimisation.py::test_F_editor_*.
"""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def money(x):
    return f"{x:,.2f}"


def metrics(at):
    return {m.label: m.value for m in at.metric}


def start(model="distance"):
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    at.sidebar.radio[0].set_value(model).run()  # the model's own slider appears after this run
    at.sidebar.slider[0].set_value(5)
    return at.run()


def run_optimiser(at):
    return at.button(key="run_optimiser").click().run()


def test_attendance_edit_flow_end_to_end():
    at = run_optimiser(start("distance"))
    assert not at.exception
    first = at.session_state["current"]
    assert first["model"] == "distance" and first["result"].solution.has_plan
    assert metrics(at)["Attendees"] == "1244"

    # Edit pending: old results stay, but the page says they are stale.
    at.session_state["overrides"] = {"NP11": 15}
    at.run()
    assert any("do not include your changes to: attendance" in w.value for w in at.warning)
    assert metrics(at)["Attendees"] == "1244"

    # Run: every figure comes from the new plan.
    run_optimiser(at)
    cur = at.session_state["current"]
    sol, base = cur["result"].solution, cur["result"].baseline
    ds = cur["dataset"]
    assert cur["overrides"] == {"NP11": 15}
    assert int(ds.schools.loc[ds.schools.school_id == "NP11", "attendees"].iloc[0]) == 15
    m = metrics(at)
    assert m["Attendees"] == "1241"
    assert m["Total cost"] == money(sol.total_cost)
    assert m["Savings"] == money(base.total_cost - sol.total_cost)
    assert base.total_attendees == 1241
    assert not any("do not include your changes" in w.value for w in at.warning)
    # Compare uses the previous valid plan and the latest one.
    assert at.session_state["previous"]["result"] is first["result"]
    assert m["Previous plan cost"] == money(first["result"].solution.total_cost)
    assert m["Revised plan cost"] == money(sol.total_cost)

    # Invalid edit blocks the run and keeps the last plan.
    at.session_state["attendance_errors"] = ["NP11: attendance is empty."]
    run_optimiser(at)
    assert at.session_state["current"]["result"] is cur["result"]
    assert any("Not optimised. Invalid attendance" in e.value for e in at.error)

    # Reset restores the source attendance. Those inputs were optimised before,
    # so the stored plan is reused at once: no solver run, no stale warning.
    runs = at.session_state["solver_runs"]
    at.button(key="reset_attendance").click().run()
    assert at.session_state["overrides"] == {} and at.session_state["attendance_errors"] == []
    assert at.session_state["solver_runs"] == runs
    assert at.session_state["current"] is first
    assert not any("do not include your changes" in w.value for w in at.warning)
    assert metrics(at)["Attendees"] == "1244"


def test_infeasible_edit_never_shows_old_plan_as_new():
    at = run_optimiser(start("distance"))
    good = at.session_state["current"]
    at.session_state["overrides"] = {"NP11": 60}
    run_optimiser(at)
    cur = at.session_state["current"]
    assert not cur["result"].solution.has_plan
    assert any("No valid plan" in e.value and "NP11" in e.value for e in at.error)
    assert "Total cost" not in metrics(at)  # no KPI cards from the old plan
    assert at.session_state["previous"] is good  # kept only as the comparison point


def test_model_switch_runs_selected_model():
    at = run_optimiser(start("distance"))
    # The fixed model re-optimises automatically (exact, a few seconds).
    at.sidebar.radio[0].set_value("fixed").run()
    cur = at.session_state["current"]
    assert cur["model"] == "fixed" and cur["trigger"] == "auto"
    assert cur["result"].solution.time_limit_s == 20  # each model has its own slider and default
    assert not any("do not include your changes" in w.value for w in at.warning)
    # A new time limit is a setting: flagged, but not run automatically.
    at.sidebar.slider[0].set_value(5).run()
    assert at.session_state["current"] is cur
    assert any("used a 20 s limit" in i.value for i in at.sidebar.info)
    run_optimiser(at)
    cur = at.session_state["current"]
    assert cur["model"] == "fixed"
    assert cur["result"].solution.status in ("OPTIMAL", "FEASIBLE")
    assert cur["result"].solution.time_limit_s == 5


def test_time_model_labels_its_travel_time_source():
    at = start("time")
    at.sidebar.checkbox(key="allow_online").uncheck().run()  # no network request
    run_optimiser(at)
    cur = at.session_state["current"]
    sol = cur["result"].solution
    assert cur["model"] == "time" and sol.has_plan
    # Cached road times if present, otherwise the labelled estimate. Never unlabelled.
    assert sol.matrix_source.startswith(("OSRM road travel times", "APPROXIMATE"))
    assert any(sol.matrix_source in md.value for md in at.markdown)
