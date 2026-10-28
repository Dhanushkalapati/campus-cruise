"""Explainability (src/explain.py) and map data (src/mapview.py). Synthetic data is SYNTHETIC."""

import numpy as np
import pandas as pd
import pytest

from src.data import Dataset
from src.explain import audit_report, explain_plan, model_difference, plan_totals, safe_percent, summary_row
from src.mapview import _mercator, bus_of_school, filter_routes, map_view, route_details, school_rows
from src.model import BusRoute, Constraints, Solution
from src.planner import optimise
from tests.conftest import FIXED, PER_MILE, make_schools

SCHOOLS = make_schools([("A", 51.60, -3.30, 30), ("B", 51.61, -3.31, 20), ("C", 51.70, -3.60, 12)])


def ds(rates):
    return Dataset(schools=SCHOOLS, rates=rates)


def entry(result, data, constraints=Constraints()):
    return {"result": result, "dataset": data, "model": result.solution.model, "constraints": constraints,
            "run_no": 1, "trigger": "manual", "finished_at": "12:00:00",
            "settings": {"time_limit": 5.0, "warm_start": False}}


@pytest.fixture(scope="module")
def fixed():
    data = ds({"fixed": FIXED})
    return data, optimise(data, "fixed", Constraints(), 5)


@pytest.fixture(scope="module")
def dist():
    data = ds({"distance": PER_MILE})
    return data, optimise(data, "distance", Constraints(), 2)


def test_fixed_explanation_is_grounded(fixed):
    data, res = fixed
    lines = explain_plan(res.solution, res.baseline, data, Constraints(), res.matrix)
    text = " ".join(t for _, t in lines)
    assert res.solution.proven_optimal and "No plan can cost less" in text   # 115 = lower bound
    assert "cheapest available size" in text
    assert f"{res.baseline.total_cost - res.solution.total_cost:,.2f} saved" in text
    assert all(k in ("fact", "interpretation") for k, _ in lines)


def test_unproven_plan_is_never_called_optimal(dist):
    data, res = dist
    lines = explain_plan(res.solution, res.baseline, data, Constraints(), res.matrix)
    facts = " ".join(t for k, t in lines if k == "fact")
    assert "Feasible (optimality not proven)" in facts and "No plan can cost less" not in facts


def test_wrong_bus_size_is_reported(fixed):
    data, res = fixed
    sol = Solution(model="fixed", status="OPTIMAL", routes=[
        BusRoute("Bus 01", 50, ["A"], 30, 0, 75.0), BusRoute("Bus 02", 35, ["B"], 20, 0, 60.0),
        BusRoute("Bus 03", 15, ["C"], 12, 0, 40.0)], total_cost=175.0)
    text = " ".join(t for _, t in explain_plan(sol, res.baseline, data, Constraints(), res.matrix))
    assert "Bus 01 (50-seat, 35-seat would fit)" in text


def test_summary_row_and_report_use_same_figures(fixed):
    data, res = fixed
    e = entry(res, data)
    row = summary_row(e, True, res.matrix, None, None)
    sol, base = res.solution, res.baseline
    assert row["Optimised cost"] == round(sol.total_cost, 2) and row["Baseline cost"] == round(base.total_cost, 2)
    assert row["Change %"] == round(-(base.total_cost - sol.total_cost) / base.total_cost * 100, 1)
    assert row["Driving hours"] is None and row["Hours source"] == "not available"
    md = audit_report(e, data, [{"run": 1}], pd.DataFrame([row]), res.matrix, current=True)
    assert f"{sol.total_cost:,.2f}" in md and "Independent validation: passed" in md
    assert "Out of date" not in md
    for heading in ("Executive summary", "Baseline methodology", "Constraint validation",
                    "Limitations", "Responsible AI"):
        assert heading in md
    stale = audit_report(e, data, [], pd.DataFrame(), res.matrix, current=False)
    assert "Out of date" in stale


def test_report_refuses_invalid_plan(fixed):
    data, res = fixed
    bad = Solution(model="fixed", status="INVALID", routes=res.solution.routes, violations=["x"])
    e = entry(type(res)(bad, res.baseline, res.matrix), data)
    with pytest.raises(ValueError):
        audit_report(e, data, [], pd.DataFrame(), res.matrix, current=True)


def test_plan_totals_with_and_without_hours(dist):
    data, res = dist
    sol = res.solution
    t = plan_totals(sol, data, res.matrix, res.matrix / 30)
    assert t["miles"] == pytest.approx(sum(r.metric for r in sol.routes))
    assert t["hours"] == pytest.approx(t["miles"] / 30)
    assert plan_totals(sol, data, res.matrix, None)["hours"] is None


def test_safe_percent():
    assert safe_percent(1, 4) == 25.0
    assert safe_percent(1, 0) is None and safe_percent(float("nan"), 3) is None


def test_model_difference(fixed, dist):
    a, b = fixed[1].solution, dist[1].solution
    lines = model_difference(a, b)
    assert "not comparable" in lines[-1]


def test_map_filters_and_details_stay_in_step(fixed):
    data, res = fixed
    sol = res.solution
    assert filter_routes(sol) == sol.routes and filter_routes(sol, []) == sol.routes
    bus = sol.routes[0].bus_id
    only = filter_routes(sol, [bus])
    assert [r.bus_id for r in only] == [bus]
    school = sol.routes[-1].stops[0]
    picked = filter_routes(sol, None, school)
    assert len(picked) == 1 and school in picked[0].stops
    assert bus_of_school(sol, school) == picked[0].bus_id and bus_of_school(sol, "ZZ") is None
    details = route_details(only, sol, data, res.matrix, None, None)
    r = only[0]
    assert details.loc[0, "Attendees"] == r.attendees and details.loc[0, "Seats left"] == r.bus_size - r.attendees
    assert details.loc[0, "Time source"] == "not available"
    schools = school_rows(only, data)
    assert set(schools["School"]) == set(r.stops) and list(schools["Pickup #"]) == list(range(1, len(r.stops) + 1))


def test_route_details_hours_match_matrix(dist):
    data, res = dist
    hours = np.full_like(res.matrix, 0.5)
    np.fill_diagonal(hours, 0)
    d = route_details(res.solution.routes, res.solution, data, res.matrix, hours, "TEST SOURCE")
    for _, row in d.iterrows():
        assert row["Driving minutes"] == round(row["Schools"] * 0.5 * 60)  # one leg per stop
        assert row["Time source"] == "TEST SOURCE"


# --------------------------------------------------------------------------- #
# Dashboard (AppTest, real data, fixed model with a short limit)
# --------------------------------------------------------------------------- #

def test_map_view_fits_every_point_at_a_sensible_zoom():
    # Regression: the map opened at maximum zoom on an empty field, because Leaflet's
    # fitBounds ran while the map's tab was hidden. The view is now computed in Python.
    from src.data import load_dataset
    ds = load_dataset()
    pts = list(zip(ds.schools["latitude"], ds.schools["longitude"])) + [(ds.event.latitude, ds.event.longitude)]
    (lat, lon), zoom = map_view(pts, 640, 480)
    assert 5 <= zoom <= 12                                   # region-level, not street-level
    lats, lons = [p[0] for p in pts], [p[1] for p in pts]
    assert min(lats) < lat < max(lats) and min(lons) < lon < max(lons)
    # Every point lies inside the 640 x 480 view at that zoom.
    scale = 256 * 2 ** zoom
    cx, cy = _mercator(lat, lon)
    for p in pts:
        x, y = _mercator(*p)
        assert abs(x - cx) * scale <= 320 and abs(y - cy) * scale <= 240


def test_map_view_single_point_and_empty():
    (lat, lon), zoom = map_view([(51.48, -3.18)])
    assert (round(lat, 4), round(lon, 4)) == (51.48, -3.18) and zoom == 14
    with pytest.raises(ValueError):
        map_view([])


from pathlib import Path  # noqa: E402

from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture
def app():
    at = AppTest.from_file(APP, default_timeout=240).run()
    at.sidebar.slider[0].set_value(5).run()
    return at.button(key="run_optimiser").click().run()


def frame_with(at, column):
    return next(d.value for d in at.dataframe if column in d.value.columns)


def test_map_filters_keep_table_in_step(app):
    at = app
    sol = at.session_state["current"]["result"].solution
    runs = at.session_state["solver_runs"]
    table = frame_with(at, "Seats left")
    assert list(table["Bus"]) == [r.bus_id for r in sol.routes]           # whole plan by default
    bus = sol.routes[3]
    at.multiselect(key="map_buses").set_value([bus.bus_id]).run()
    table = frame_with(at, "Seats left")
    assert list(table["Bus"]) == [bus.bus_id] and table.iloc[0]["Seats left"] == bus.empty_seats
    m = {x.label: x.value for x in at.metric}
    assert m["Passengers"] == f"{bus.attendees} / {bus.bus_size}"
    shares = frame_with(at, "Cost share")
    assert set(shares["Bus"]) == {bus.bus_id}
    school = sol.routes[0].stops[-1]
    at.multiselect(key="map_buses").set_value([]).run()
    at.selectbox(key="map_school").set_value(school).run()
    assert list(frame_with(at, "Seats left")["Bus"]) == [sol.routes[0].bus_id]
    assert any(f"{school} rides" in c.value for c in at.caption)
    at.button(key="map_clear").click().run()
    assert len(frame_with(at, "Seats left")) == len(sol.routes)
    assert at.session_state["solver_runs"] == runs                         # filters never run the solver


def test_audit_export_and_run_log(app):
    at = app
    sol = at.session_state["current"]["result"].solution
    log = at.session_state["run_log"]
    assert len(log) == 1 and log[0]["Validation"] == "passed" and log[0]["Cost"] == round(sol.total_cost, 2)
    assert any("Why this plan?" in e.label for e in at.expander)
    labels = [b.label for b in at.get("download_button")]
    assert "Download audit report (Markdown)" in labels and "Download model metrics (CSV)" in labels
    preview = " ".join(md.value for md in at.markdown)
    assert "Campus Cruise audit report" in preview and f"{sol.total_cost:,.2f}" in preview


def test_ml_section_is_honest(app):
    text = " ".join(md.value for md in app.markdown) + " ".join(c.value for c in app.caption)
    assert "Observed journey records:** 0" in text
    assert "not real journeys" in text or "No trained ML artefact" in " ".join(i.value for i in app.info)


def test_map_can_show_another_models_plan_across_runs(app):
    at = app
    fixed = at.session_state["current"]
    at.sidebar.radio[0].set_value("distance").run()
    at.sidebar.slider[0].set_value(5).run()
    at.button(key="run_optimiser").click().run()
    at.selectbox(key="map_plan").set_value("fixed").run()
    assert any("Other model" not in md.value and "Showing:" in md.value and "Fixed" in md.value
               for md in at.markdown)
    assert list(frame_with(at, "Seats left")["Bus"]) == [r.bus_id for r in fixed["result"].solution.routes]
    at.button(key="run_optimiser").click().run()       # a new run changes run numbers, not the labels
    assert not at.exception
