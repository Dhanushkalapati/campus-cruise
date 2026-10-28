"""Uploaded school data and a configurable event destination.

All school and venue data here is SYNTHETIC (made-up outcodes and coordinates
around Bristol), except tests that load the bundled default files. OSRM is
never contacted: a fake session stands in for the network.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from src.data import (
    DEFAULT_EVENT,
    Event,
    check_school_upload,
    far_schools,
    load_dataset,
    make_event,
    parse_venue_csv,
)
from src.distances import (
    OSRM_MAX_POINTS,
    SOURCE_APPROX,
    SOURCE_OSRM_CACHE,
    SOURCE_OSRM_LIVE,
    RoutingServiceError,
    _cache_key,
    haversine_miles,
    road_time_matrix,
    straight_line_matrix,
)
from src.mapview import route_details
from src.model import Constraints
from src.planner import optimise, points_of
from src.report import plan_csv
from src.validation import route_metric, validate_solution

ROOT = Path(__file__).resolve().parent.parent
APP = str(ROOT / "app.py")

# SYNTHETIC: eight pickup points around Bristol and a venue in the city.
BRISTOL = Event("Synthetic Arena, Bristol", 51.4400, -2.6200)
FAR_VENUE = Event("Synthetic Hall, Bath", 51.3800, -2.3600)
BRISTOL_CSV = (
    "outcode,latitude,longitude,attendants\n"
    "BS1,51.4545,-2.5879,22\n"
    "BS2,51.4600,-2.5700,9\n"
    "BS3,51.4400,-2.6000,14\n"
    "BS4,51.4300,-2.5600,30\n"
    "BS5,51.4650,-2.5500,6\n"
    "BS6,51.4750,-2.6050,18\n"
    "BS7,51.4900,-2.5800,11\n"
    "BS8,51.4550,-2.6200,25\n"
).encode()


def csv_bytes(*rows: str, header: str = "outcode,latitude,longitude,attendants") -> bytes:
    return ("\n".join((header,) + rows) + "\n").encode()


def bristol(event: Event = BRISTOL):
    return load_dataset(ROOT, schools_csv=BRISTOL_CSV, event=event)


class FakeResponse:
    def __init__(self, body, status=200):
        self.body, self.status_code = body, status

    def json(self):
        return self.body


class FakeOSRM:
    """Records every table request and answers with plausible durations (60 s per straight-line mile)."""

    def __init__(self, fail: bool = False):
        self.urls, self.fail = [], fail

    def get(self, url, params=None, timeout=None):
        self.urls.append(url)
        if self.fail:
            return FakeResponse({}, status=503)
        coords = url.rsplit("/", 1)[1].split(";")
        points = [(float(c.split(",")[1]), float(c.split(",")[0])) for c in coords]
        seconds = (straight_line_matrix(points) * 60).tolist()
        return FakeResponse({"code": "Ok", "durations": seconds})


def grid_points(n: int) -> list[tuple[float, float]]:
    return [(51.40 + (k // 10) * 0.01, -2.70 + (k % 10) * 0.01) for k in range(n)]


# --------------------------------------------------------------------------- #
# School upload validation
# --------------------------------------------------------------------------- #


def test_valid_upload_is_accepted_with_counts():
    up = check_school_upload(BRISTOL_CSV)
    assert up.ok and up.errors == []
    assert len(up.schools) == 8 and up.total_attendees == 135
    assert list(up.schools.columns) == ["school_id", "name", "latitude", "longitude", "attendees"]


def test_upload_accepts_bom_case_and_spaces():
    content = "﻿Outcode, Latitude ,Longitude,ATTENDANTS\nX1, 51.45 ,-2.59, 10\n".encode("utf-8")
    up = check_school_upload(content)
    assert up.ok and up.schools.iloc[0]["attendees"] == 10


def test_missing_required_columns_are_named():
    up = check_school_upload(csv_bytes("BS1,51.45,-2.58", header="outcode,latitude,longitude"))
    assert not up.ok and "attendants" in up.errors[0] and up.schools.empty


@pytest.mark.parametrize("row,problem", [
    ("BS1,95.0,-2.58,10", "latitude"),
    ("BS1,51.45,-200,10", "longitude"),
    ("BS1,north,-2.58,10", "latitude"),
    ("BS1,,-2.58,10", "latitude"),
])
def test_invalid_coordinates_reject_the_whole_file(row, problem):
    up = check_school_upload(csv_bytes("BS0,51.44,-2.60,5", row))
    assert not up.ok and up.schools.empty  # the valid row is not used on its own either
    assert "row 3 (BS1)" in up.errors[0] and problem in up.errors[0] and "no row was dropped" in up.errors[0]


@pytest.mark.parametrize("value,problem", [
    ("", "missing attendance"), ("-4", "negative attendance"), ("12.5", "non-integer attendance"),
    ("lots", "missing attendance"),
])
def test_missing_and_invalid_attendee_counts(value, problem):
    up = check_school_upload(csv_bytes(f"BS1,51.45,-2.58,{value}"))
    assert not up.ok and problem in up.errors[0]


def test_missing_outcode_is_reported_by_row_number():
    up = check_school_upload(csv_bytes("BS1,51.45,-2.58,4", ",51.46,-2.57,3"))
    assert not up.ok and "row 3" in up.errors[0] and "missing outcode" in up.errors[0]


@pytest.mark.parametrize("content,needle", [
    (b"", "empty"), (b"   \n", "empty"), (csv_bytes(), "no schools"),
    ("outcode,latitude,longitude,attendants\nB\xe9,51,-2,3\n".encode("latin-1"), "UTF-8"),
])
def test_empty_or_unreadable_files(content, needle):
    up = check_school_upload(content)
    assert not up.ok and needle in up.errors[0]


def test_oversized_file_is_rejected():
    assert not check_school_upload(BRISTOL_CSV, max_bytes=50).ok


def test_all_zero_attendance_has_nothing_to_plan():
    up = check_school_upload(csv_bytes("BS1,51.45,-2.58,0", "BS2,51.46,-2.57,0"))
    assert not up.ok and "zero" in up.errors[0]


def test_duplicate_outcodes_are_an_error():
    up = check_school_upload(csv_bytes("BS1,51.45,-2.58,4", "BS1,51.45,-2.58,4"))
    assert not up.ok and any("Duplicate school identifiers: BS1" in e for e in up.errors)


def test_same_coordinates_under_different_outcodes_is_a_warning():
    up = check_school_upload(csv_bytes("BS1,51.45,-2.58,4", "BS9,51.45,-2.58,7"))
    assert up.ok and any("identical coordinates" in w and "BS1, BS9" in w for w in up.warnings)


def test_zero_attendance_school_and_extra_columns_are_warnings():
    up = check_school_upload(csv_bytes("BS1,51.45,-2.58,4,x", "BS2,51.46,-2.57,0,y",
                                       header="outcode,latitude,longitude,attendants,notes"))
    assert up.ok
    assert any("zero attendees" in w and "BS2" in w for w in up.warnings)
    assert any("Extra column(s) ignored: notes" in w for w in up.warnings)


def test_upload_never_writes_the_bundled_file():
    before = (ROOT / "attendants_dataset.csv").read_bytes()
    check_school_upload(BRISTOL_CSV)
    bristol()
    assert (ROOT / "attendants_dataset.csv").read_bytes() == before


# --------------------------------------------------------------------------- #
# Default data and destination
# --------------------------------------------------------------------------- #


def test_default_dataset_and_venue_are_unchanged(real_dataset):
    assert real_dataset.event == DEFAULT_EVENT == Event()
    assert DEFAULT_EVENT.short_name == "Principality Stadium"
    assert (DEFAULT_EVENT.latitude, DEFAULT_EVENT.longitude) == (51.478283, -3.182652)
    assert len(real_dataset.schools) == 60 and real_dataset.report.ok
    assert points_of(real_dataset)[-1] == (51.478283, -3.182652)


def test_make_event_accepts_text_and_numbers():
    e = make_event("  Ashton Gate, Bristol ", "51.44", -2.62)
    assert e == Event("Ashton Gate, Bristol", 51.44, -2.62) and e.short_name == "Ashton Gate"


@pytest.mark.parametrize("name,lat,lon,needle", [
    ("", 51.0, -2.0, "venue name"),
    ("V", None, -2.0, "Latitude is missing"),
    ("V", "", -2.0, "Latitude is missing"),
    ("V", 51.0, "west", "Longitude is missing"),
    ("V", 91, -2.0, "between -90 and 90"),
    ("V", 51.0, 181, "between -180 and 180"),
    ("V", float("nan"), -2.0, "Latitude is missing"),
    ("V", 51.0, float("inf"), "Longitude is missing"),
    ("V", True, -2.0, "Latitude is missing"),
])
def test_make_event_rejects_invalid_input(name, lat, lon, needle):
    with pytest.raises(ValueError, match=needle):
        make_event(name, lat, lon)


def test_venue_csv():
    assert parse_venue_csv(b"name,latitude,longitude\nSynthetic Arena,51.44,-2.62\n") == \
        Event("Synthetic Arena", 51.44, -2.62)
    with pytest.raises(ValueError, match="exactly one"):
        parse_venue_csv(b"name,latitude,longitude\nA,51,-2\nB,52,-2\n")
    with pytest.raises(ValueError, match="missing column"):
        parse_venue_csv(b"name,lat,lon\nA,51,-2\n")
    with pytest.raises(ValueError, match="between -90 and 90"):
        parse_venue_csv(b"name,latitude,longitude\nA,99,-2\n")


def test_far_schools_flags_likely_coordinate_typos():
    ds = bristol()
    assert far_schools(ds.schools, BRISTOL) == []
    assert len(far_schools(ds.schools, Event("Swapped", -2.62, 51.44))) == 8


def test_changing_the_venue_changes_every_distance_to_it():
    a, b = bristol(BRISTOL), bristol(FAR_VENUE)
    pa, pb = points_of(a), points_of(b)
    assert pa[:-1] == pb[:-1] and pa[-1] == (BRISTOL.latitude, BRISTOL.longitude)
    assert pb[-1] == (FAR_VENUE.latitude, FAR_VENUE.longitude)
    ma, mb = straight_line_matrix(pa), straight_line_matrix(pb)
    assert np.allclose(ma[:-1, :-1], mb[:-1, :-1])  # school-to-school legs unchanged
    assert not np.allclose(ma[:-1, -1], mb[:-1, -1])  # every leg to the venue changed
    expected = haversine_miles(51.4545, -2.5879, FAR_VENUE.latitude, FAR_VENUE.longitude)
    assert mb[0, -1] == pytest.approx(expected)


# --------------------------------------------------------------------------- #
# Routing pipeline: OSRM request, cache and fallback
# --------------------------------------------------------------------------- #


def test_osrm_request_uses_the_new_schools_and_venue(tmp_path):
    fake = FakeOSRM()
    ds = bristol(FAR_VENUE)
    tm = road_time_matrix(points_of(ds), cache_dir=tmp_path, session=fake)
    assert tm.source == SOURCE_OSRM_LIVE and tm.hours.shape == (9, 9)
    coords = fake.urls[0].rsplit("/", 1)[1].split(";")
    assert coords[0] == "-2.587900,51.454500"  # OSRM order is lon,lat
    assert coords[-1] == f"{FAR_VENUE.longitude:.6f},{FAR_VENUE.latitude:.6f}"


def test_cache_is_separate_per_coordinate_set(tmp_path):
    fake = FakeOSRM()
    a, b = points_of(bristol(BRISTOL)), points_of(bristol(FAR_VENUE))
    assert _cache_key(a) != _cache_key(b)
    first = road_time_matrix(a, cache_dir=tmp_path, session=fake)
    other_venue = road_time_matrix(b, cache_dir=tmp_path, session=fake)
    again = road_time_matrix(a, cache_dir=tmp_path, session=fake)
    assert len(fake.urls) == 2  # the new venue made a request; repeating the first did not
    assert other_venue.source == SOURCE_OSRM_LIVE and again.source == SOURCE_OSRM_CACHE
    assert np.allclose(first.hours, again.hours) and not np.allclose(first.hours, other_venue.hours)
    assert len(list(tmp_path.glob("osrm_*.json"))) == 2


def test_cache_file_for_other_points_is_never_reused(tmp_path):
    """A cache file whose stored points differ (e.g. a hash collision) is ignored."""
    a, b = points_of(bristol(BRISTOL)), points_of(bristol(FAR_VENUE))
    (tmp_path / f"osrm_{_cache_key(a)}.json").write_text(
        json.dumps({"points": b, "durations": [[0.0] * len(b)] * len(b)}))
    tm = road_time_matrix(a, allow_online=False, cache_dir=tmp_path)
    assert tm.source == SOURCE_APPROX and any("differ" in n for n in tm.notes)


def test_osrm_failure_falls_back_to_labelled_estimate(tmp_path):
    pts = points_of(bristol())
    tm = road_time_matrix(pts, cache_dir=tmp_path, session=FakeOSRM(fail=True))
    assert tm.source == SOURCE_APPROX and "APPROXIMATE" in tm.source and not tm.is_road_data
    assert any("HTTP 503" in n for n in tm.notes) and any("approximate" in n for n in tm.notes)
    assert np.allclose(tm.hours, straight_line_matrix(pts) * 1.3 / 35.0)  # formula unchanged
    with pytest.raises(RoutingServiceError):
        road_time_matrix(pts, cache_dir=tmp_path, session=FakeOSRM(fail=True), allow_approximate=False)
    assert not list(tmp_path.glob("osrm_*.json"))  # a failure is never cached


def test_osrm_point_limit_includes_the_venue(tmp_path):
    fake = FakeOSRM()
    at_limit = grid_points(OSRM_MAX_POINTS)  # 99 schools + venue
    assert road_time_matrix(at_limit, cache_dir=tmp_path, session=fake).source == SOURCE_OSRM_LIVE
    over = grid_points(OSRM_MAX_POINTS + 1)  # 100 schools + venue
    tm = road_time_matrix(over, cache_dir=tmp_path, session=fake)
    assert len(fake.urls) == 1  # no request was sent for the oversized set
    assert tm.source == SOURCE_APPROX and any("exceed the public OSRM limit" in n for n in tm.notes)


def test_upload_warns_only_above_the_point_limit():
    def rows(n):
        return csv_bytes(*(f"S{k},{lat},{lon},5" for k, (lat, lon) in enumerate(grid_points(n))))

    assert not any("OSRM limit" in w for w in check_school_upload(rows(OSRM_MAX_POINTS - 1)).warnings)
    big = check_school_upload(rows(OSRM_MAX_POINTS))
    assert big.ok and any("exceed the OSRM limit" in w for w in big.warnings)


# --------------------------------------------------------------------------- #
# All three cost models on new schools and a new venue
# --------------------------------------------------------------------------- #


def check_plan(result, ds, constraints=None):
    sol = result.solution
    assert sol.has_plan, sol.notes
    rates = ds.rates[sol.model]
    matrix = None if sol.model == "fixed" else result.matrix
    assert validate_solution(sol, ds.schools, rates, constraints, matrix) == []
    stops = [s for r in sol.routes for s in r.stops]
    assert sorted(stops) == sorted(ds.schools["school_id"])  # everyone once, no split
    assert all(r.attendees <= r.bus_size for r in sol.routes)
    assert sol.total_attendees == int(ds.schools["attendees"].sum())


@pytest.mark.parametrize("model", ["fixed", "distance", "time"])
def test_every_model_plans_new_schools_to_a_new_venue(model):
    ds = bristol(FAR_VENUE)
    tm = road_time_matrix(points_of(ds), allow_online=False, cache_dir=Path("/nonexistent-cache"))
    result = optimise(ds, model, Constraints(), 3 if model == "fixed" else 2, tm if model == "time" else None)
    check_plan(result, ds)
    if model != "fixed":
        # Route costs are measured to the chosen venue, not to the default stadium.
        idx = {s: i for i, s in enumerate(ds.schools["school_id"])}
        r = result.solution.routes[0]
        last = ds.schools.iloc[idx[r.stops[-1]]]
        to_venue = haversine_miles(last.latitude, last.longitude, FAR_VENUE.latitude, FAR_VENUE.longitude)
        if model == "distance":
            assert result.matrix[idx[r.stops[-1]], -1] == pytest.approx(to_venue)
        assert r.metric == pytest.approx(route_metric([idx[s] for s in r.stops], result.matrix))


def test_venue_changes_the_distance_cost():
    near = optimise(bristol(BRISTOL), "distance", Constraints(), 2)
    far = optimise(bristol(FAR_VENUE), "distance", Constraints(), 2)
    check_plan(near, bristol(BRISTOL))
    check_plan(far, bristol(FAR_VENUE))
    assert far.solution.total_cost > near.solution.total_cost  # Bath is further from every school
    assert far.baseline.total_cost > near.baseline.total_cost


@pytest.mark.parametrize("model", ["fixed", "distance"])
def test_sharing_rules_and_capacity_hold_on_uploaded_data(model):
    ds = bristol()
    rules = Constraints(must_share=[("BS1", "BS8")], cannot_share=[("BS4", "BS6")], solo=["BS7"])
    result = optimise(ds, model, rules, 3)
    check_plan(result, ds, rules)
    bus = {s: r for r in result.solution.routes for s in r.stops}
    assert bus["BS1"] is bus["BS8"] and bus["BS4"] is not bus["BS6"] and bus["BS7"].stops == ["BS7"]


def test_uploaded_school_larger_than_every_bus_blocks_the_plan():
    ds = load_dataset(ROOT, schools_csv=csv_bytes("BIG,51.45,-2.58,51", "OK,51.46,-2.57,10"), event=BRISTOL)
    assert ds.report.capacity_errors["fixed"]
    result = optimise(ds, "fixed", Constraints(), 3)
    assert not result.solution.has_plan and any("BIG" in n for n in result.solution.notes)


def test_attendance_edits_work_on_uploaded_data():
    ds = bristol().with_attendance({"BS4": 40})
    assert ds.event == BRISTOL
    check_plan(optimise(ds, "fixed", Constraints(), 3), ds)


def test_exports_and_tables_name_the_chosen_venue():
    ds = bristol(FAR_VENUE)
    result = optimise(ds, "distance", Constraints(), 2)
    csv = plan_csv(result.solution, ds)
    assert "Destination,Destination latitude,Destination longitude" in csv.splitlines()[0]
    assert "Synthetic Hall, Bath" in csv and "Principality" not in csv
    details = route_details(result.solution.routes, result.solution, ds, result.matrix, None, None)
    assert details["Pickup order"].str.endswith("-> Synthetic Hall").all()


# --------------------------------------------------------------------------- #
# Dashboard (headless AppTest; the uploaded file is placed in session state,
# which is what the uploader callback does after validation)
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not list((ROOT / "data" / "cache").glob("osrm_*.json")), reason="no bundled OSRM cache")
def test_dashboard_finds_the_cached_road_times_for_the_default_inputs():
    """Regression: the app's matrix key must hit the bundled cache (no pre-rounding of coordinates)."""
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    assert not at.exception
    assert any("cached for these schools and this venue" in s.value for s in at.sidebar.success)


def test_dashboard_uses_uploaded_schools_and_new_venue():
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    assert not at.exception
    assert at.session_state["venue"] == DEFAULT_EVENT and at.session_state["school_source"] is None

    at.session_state["school_source"] = {"name": "bristol.csv", "content": BRISTOL_CSV, "digest": "x",
                                         "warnings": []}
    at.run()
    assert not at.exception
    assert any("bristol.csv" in m.value and "8 schools" in m.value for m in at.sidebar.markdown)

    # Invalid destination: rejected with a message, venue unchanged.
    at.sidebar.text_input(key="venue_lat").set_value("123")
    at.sidebar.button[[b.label for b in at.sidebar.button].index("Apply destination")].click().run()
    assert not at.exception and at.session_state["venue"] == DEFAULT_EVENT
    assert any("between -90 and 90" in e.value for e in at.sidebar.error)

    # Valid destination: used by the next run.
    at.sidebar.text_input(key="venue_name").set_value(FAR_VENUE.name)
    at.sidebar.text_input(key="venue_lat").set_value(str(FAR_VENUE.latitude))
    at.sidebar.text_input(key="venue_lon").set_value(str(FAR_VENUE.longitude))
    at.sidebar.button[[b.label for b in at.sidebar.button].index("Apply destination")].click().run()
    assert not at.exception and at.session_state["venue"] == FAR_VENUE

    at.sidebar.radio[0].set_value("fixed").run()
    at.button(key="run_optimiser").click().run()
    assert not at.exception
    cur = at.session_state["current"]
    assert cur["result"].solution.has_plan
    assert cur["dataset"].event == FAR_VENUE and len(cur["dataset"].schools) == 8


def test_changing_inputs_clears_plans_made_for_the_old_inputs():
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    at.sidebar.radio[0].set_value("fixed").run()
    at.sidebar.slider[0].set_value(5)
    at.run()
    at.button(key="run_optimiser").click().run()
    assert at.session_state["current"] is not None
    at.sidebar.text_input(key="venue_name").set_value(BRISTOL.name)
    at.sidebar.text_input(key="venue_lat").set_value(str(BRISTOL.latitude))
    at.sidebar.text_input(key="venue_lon").set_value(str(BRISTOL.longitude))
    at.sidebar.button[[b.label for b in at.sidebar.button].index("Apply destination")].click().run()
    assert not at.exception
    assert at.session_state["current"] is None and at.session_state["result_cache"] == {}
    assert at.session_state["venue"] == BRISTOL
    # Reset returns to Principality Stadium.
    at.sidebar.button[[b.label for b in at.sidebar.button].index("Reset to Principality Stadium")].click().run()
    assert not at.exception and at.session_state["venue"] == DEFAULT_EVENT
    assert math.isclose(float(at.session_state["venue_lat"]), DEFAULT_EVENT.latitude)
