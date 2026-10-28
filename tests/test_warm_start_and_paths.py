"""Warm start (re-optimising from a previous plan) and map road paths.

Instances built with make_schools are SYNTHETIC. Road-path tests use fake
sessions, so no network is needed.
"""

import json

import pytest
import requests

from conftest import EVENT, FIXED, PER_MILE, make_schools
from src.distances import RoutingServiceError, road_path, straight_line_matrix
from src.model import Constraints
from src.planner import optimise
from src.solvers import check_constraints, event_points, repair_plan, solve_fixed, solve_routing
from src.validation import route_metric

SYN = make_schools([
    ("A", 51.70, -3.60, 10), ("B", 51.72, -3.58, 12), ("C", 51.65, -3.00, 20),
    ("D", 51.80, -3.30, 30), ("E", 51.62, -3.02, 9), ("F", 51.60, -3.10, 0),
])
M = straight_line_matrix(event_points(SYN, *EVENT))
IDS = SYN["school_id"].tolist()


def ids_of(groups):
    return [[IDS[k] for k in g] for g in groups]


def assert_valid_start(groups, schools, constraints):
    att = dict(zip(schools.school_id, schools.attendees))
    flat = [s for g in ids_of(groups) for s in g]
    assert sorted(flat) == sorted(s for s, a in att.items() if a > 0)
    for g in ids_of(groups):
        assert sum(att[s] for s in g) <= max(PER_MILE)
    bus = {s: k for k, g in enumerate(ids_of(groups)) for s in g}
    for a, b in constraints.must_share:
        assert bus[a] == bus[b]
    for a, b in constraints.cannot_share:
        assert bus[a] != bus[b]
    for s in constraints.solo:
        assert len(ids_of(groups)[bus[s]]) == 1


def test_repair_keeps_unchanged_plan():
    previous = [["A", "B"], ["C", "E"], ["D"]]
    groups, note = repair_plan(previous, SYN, PER_MILE, Constraints(), M)
    assert ids_of(groups) == previous
    assert "5 of 5" in note


def test_repair_drops_zero_and_adds_new_schools():
    previous = [["A", "F"], ["C"]]  # F has 0 attendees; B, D, E are new
    groups, note = repair_plan(previous, SYN, PER_MILE, Constraints(), M)
    assert_valid_start(groups, SYN, Constraints())
    assert "new to the plan" in note


def test_repair_fixes_capacity_overflow():
    bigger = SYN.copy()
    bigger.loc[bigger.school_id == "D", "attendees"] = 45
    groups, note = repair_plan([["A", "D"]], bigger, PER_MILE, Constraints(), M)
    assert_valid_start(groups, bigger, Constraints())
    assert "re-placed" in note


def test_repair_enforces_new_constraints():
    c = Constraints(must_share=[("A", "C")], cannot_share=[("D", "E")], solo=["B"])
    assert check_constraints(SYN, PER_MILE, c) == []
    groups, _ = repair_plan([["A", "B"], ["D", "E"], ["C"]], SYN, PER_MILE, c, M)
    assert_valid_start(groups, SYN, c)


def test_routing_warm_start_never_worse_than_start():
    first = solve_routing(SYN, PER_MILE, M, "distance", time_limit=1)
    again = solve_routing(SYN, PER_MILE, M, "distance", time_limit=1,
                          warm_start=[r.stops for r in first.routes])
    assert again.has_plan
    assert again.total_cost <= first.total_cost + 1e-6
    assert "warm start" in again.method


def test_real_warm_start_same_inputs_keeps_cost(real_dataset):
    first = optimise(real_dataset, "distance", time_limit=5).solution
    again = optimise(real_dataset, "distance", time_limit=2, warm_start=first).solution
    assert again.has_plan
    assert again.total_cost <= first.total_cost + 1e-6


def test_real_warm_start_after_change_beats_repaired_start(real_dataset):
    first = optimise(real_dataset, "distance", time_limit=5).solution
    changed = real_dataset.with_attendance({"CF15": 25, "SA18": 20})
    c = Constraints(solo=["NP24"])
    after = optimise(changed, "distance", constraints=c, time_limit=5, warm_start=first).solution
    assert after.has_plan, after.violations
    groups, _ = repair_plan([r.stops for r in first.routes], changed.schools,
                            changed.rates["distance"], c, after_matrix := straight_line_matrix(
                                event_points(changed.schools, *EVENT)))
    rates = changed.rates["distance"]
    att = changed.schools.attendees.tolist()
    start_cost = sum(
        min(rates[t] for t in rates if t >= sum(att[k] for k in g)) * route_metric(g, after_matrix)
        for g in groups
    )
    assert after.total_cost <= start_cost + 1e-6


def test_fixed_warm_start_keeps_optimum():
    schools = make_schools([(f"S{i}", 51.5, -3.3 - i * 0.01, a) for i, a in enumerate([30, 20, 15, 10, 25, 5])])
    first = solve_fixed(schools, FIXED, *EVENT, time_limit=5)
    again = solve_fixed(schools, FIXED, *EVENT, time_limit=5, warm_start=[r.stops for r in first.routes])
    assert again.status == "OPTIMAL" and again.total_cost == first.total_cost
    assert any("Warm start" in n for n in again.notes)


def test_warm_start_ignores_invalid_previous_plan(real_dataset):
    from src.model import INVALID, Solution

    bad = Solution("distance", INVALID)
    result = optimise(real_dataset, "distance", time_limit=2, warm_start=bad).solution
    assert result.has_plan and "warm start" not in result.method


# ---- road paths ---------------------------------------------------------- #


class FakeResponse:
    def __init__(self, status=200, body=None):
        self.status_code, self._body = status, body

    def json(self):
        if self._body is None:
            raise ValueError
        return self._body


class FakeSession:
    def __init__(self, response=None, exc=None):
        self.response, self.exc, self.calls = response, exc, 0

    def get(self, *a, **k):
        self.calls += 1
        if self.exc:
            raise self.exc
        return self.response


STOPS = [(51.62, -3.94), (51.61, -3.99), EVENT]
GOOD = {"code": "Ok", "routes": [{"distance": 16093.44, "duration": 1800,
                                  "geometry": {"coordinates": [[-3.94, 51.62], [-3.5, 51.55], [-3.18, 51.48]]}}]}


@pytest.fixture(autouse=True)
def no_throttle(monkeypatch):
    monkeypatch.setattr("src.distances.MIN_REQUEST_INTERVAL_S", 0.0)


def test_road_path_parsed_and_cached(tmp_path):
    session = FakeSession(FakeResponse(body=GOOD))
    first = road_path(STOPS, cache_dir=tmp_path, session=session)
    assert first.coords[0] == (51.62, -3.94)  # converted from OSRM lon,lat
    assert first.miles == pytest.approx(10.0) and first.hours == pytest.approx(0.5)
    assert not first.cached
    second = road_path(STOPS, cache_dir=tmp_path, session=session)
    assert second.cached and session.calls == 1


@pytest.mark.parametrize("session", [
    FakeSession(exc=requests.ConnectionError("offline")),
    FakeSession(FakeResponse(status=429)),
    FakeSession(FakeResponse(body={"code": "NoRoute"})),
    FakeSession(FakeResponse(body={"code": "Ok", "routes": [{"distance": 1, "duration": 1, "geometry": {}}]})),
    FakeSession(FakeResponse(body=None)),
])
def test_road_path_failures_raise(tmp_path, session):
    with pytest.raises(RoutingServiceError):
        road_path(STOPS, cache_dir=tmp_path, session=session)


def test_road_path_offline_without_cache_raises(tmp_path):
    with pytest.raises(RoutingServiceError):
        road_path(STOPS, cache_dir=tmp_path, allow_online=False)


def test_corrupt_route_cache_refetched(tmp_path):
    from src.distances import _cache_key

    (tmp_path / f"route_{_cache_key(STOPS)}.json").write_text(json.dumps({"code": "Ok", "routes": []}))
    session = FakeSession(FakeResponse(body=GOOD))
    assert not road_path(STOPS, cache_dir=tmp_path, session=session).cached
    assert session.calls == 1
