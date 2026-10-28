"""End-to-end integrity on the real dataset, plus travel-time source handling."""

import json

import numpy as np
import pytest
import requests

from src.distances import (
    SOURCE_APPROX,
    SOURCE_OSRM_CACHE,
    SOURCE_OSRM_LIVE,
    RoutingServiceError,
    road_time_matrix,
)
from src.model import Constraints
from src.planner import optimise, points_of


def assert_integrity(result, dataset):
    sol = result.solution
    assert sol.has_plan, (sol.status, sol.notes, sol.violations)
    att = dict(zip(dataset.schools.school_id, dataset.schools.attendees))
    seen = [s for r in sol.routes for s in r.stops]
    assert sorted(seen) == sorted(s for s, a in att.items() if a > 0)  # every school once, none split
    for r in sol.routes:
        assert r.attendees == sum(att[s] for s in r.stops) <= r.bus_size
    assert sol.total_cost <= result.baseline.total_cost + 1e-6


def test_fixed_real(real_dataset):
    assert_integrity(optimise(real_dataset, "fixed", time_limit=20), real_dataset)


def test_distance_real(real_dataset):
    assert_integrity(optimise(real_dataset, "distance", time_limit=8), real_dataset)


def test_time_real_offline_estimate(real_dataset, tmp_path):
    tm = road_time_matrix(points_of(real_dataset), allow_online=False, cache_dir=tmp_path)
    assert tm.source == SOURCE_APPROX and not tm.is_road_data
    result = optimise(real_dataset, "time", time_limit=8, time_matrix=tm)
    assert_integrity(result, real_dataset)
    assert any("approximation" in n for n in result.solution.notes)


def test_real_constraints_and_attendance_change(real_dataset):
    edited = real_dataset.with_attendance({"CF15": 25})
    c = Constraints(must_share=[("SA1", "SA2")], cannot_share=[("CF62", "CF63")], solo=["NP24"])
    result = optimise(edited, "distance", constraints=c, time_limit=8)
    assert_integrity(result, edited)
    bus = result.solution.bus_of()
    assert bus["SA1"] == bus["SA2"] and bus["CF62"] != bus["CF63"]


# ---- travel-time sources (no network used: sessions are faked) ----------- #


class FakeResponse:
    def __init__(self, status=200, body=None):
        self.status_code = status
        self._body = body

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body


class FakeSession:
    def __init__(self, response=None, exc=None):
        self.response, self.exc, self.calls = response, exc, 0

    def get(self, *args, **kwargs):
        self.calls += 1
        if self.exc:
            raise self.exc
        return self.response


POINTS = [(51.6, -3.3), (51.7, -3.4), (51.478283, -3.182652)]


def test_osrm_success_is_cached(tmp_path):
    durations = [[0, 600, 1200], [650, 0, 1500], [1250, 1550, 0]]
    session = FakeSession(FakeResponse(body={"code": "Ok", "durations": durations}))
    first = road_time_matrix(POINTS, cache_dir=tmp_path, session=session)
    assert first.source == SOURCE_OSRM_LIVE and first.is_road_data
    assert first.hours[0, 1] == pytest.approx(600 / 3600)
    second = road_time_matrix(POINTS, cache_dir=tmp_path, session=session)
    assert second.source == SOURCE_OSRM_CACHE and session.calls == 1
    assert np.allclose(first.hours, second.hours)


@pytest.mark.parametrize("session", [
    FakeSession(exc=requests.ConnectionError("offline")),
    FakeSession(FakeResponse(status=503)),
    FakeSession(FakeResponse(body={"code": "InvalidQuery"})),
    FakeSession(FakeResponse(body={"code": "Ok", "durations": [[0, 1], [1, 0]]})),
    FakeSession(FakeResponse(body={"code": "Ok", "durations": [[0, None, 1], [1, 0, 1], [1, 1, 0]]})),
    FakeSession(FakeResponse(body=None)),
])
def test_osrm_failures_fall_back_to_labelled_estimate(tmp_path, session):
    tm = road_time_matrix(POINTS, cache_dir=tmp_path, session=session)
    assert tm.source == SOURCE_APPROX
    assert tm.notes


def test_osrm_failure_without_fallback_raises(tmp_path):
    with pytest.raises(RoutingServiceError):
        road_time_matrix(POINTS, cache_dir=tmp_path, allow_approximate=False,
                         session=FakeSession(exc=requests.Timeout("slow")))


def test_corrupt_cache_is_ignored(tmp_path):
    from src.distances import _cache_key

    (tmp_path / f"osrm_{_cache_key(POINTS)}.json").write_text(json.dumps({"durations": "bad"}))
    tm = road_time_matrix(POINTS, cache_dir=tmp_path, allow_online=False)
    assert tm.source == SOURCE_APPROX
    assert any("unreadable cache" in n for n in tm.notes)
