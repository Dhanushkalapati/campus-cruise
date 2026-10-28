"""Destination search (src/geocode.py) and its dashboard flow.

Nominatim is never contacted: a fake session or a patched ``geocode`` stands
in for the network. Places and coordinates here are illustrative test values.
"""

from pathlib import Path

import pytest
import requests
from streamlit.testing.v1 import AppTest

import src.geocode as geo
from src.data import DEFAULT_EVENT
from src.geocode import GeocodeError, Place, geocode

APP = str(Path(__file__).resolve().parent.parent / "app.py")

ASHTON = {"lat": "51.4400", "lon": "-2.6203", "display_name": "Ashton Gate Stadium, Ashton Road, Bristol, BS3 2EJ, UK"}
OTHER = {"lat": "53.4000", "lon": "-2.9000", "display_name": "Ashton Gate, Liverpool, UK"}


@pytest.fixture(autouse=True)
def no_wait(monkeypatch):
    monkeypatch.setattr(geo, "MIN_REQUEST_INTERVAL_S", 0.0)


class FakeResponse:
    def __init__(self, body, status=200, bad_json=False):
        self.body, self.status_code, self.bad_json = body, status, bad_json

    def json(self):
        if self.bad_json:
            raise ValueError("not json")
        return self.body


class FakeSession:
    def __init__(self, response=None, exc=None):
        self.response, self.exc, self.calls = response, exc, []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append((url, params, headers))
        if self.exc:
            raise self.exc
        return self.response


def test_best_match_first_and_request_is_identified(tmp_path):
    s = FakeSession(FakeResponse([ASHTON, OTHER]))
    places = geocode("  Ashton   Gate ", session=s, cache_dir=tmp_path)
    assert places[0] == Place(ASHTON["display_name"], 51.44, -2.6203) and len(places) == 2
    url, params, headers = s.calls[0]
    assert url == geo.NOMINATIM_URL and params["q"] == "Ashton Gate" and params["format"] == "jsonv2"
    assert "CampusCruise" in headers["User-Agent"]


def test_repeat_search_uses_the_cache(tmp_path):
    s = FakeSession(FakeResponse([ASHTON]))
    geocode("Ashton Gate", session=s, cache_dir=tmp_path)
    assert geocode("ashton gate", session=s, cache_dir=tmp_path)[0].latitude == 51.44
    assert len(s.calls) == 1


@pytest.mark.parametrize("query", ["", "  ", "ab", None, "x" * 201])
def test_query_length_is_checked_before_any_request(query, tmp_path):
    s = FakeSession(FakeResponse([ASHTON]))
    with pytest.raises(GeocodeError):
        geocode(query, session=s, cache_dir=tmp_path)
    assert s.calls == []


@pytest.mark.parametrize("session,needle", [
    (FakeSession(FakeResponse([])), "No place found"),
    (FakeSession(FakeResponse([{"lat": "95", "lon": "0", "display_name": "Bad"}])), "No place found"),
    (FakeSession(FakeResponse({}, status=429)), "HTTP 429"),
    (FakeSession(FakeResponse(None, bad_json=True)), "unreadable"),
    (FakeSession(FakeResponse({"error": "x"})), "unexpected"),
    (FakeSession(exc=requests.ConnectionError("down")), "could not be reached"),
])
def test_failures_give_a_clear_message_and_nothing_is_cached(session, needle, tmp_path):
    with pytest.raises(GeocodeError, match=needle):
        geocode("Ashton Gate", session=session, cache_dir=tmp_path)
    assert not list(tmp_path.glob("*.json"))


# --------------------------------------------------------------------------- #
# Dashboard
# --------------------------------------------------------------------------- #


def fake_geocode(query, *a, **k):
    if "nowhere" in query.lower():
        raise GeocodeError(f'No place found for "{query}".')
    return [Place(ASHTON["display_name"], 51.44, -2.6203), Place(OTHER["display_name"], 53.4, -2.9)]


def sidebar_button(at, label):
    return next(b for b in at.sidebar.button if b.label == label)


def test_search_moves_the_venue_and_fills_the_coordinates(monkeypatch):
    monkeypatch.setattr(geo, "geocode", fake_geocode)
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    at.sidebar.text_input(key="venue_query").set_value("Ashton Gate Stadium")
    sidebar_button(at, "Find and use").click().run()
    assert not at.exception
    v = at.session_state["venue"]
    assert (v.latitude, v.longitude) == (51.44, -2.6203) and v.short_name == "Ashton Gate Stadium"
    assert at.session_state["venue_lat"] == "51.440000" and at.session_state["venue_lon"] == "-2.620300"

    # Picking another match moves the venue again.
    at.sidebar.selectbox(key="venue_match").set_value(1).run()
    assert not at.exception and at.session_state["venue"].latitude == 53.4

    # A failed search keeps the venue and says why.
    at.sidebar.text_input(key="venue_query").set_value("Nowhere Land")
    sidebar_button(at, "Find and use").click().run()
    assert at.session_state["venue"].latitude == 53.4
    assert any("No place found" in e.value for e in at.sidebar.error)


def test_changing_only_the_name_looks_it_up(monkeypatch):
    """Regression: a new name with the old coordinates used to keep the old location."""
    monkeypatch.setattr(geo, "geocode", fake_geocode)
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    at.sidebar.text_input(key="venue_name").set_value("Ashton Gate Stadium, Bristol")
    sidebar_button(at, "Apply destination").click().run()
    assert not at.exception
    assert at.session_state["venue"] != DEFAULT_EVENT and at.session_state["venue"].latitude == 51.44
