"""Find a venue's coordinates from a typed name or address.

Uses the public OpenStreetMap Nominatim search service (no API key). Its usage
policy asks for at most 1 request per second and an identifying User-Agent;
answers are cached on disk so a repeated search sends no request.

Only the text the user typed is sent. No school data, head counts or plans.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import requests

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "CampusCruise/1.0 (school transport planner, student competition project)"
CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache" / "geocode"
MIN_QUERY, MAX_QUERY = 3, 200
MAX_RESULTS = 5
MIN_REQUEST_INTERVAL_S = 1.0
_last_request = [0.0]


class GeocodeError(RuntimeError):
    """The search could not return a usable place. The message is shown to the user."""


@dataclass(frozen=True)
class Place:
    """One search match. ``name`` is the full OpenStreetMap address line."""

    name: str
    latitude: float
    longitude: float


def _clean_query(query: object) -> str:
    text = " ".join(str(query or "").split())
    if len(text) < MIN_QUERY:
        raise GeocodeError(f"Type at least {MIN_QUERY} characters of a venue name or address.")
    if len(text) > MAX_QUERY:
        raise GeocodeError(f"The search text must be at most {MAX_QUERY} characters.")
    return text


def _parse(body: object) -> list[Place]:
    if not isinstance(body, list):
        raise GeocodeError("The address service returned an unexpected answer.")
    places = []
    for item in body:
        try:
            lat, lon = float(item["lat"]), float(item["lon"])
            name = str(item.get("display_name") or "").strip()
        except (TypeError, ValueError, KeyError):
            continue
        if name and -90 <= lat <= 90 and -180 <= lon <= 180:
            places.append(Place(name, lat, lon))
    return places


def _throttle() -> None:
    wait = MIN_REQUEST_INTERVAL_S - (time.monotonic() - _last_request[0])
    if wait > 0:
        time.sleep(wait)
    _last_request[0] = time.monotonic()


def geocode(
    query: object,
    limit: int = MAX_RESULTS,
    session: requests.Session | None = None,
    timeout: float = 10.0,
    cache_dir: Path | None = CACHE_DIR,
) -> list[Place]:
    """Matches for ``query``, best first. Raises GeocodeError with a user-facing message."""
    text = _clean_query(query)
    cache_file = None
    if cache_dir is not None:
        key = hashlib.sha256(f"{text.lower()}|{limit}".encode()).hexdigest()[:16]
        cache_file = Path(cache_dir) / f"place_{key}.json"
        if cache_file.exists():
            try:
                places = _parse(json.loads(cache_file.read_text(encoding="utf-8")))
                if places:
                    return places
            except (OSError, ValueError, GeocodeError):
                pass  # unreadable cache: search again

    getter = session or requests
    _throttle()
    try:
        resp = getter.get(NOMINATIM_URL, params={"q": text, "format": "jsonv2", "limit": limit},
                          headers={"User-Agent": USER_AGENT}, timeout=timeout)
    except requests.RequestException as exc:
        raise GeocodeError(f"The address service could not be reached ({type(exc).__name__}). "
                           "Check the internet connection, or enter the coordinates by hand.") from exc
    if resp.status_code != 200:
        raise GeocodeError(f"The address service returned HTTP {resp.status_code}. Try again in a "
                           "moment, or enter the coordinates by hand.")
    try:
        body = resp.json()
    except ValueError as exc:
        raise GeocodeError("The address service returned an unreadable answer.") from exc
    places = _parse(body)
    if not places:
        raise GeocodeError(f"No place found for \"{text}\". Add the town or postcode, "
                           "or enter the coordinates by hand.")
    if cache_file is not None:
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(body), encoding="utf-8")
        except OSError:
            pass  # caching is an optimisation only
    return places
