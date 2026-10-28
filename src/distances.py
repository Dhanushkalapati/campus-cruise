"""Distance and travel-time matrices.

All matrices put the schools first (in dataset order) and the event last, so
``matrix[i, j]`` is the leg from point i to point j.

Travel time sources, in order of preference:
    1. OSRM public table service (real road travel times), cached on disk.
    2. Approximate offline estimate: straight-line miles x road factor / speed.
       This is NOT road routing and is always labelled as an estimate.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests

EARTH_RADIUS_MILES = 3958.7613
OSRM_URL = "https://router.project-osrm.org/table/v1/driving/"
OSRM_MAX_POINTS = 100  # public demo server limit per table request
CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"

# Offline estimate parameters (documented assumptions, not measured values).
ROAD_FACTOR = 1.3
AVERAGE_SPEED_MPH = 35.0

SOURCE_OSRM_LIVE = "OSRM road travel times (live request)"
SOURCE_OSRM_CACHE = "OSRM road travel times (cached)"
SOURCE_APPROX = (
    f"APPROXIMATE: straight-line miles x {ROAD_FACTOR} / {AVERAGE_SPEED_MPH:g} mph (not road routing)"
)


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle ("as the crow flies") distance in miles."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_MILES * math.asin(min(1.0, math.sqrt(a)))


def straight_line_matrix(points: list[tuple[float, float]]) -> np.ndarray:
    """Symmetric matrix of haversine miles between (lat, lon) points."""
    n = len(points)
    m = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            m[i, j] = m[j, i] = haversine_miles(*points[i], *points[j])
    return m


def approximate_time_matrix(points: list[tuple[float, float]]) -> np.ndarray:
    """Offline travel-time estimate in hours. Labelled approximate everywhere."""
    return straight_line_matrix(points) * ROAD_FACTOR / AVERAGE_SPEED_MPH


@dataclass
class TimeMatrix:
    hours: np.ndarray
    source: str
    notes: list[str]

    @property
    def is_road_data(self) -> bool:
        return self.source in (SOURCE_OSRM_LIVE, SOURCE_OSRM_CACHE)


class RoutingServiceError(RuntimeError):
    """OSRM could not provide a valid matrix."""


def _cache_key(points: list[tuple[float, float]]) -> str:
    """Hash of every point in order (schools, then the event), to 6 decimal places.

    Any change to a school, the venue or the point order gives a new key, so a
    matrix is never reused for other coordinates. The routing settings are
    fixed (public OSRM ``driving`` profile, durations only) and separate caches
    are used for the table and route services, so they need no key part.
    """
    text = ";".join(f"{lat:.6f},{lon:.6f}" for lat, lon in points)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _same_points(stored: object, points: list[tuple[float, float]]) -> bool:
    """True if a cache file's stored points are ``points`` (guards against a hash collision)."""
    if stored is None:
        return True  # older cache files hold no points; the full-length key still matches
    try:
        return len(stored) == len(points) and all(
            f"{float(a):.6f},{float(c):.6f}" == f"{b:.6f},{d:.6f}" for (a, c), (b, d) in zip(stored, points))
    except (TypeError, ValueError):
        return False


def _check_matrix(matrix: object, n: int) -> np.ndarray:
    if not isinstance(matrix, list) or len(matrix) != n or any(
        not isinstance(row, list) or len(row) != n for row in matrix
    ):
        raise RoutingServiceError(f"OSRM matrix has wrong shape (expected {n}x{n}).")
    if any(v is None for row in matrix for v in row):
        raise RoutingServiceError("OSRM could not route between some points (null durations).")
    arr = np.array(matrix, dtype=float)
    if not np.all(np.isfinite(arr)) or np.any(arr < 0):
        raise RoutingServiceError("OSRM matrix contains invalid durations.")
    return arr


def fetch_osrm_durations(
    points: list[tuple[float, float]], timeout: float = 20.0, session: requests.Session | None = None
) -> np.ndarray:
    """Request a full duration matrix (seconds) from OSRM in one table call."""
    if len(points) > OSRM_MAX_POINTS:
        raise RoutingServiceError(
            f"{len(points)} points exceed the public OSRM limit of {OSRM_MAX_POINTS} per request."
        )
    coords = ";".join(f"{lon:.6f},{lat:.6f}" for lat, lon in points)
    getter = session or requests
    try:
        resp = getter.get(OSRM_URL + coords, params={"annotations": "duration"}, timeout=timeout)
    except requests.RequestException as exc:
        raise RoutingServiceError(f"OSRM request failed: {exc}") from exc
    if resp.status_code != 200:
        raise RoutingServiceError(f"OSRM returned HTTP {resp.status_code}.")
    try:
        body = resp.json()
    except ValueError as exc:
        raise RoutingServiceError("OSRM returned a non-JSON response.") from exc
    if body.get("code") != "Ok":
        raise RoutingServiceError(f"OSRM error: {body.get('code')} {body.get('message', '')}".strip())
    return _check_matrix(body.get("durations"), len(points))


def road_time_matrix(
    points: list[tuple[float, float]],
    allow_online: bool = True,
    allow_approximate: bool = True,
    cache_dir: Path = CACHE_DIR,
    timeout: float = 20.0,
    session: requests.Session | None = None,
) -> TimeMatrix:
    """Travel times in hours: cache, then OSRM, then (if allowed) the estimate.

    Raises RoutingServiceError when no allowed source works.
    """
    notes: list[str] = []
    cache_file = Path(cache_dir) / f"osrm_{_cache_key(points)}.json"
    if cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text())
            if not _same_points(cached.get("points"), points):
                raise RoutingServiceError("cached points differ from the requested points")
            seconds = _check_matrix(cached["durations"], len(points))
            return TimeMatrix(seconds / 3600.0, SOURCE_OSRM_CACHE, [f"Loaded from {cache_file.name}."])
        except (OSError, ValueError, KeyError, RoutingServiceError) as exc:
            notes.append(f"Ignored unreadable cache file {cache_file.name}: {exc}")

    if allow_online:
        try:
            seconds = fetch_osrm_durations(points, timeout=timeout, session=session)
            try:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                cache_file.write_text(json.dumps({"points": points, "durations": seconds.tolist()}))
            except OSError as exc:
                notes.append(f"Could not write cache: {exc}")
            return TimeMatrix(seconds / 3600.0, SOURCE_OSRM_LIVE, notes)
        except RoutingServiceError as exc:
            notes.append(str(exc))

    if allow_approximate:
        notes.append("Using the offline estimate. Travel times are approximate, not road routing.")
        return TimeMatrix(approximate_time_matrix(points), SOURCE_APPROX, notes)
    raise RoutingServiceError("; ".join(notes) or "Online routing disabled and estimate not allowed.")


# --------------------------------------------------------------------------- #
# Road paths for the map (OSRM route service)
# --------------------------------------------------------------------------- #

OSRM_ROUTE_URL = "https://router.project-osrm.org/route/v1/driving/"
ROUTE_CACHE_DIR = CACHE_DIR / "routes"
MIN_REQUEST_INTERVAL_S = 1.0  # public OSRM demo server asks for at most 1 request per second
_last_request = [0.0]


@dataclass
class RoadPath:
    """Road geometry for one bus: first pickup -> ... -> event.

    ``coords`` are (lat, lon). ``miles`` and ``hours`` are OSRM's road values
    for the drawn path; they are for display and do not change plan costs.
    """

    coords: list[tuple[float, float]]
    miles: float
    hours: float
    cached: bool


def _throttle() -> None:
    import time

    wait = MIN_REQUEST_INTERVAL_S - (time.monotonic() - _last_request[0])
    if wait > 0:
        time.sleep(wait)
    _last_request[0] = time.monotonic()


def _parse_route(body: object) -> tuple[list[tuple[float, float]], float, float]:
    if not isinstance(body, dict) or body.get("code") != "Ok" or not body.get("routes"):
        code = body.get("code") if isinstance(body, dict) else "invalid"
        raise RoutingServiceError(f"OSRM route error: {code}")
    route = body["routes"][0]
    geometry = route.get("geometry") or {}
    points = geometry.get("coordinates")
    if not isinstance(points, list) or len(points) < 2:
        raise RoutingServiceError("OSRM route has no usable geometry.")
    try:
        coords = [(float(lat), float(lon)) for lon, lat in points]
        miles = float(route["distance"]) / 1609.344
        hours = float(route["duration"]) / 3600.0
    except (TypeError, ValueError, KeyError) as exc:
        raise RoutingServiceError(f"OSRM route is malformed: {exc}") from exc
    if not all(-90 <= a <= 90 and -180 <= b <= 180 for a, b in coords) or miles < 0 or hours < 0:
        raise RoutingServiceError("OSRM route contains invalid values.")
    return coords, miles, hours


def road_path(
    stops: list[tuple[float, float]],
    cache_dir: Path = ROUTE_CACHE_DIR,
    timeout: float = 20.0,
    session: requests.Session | None = None,
    allow_online: bool = True,
) -> RoadPath:
    """Road path through ``stops`` (lat, lon) in order, from cache or OSRM.

    Raises RoutingServiceError if no valid path is available.
    """
    if len(stops) < 2:
        raise RoutingServiceError("A road path needs at least two points.")
    cache_file = Path(cache_dir) / f"route_{_cache_key(stops)}.json"
    if cache_file.exists():
        try:
            coords, miles, hours = _parse_route(json.loads(cache_file.read_text()))
            return RoadPath(coords, miles, hours, cached=True)
        except (OSError, ValueError, RoutingServiceError):
            pass  # unreadable cache: fetch again
    if not allow_online:
        raise RoutingServiceError("No cached road path and online routing is disabled.")

    coords_text = ";".join(f"{lon:.6f},{lat:.6f}" for lat, lon in stops)
    getter = session or requests
    _throttle()
    try:
        resp = getter.get(OSRM_ROUTE_URL + coords_text,
                          params={"overview": "full", "geometries": "geojson"}, timeout=timeout)
    except requests.RequestException as exc:
        raise RoutingServiceError(f"OSRM route request failed: {exc}") from exc
    if resp.status_code != 200:
        raise RoutingServiceError(f"OSRM route returned HTTP {resp.status_code}.")
    try:
        body = resp.json()
    except ValueError as exc:
        raise RoutingServiceError("OSRM route returned a non-JSON response.") from exc
    coords, miles, hours = _parse_route(body)
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(body))
    except OSError:
        pass  # caching is an optimisation only
    return RoadPath(coords, miles, hours, cached=False)
