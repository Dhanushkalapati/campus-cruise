"""Data for the interactive route map: filtering and per-bus details.

Pure functions, so the map and the table beside it are built from the same
rows and stay in step. Drawing (folium) stays in app.py.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .data import Dataset
from .model import BusRoute, Solution
from .validation import route_metric


def filter_routes(sol: Solution, buses: list[str] | None = None, school: str | None = None) -> list[BusRoute]:
    """Routes to show. ``buses`` empty/None = all; ``school`` keeps only the bus carrying it."""
    routes = list(sol.routes)
    if buses:
        keep = set(buses)
        routes = [r for r in routes if r.bus_id in keep]
    if school:
        routes = [r for r in routes if school in r.stops]
    return routes


def bus_of_school(sol: Solution, school: str) -> str | None:
    return next((r.bus_id for r in sol.routes if school in r.stops), None)


def route_details(routes: list[BusRoute], sol: Solution, ds: Dataset, miles: np.ndarray,
                  hours: np.ndarray | None, hours_source: str | None) -> pd.DataFrame:
    """One row per route: load, capacity, pickup order, distance, time and cost.

    Straight-line miles are always computed. Hours appear only when a travel-time
    matrix is supplied, with its source; otherwise the column says not available.
    """
    idx = {sid: i for i, sid in enumerate(ds.schools["school_id"])}
    rows = []
    for r in routes:
        stops = [idx[s] for s in r.stops]
        h = None if hours is None else route_metric(stops, hours)
        rows.append({
            "Bus": r.bus_id,
            "Bus size": r.bus_size,
            "Attendees": r.attendees,
            "Seats left": r.empty_seats,
            "Utilisation %": round(r.utilisation * 100, 1),
            "Schools": len(r.stops),
            "Pickup order": " -> ".join(r.stops) + f" -> {ds.event.short_name}",
            "Straight-line miles": round(route_metric(stops, miles), 2),
            "Driving minutes": None if h is None else round(h * 60),
            "Time source": hours_source if h is not None else "not available",
            f"Cost ({sol.model} model)": round(r.cost, 2),
        })
    return pd.DataFrame(rows)


def school_rows(routes: list[BusRoute], ds: Dataset) -> pd.DataFrame:
    """Schools on the shown routes, in pickup order."""
    att = dict(zip(ds.schools["school_id"], ds.schools["attendees"]))
    rows = [{"School": s, "Attendees": int(att[s]), "Bus": r.bus_id, "Pickup #": k}
            for r in routes for k, s in enumerate(r.stops, start=1)]
    return pd.DataFrame(rows, columns=["School", "Attendees", "Bus", "Pickup #"])


def _mercator(lat: float, lon: float) -> tuple[float, float]:
    """Web Mercator position in [0, 1] x [0, 1] (the projection map tiles use)."""
    lat = max(min(lat, 85.0), -85.0)
    x = (lon + 180.0) / 360.0
    y = (1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2
    return x, y


def map_view(points: list[tuple[float, float]], width_px: int = 640, height_px: int = 480,
             padding_px: int = 40, min_zoom: int = 5, max_zoom: int = 14) -> tuple[tuple[float, float], int]:
    """Centre (lat, lon) and zoom that fit ``points`` in a map of the given size.

    Computed here rather than with Leaflet's fitBounds, because a map drawn in a
    hidden tab has zero size when fitBounds runs and zooms to its maximum.
    """
    if not points:
        raise ValueError("map_view needs at least one point")
    xy = [_mercator(lat, lon) for lat, lon in points]
    xs, ys = [p[0] for p in xy], [p[1] for p in xy]
    span_x, span_y = max(xs) - min(xs), max(ys) - min(ys)
    usable_w, usable_h = max(width_px - 2 * padding_px, 1), max(height_px - 2 * padding_px, 1)
    fits = [math.log2(usable / (256 * span)) for usable, span in ((usable_w, span_x), (usable_h, span_y))
            if span > 0]
    zoom = max_zoom if not fits else int(math.floor(min(fits)))
    zoom = max(min_zoom, min(max_zoom, zoom))
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    lon = cx * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * cy))))
    return (lat, lon), zoom
