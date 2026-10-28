"""Shared fixtures. All data built here is SYNTHETIC unless it loads the real files."""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import load_dataset  # noqa: E402

EVENT = (51.478283, -3.182652)
FIXED = {15: 40.0, 35: 60.0, 50: 75.0}
PER_MILE = {15: 2.25, 35: 3.25, 50: 4.0}
PER_HOUR = {15: 135.0, 35: 195.0, 50: 240.0}


def make_schools(rows):
    """rows: list of (id, lat, lon, attendees). Synthetic test schools."""
    frame = pd.DataFrame(rows, columns=["school_id", "latitude", "longitude", "attendees"])
    frame.insert(1, "name", frame["school_id"])
    frame["attendees"] = frame["attendees"].astype(int)
    return frame


def write_data(directory: Path, schools_text: str, fixed: str | None = None,
               distance: str | None = None, time: str | None = None) -> Path:
    (directory / "attendants_dataset.csv").write_text(schools_text)
    if fixed is not None:
        (directory / "fixed_costs.csv").write_text(fixed)
    if distance is not None:
        (directory / "direct_distance_costs.csv").write_text(distance)
    if time is not None:
        (directory / "time_costs.csv").write_text(time)
    return directory


@pytest.fixture(scope="session")
def real_dataset():
    return load_dataset(ROOT)
