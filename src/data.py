"""Data loading.

Reads the supplied challenge CSVs, validates them (see ``validation.py``) and
returns a ``Dataset``. The raw files are only ever read, never written.

Source files (project root):
    attendants_dataset.csv     outcode, latitude, longitude, attendants
    fixed_costs.csv            Bus size, Fixed price
    direct_distance_costs.csv  Bus size, Price per mile
    time_costs.csv             Bus size, Price per hour

Units: coordinates are WGS84 decimal degrees, distances are miles, times are
hours. The data does not state a currency, so costs are shown as "cost units".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .validation import ValidationReport, validate_bus_table, validate_schools

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT

SCHOOLS_FILE = "attendants_dataset.csv"
FIXED_FILE = "fixed_costs.csv"
DISTANCE_FILE = "direct_distance_costs.csv"
TIME_FILE = "time_costs.csv"

COST_MODELS = ("fixed", "distance", "time")
MODEL_FILES = {"fixed": FIXED_FILE, "distance": DISTANCE_FILE, "time": TIME_FILE}
MODEL_PRICE_COLUMNS = {"fixed": "Fixed price", "distance": "Price per mile", "time": "Price per hour"}
MODEL_LABELS = {
    "fixed": "Fixed cost per bus",
    "distance": "Straight-line distance x rate per mile",
    "time": "Travel time x rate per hour",
}
MODEL_UNITS = {"fixed": "per bus", "distance": "per mile", "time": "per hour"}

CURRENCY_LABEL = "cost units"


@dataclass(frozen=True)
class Event:
    """Event destination. Coordinates come from the challenge data README."""

    name: str = "Principality Stadium, Westgate St, Cardiff CF10 1NS"
    latitude: float = 51.478283
    longitude: float = -3.182652


@dataclass
class Dataset:
    """Validated inputs for the optimiser.

    ``schools`` columns: school_id, name, latitude, longitude, attendees.
    ``rates`` maps cost model -> {bus size: rate}. A model whose file failed
    validation is absent from ``rates``.
    """

    schools: pd.DataFrame
    rates: dict[str, dict[int, float]]
    event: Event = field(default_factory=Event)
    report: ValidationReport = field(default_factory=ValidationReport)

    @property
    def available_models(self) -> list[str]:
        return [m for m in COST_MODELS if self.rates.get(m)]

    def bus_sizes(self, model: str) -> list[int]:
        return sorted(self.rates.get(model, {}))

    def with_attendance(self, overrides: dict[str, int]) -> "Dataset":
        """Return a copy with edited attendance. The original is not changed.

        Every value must be a whole number >= 0 (0 means the school needs no bus).
        """
        schools = self.schools.copy()
        known = set(schools["school_id"])
        for school_id, value in overrides.items():
            if school_id not in known:
                raise ValueError(f"Unknown school {school_id!r}.")
            count = whole_attendance(value)
            if count is None:
                raise ValueError(f"Attendance for {school_id} must be a whole number >= 0, got {value!r}.")
            schools.loc[schools["school_id"] == school_id, "attendees"] = count
        report = ValidationReport(
            errors=list(self.report.errors),
            warnings=list(self.report.warnings),
            excluded=list(self.report.excluded),
        )
        for model, rates in self.rates.items():
            report.check_capacity(schools, model, rates)
        return Dataset(schools=schools, rates=self.rates, event=self.event, report=report)


def whole_attendance(value: object) -> int | None:
    """``value`` as a whole number >= 0, or None if it is not one.

    Rejects missing values, text, booleans, decimals (15.5) and negatives.
    A float with no fractional part (15.0) is accepted as 15.
    """
    if value is None or isinstance(value, (bool, str)):
        return None  # typed text is not accepted, even "15"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number < 0 or number != int(number):
        return None
    return int(number)


def attendance_edits(
    edited_rows: dict, school_ids: list[str], originals: list[int], column: str = "Attendees"
) -> tuple[dict[str, int], list[str]]:
    """Turn ``st.data_editor`` edits into attendance overrides.

    ``edited_rows`` is the editor's ``{row position: {column: value}}`` state.
    Returns ``(overrides, errors)``. Overrides hold only values that differ
    from the original; errors describe values that were rejected. A rejected
    value never becomes an override, so the optimiser cannot run on it.
    """
    overrides: dict[str, int] = {}
    errors: list[str] = []
    for position, change in sorted(edited_rows.items(), key=lambda kv: int(kv[0])):
        if column not in change:
            continue
        k = int(position)
        if not 0 <= k < len(school_ids):
            errors.append(f"Edited row {k} does not match any school.")
            continue
        school, value = school_ids[k], change[column]
        if value is None or (isinstance(value, float) and math.isnan(value)):
            errors.append(f"{school}: attendance is empty. Enter a whole number (0 = no bus needed).")
            continue
        count = whole_attendance(value)
        if count is None:
            errors.append(f"{school}: {value!r} is not a whole number >= 0.")
            continue
        if count != int(originals[k]):
            overrides[school] = count
    return overrides, errors


def _read_csv(path: Path, report: ValidationReport) -> pd.DataFrame | None:
    if not path.exists():
        report.errors.append(f"File not found: {path.name}")
        return None
    try:
        frame = pd.read_csv(path, dtype=str, skipinitialspace=True)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        report.errors.append(f"Could not read {path.name}: {exc}")
        return None
    frame.columns = [str(c).strip() for c in frame.columns]
    return frame


def load_dataset(data_dir: Path | str = DEFAULT_DATA_DIR) -> Dataset:
    """Load and validate all challenge files from ``data_dir``."""
    data_dir = Path(data_dir)
    report = ValidationReport()

    raw_schools = _read_csv(data_dir / SCHOOLS_FILE, report)
    schools = validate_schools(raw_schools, report)

    rates: dict[str, dict[int, float]] = {}
    for model in COST_MODELS:
        file_name = MODEL_FILES[model]
        path = data_dir / file_name
        if not path.exists():
            report.warnings.append(f"{file_name} not found: the '{model}' cost model is unavailable.")
            continue
        raw = _read_csv(path, report)
        table = validate_bus_table(raw, MODEL_PRICE_COLUMNS[model], file_name, report)
        if table:
            rates[model] = table

    if not rates:
        report.errors.append("No valid bus cost table found: nothing can be optimised.")
    for model, table in rates.items():
        report.check_capacity(schools, model, table)

    return Dataset(schools=schools, rates=rates, report=report)
