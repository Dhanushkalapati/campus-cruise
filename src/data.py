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

import io
import math
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .distances import OSRM_MAX_POINTS, haversine_miles
from .validation import SCHOOL_COLUMNS, ValidationReport, validate_bus_table, validate_schools

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

# Default solver time limits in seconds, used by the dashboard, demo.py and
# benchmark.py unless the caller passes its own limit. The fixed model is exact
# and usually proves optimality well inside its limit; the time left groups
# nearby schools at the same cost.
# The routing models are heuristic: guided local search keeps improving until
# the limit is reached, so a run uses the full limit. A longer limit gives the
# search more time; it does not guarantee a cheaper plan.
DEFAULT_TIME_LIMITS = {"fixed": 20.0, "distance": 300.0, "time": 300.0}
MIN_TIME_LIMIT = 5
MAX_TIME_LIMIT = 600

CURRENCY_LABEL = "cost units"


@dataclass(frozen=True)
class Event:
    """Event destination. The default coordinates come from the challenge data README.

    Every matrix, route and validation check takes the destination from here,
    so a different ``Event`` changes the optimisation, not only the label.
    """

    name: str = "Principality Stadium, Westgate St, Cardiff CF10 1NS"
    latitude: float = 51.478283
    longitude: float = -3.182652

    @property
    def short_name(self) -> str:
        """Name up to the first comma, for route labels ("A -> B -> Principality Stadium")."""
        return self.name.split(",")[0].strip() or "Event"


DEFAULT_EVENT = Event()
MAX_VENUE_NAME = 120


def make_event(name: object, latitude: object, longitude: object) -> Event:
    """A validated ``Event``. Raises ValueError listing every problem.

    Rejects an empty name, missing or non-numeric coordinates, and values
    outside latitude -90..90 or longitude -180..180.
    """
    problems = []
    text = "" if name is None else str(name).strip()
    if not text:
        problems.append("Enter a venue name.")
    elif len(text) > MAX_VENUE_NAME:
        problems.append(f"The venue name must be at most {MAX_VENUE_NAME} characters.")
    coords = []
    for label, value, low, high in (("Latitude", latitude, -90, 90), ("Longitude", longitude, -180, 180)):
        number = _coordinate(value)
        if number is None:
            problems.append(f"{label} is missing or not a number.")
        elif not low <= number <= high:
            problems.append(f"{label} must be between {low} and {high}, got {number:g}.")
        coords.append(number)
    if problems:
        raise ValueError(" ".join(problems))
    return Event(text, float(coords[0]), float(coords[1]))


def _coordinate(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value.strip()) if isinstance(value, str) else float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


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

    def with_price_change(self, model: str, percent: int) -> "Dataset":
        """Return a copy where every bus price of ``model`` changes by ``percent`` (what-if only).

        The original rates are not changed. ``percent`` must be a whole number
        from -90 to +300, so every price stays positive.
        """
        if model not in self.rates:
            raise ValueError(f"Cost model {model!r} is not available.")
        if isinstance(percent, bool) or not isinstance(percent, int) or not -90 <= percent <= 300:
            raise ValueError("The price change must be a whole number of percent from -90 to 300.")
        rates = {m: dict(r) for m, r in self.rates.items()}
        rates[model] = {size: price * (100 + percent) / 100 for size, price in rates[model].items()}
        return Dataset(schools=self.schools.copy(), rates=rates, event=self.event, report=self.report)


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


def load_dataset(
    data_dir: Path | str = DEFAULT_DATA_DIR,
    schools_csv: bytes | None = None,
    event: Event | None = None,
) -> Dataset:
    """Load and validate all challenge files from ``data_dir``.

    ``schools_csv``: content of an uploaded school file to use instead of
    ``attendants_dataset.csv`` (same columns). The file on disk is not changed.
    ``event``: destination to use instead of ``DEFAULT_EVENT``.
    """
    data_dir = Path(data_dir)
    report = ValidationReport()

    if schools_csv is None:
        raw_schools = _read_csv(data_dir / SCHOOLS_FILE, report)
    else:
        raw_schools = _read_csv_bytes(schools_csv, "uploaded school file", report)
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

    return Dataset(schools=schools, rates=rates, event=event or DEFAULT_EVENT, report=report)


# --------------------------------------------------------------------------- #
# Uploaded school files and venue files
# --------------------------------------------------------------------------- #

MAX_UPLOAD_BYTES = 2_000_000
SCHOOL_TEMPLATE_CSV = (
    "outcode,latitude,longitude,attendants\n"
    "CF10,51.4816,-3.1791,32\n"
    "CF14,51.5230,-3.2010,18\n"
    "NP20,51.5880,-2.9980,25\n"
)
VENUE_COLUMNS = {"name", "latitude", "longitude"}
FAR_FROM_VENUE_MILES = 150.0


def _read_csv_bytes(content: bytes, label: str, report: ValidationReport) -> pd.DataFrame | None:
    """Read uploaded CSV bytes as text columns (same conventions as ``_read_csv``)."""
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        report.errors.append(f"Could not read the {label}: it is not UTF-8 text. Save it as CSV (UTF-8).")
        return None
    try:
        frame = pd.read_csv(io.StringIO(text), dtype=str, skipinitialspace=True)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        report.errors.append(f"Could not read the {label}: {exc}")
        return None
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    return frame


@dataclass
class SchoolUpload:
    """Result of checking an uploaded school file. Usable only when ``errors`` is empty.

    Invalid rows are never dropped: any invalid row rejects the whole file, so
    a plan cannot silently leave schools out.
    """

    schools: pd.DataFrame
    errors: list[str]
    warnings: list[str]

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def total_attendees(self) -> int:
        return int(self.schools["attendees"].sum()) if len(self.schools) else 0


def check_school_upload(content: bytes, max_bytes: int = MAX_UPLOAD_BYTES) -> SchoolUpload:
    """Validate an uploaded school CSV (outcode, latitude, longitude, attendants)."""
    empty = validate_schools(None, ValidationReport())
    if not content or not content.strip():
        return SchoolUpload(empty, ["The file is empty. Use the template: one header row, then one row per school."], [])
    if len(content) > max_bytes:
        return SchoolUpload(empty, [f"The file is {len(content) / 1e6:.1f} MB; the limit is "
                                    f"{max_bytes / 1e6:.0f} MB."], [])
    report = ValidationReport()
    raw = _read_csv_bytes(content, "file", report)
    if raw is None:
        return SchoolUpload(empty, report.errors, [])
    missing = SCHOOL_COLUMNS - set(raw.columns)
    if missing:
        return SchoolUpload(empty, [
            f"Missing column(s): {', '.join(sorted(missing))}. The first row must be: "
            "outcode,latitude,longitude,attendants (download the template)."], [])
    if raw.empty:
        return SchoolUpload(empty, ["The file has a header row but no schools."], [])
    schools = validate_schools(raw, report)

    errors = [e for e in report.errors if not e.startswith("No valid school rows")]
    if report.excluded:
        errors.insert(0, f"{len(report.excluded)} row(s) have invalid values. Fix them and upload again "
                         "(no row was dropped): " + "; ".join(report.excluded))
    warnings = [w for w in report.warnings if "excluded because of invalid data" not in w]
    extra = sorted(set(raw.columns) - SCHOOL_COLUMNS)
    if extra:
        warnings.append(f"Extra column(s) ignored: {', '.join(extra)}.")
    if not errors:
        if int(schools["attendees"].sum()) == 0:
            errors.append("Every school has zero attendants: there is nothing to plan.")
        same_place = schools[schools.duplicated(["latitude", "longitude"], keep=False)]
        if len(same_place):
            groups = same_place.groupby(["latitude", "longitude"])["school_id"].apply(", ".join)
            warnings.append("Schools with identical coordinates (check for duplicate records): "
                            + "; ".join(groups.tolist()) + ".")
        if len(schools) + 1 > OSRM_MAX_POINTS:
            warnings.append(f"{len(schools)} schools plus the venue exceed the OSRM limit of {OSRM_MAX_POINTS} "
                            "points per request, so the driving-time goal will use the labelled fixed-speed "
                            "estimate instead of road times.")
    return SchoolUpload(empty if errors else schools, errors, warnings)


def parse_venue_csv(content: bytes) -> Event:
    """One-row venue CSV (name, latitude, longitude) as a validated Event. Raises ValueError."""
    report = ValidationReport()
    raw = _read_csv_bytes(content or b"", "venue file", report)
    if raw is None:
        raise ValueError(" ".join(report.errors))
    missing = VENUE_COLUMNS - set(raw.columns)
    if missing:
        raise ValueError(f"Venue file is missing column(s): {', '.join(sorted(missing))}. "
                         "The first row must be: name,latitude,longitude")
    if len(raw) != 1:
        raise ValueError(f"Venue file must have exactly one venue row, found {len(raw)}.")
    row = raw.iloc[0]
    return make_event(*(None if pd.isna(row[c]) else row[c] for c in ("name", "latitude", "longitude")))


def far_schools(schools: pd.DataFrame, event: Event, limit_miles: float = FAR_FROM_VENUE_MILES) -> list[str]:
    """Schools more than ``limit_miles`` straight-line from the venue (often a coordinate typo)."""
    return [row.school_id for row in schools.itertuples()
            if haversine_miles(row.latitude, row.longitude, event.latitude, event.longitude) > limit_miles]
