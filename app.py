"""Campus Cruise: Streamlit dashboard.

Run:  streamlit run app.py
All optimisation logic lives in ``src/``; this file only handles the UI.
"""

from __future__ import annotations

import hashlib
import math
import time

import pandas as pd
import streamlit as st

from src.data import (
    COST_MODELS,
    CURRENCY_LABEL,
    DEFAULT_EVENT,
    DEFAULT_TIME_LIMITS,
    FAR_FROM_VENUE_MILES,
    MAX_TIME_LIMIT,
    MAX_VENUE_NAME,
    MIN_TIME_LIMIT,
    MODEL_LABELS,
    MODEL_UNITS,
    SCHOOL_TEMPLATE_CSV,
    Event,
    attendance_edits,
    check_school_upload,
    far_schools,
    load_dataset,
    make_event,
    parse_venue_csv,
)
from src.assistant import (
    MAX_REQUEST_CHARS,
    REQUEST_TYPES,
    AssistantUnavailable,
    assist,
    build_context,
    provider_from_config,
    validate_proposal,
)
from src.distances import (
    OSRM_MAX_POINTS,
    SOURCE_APPROX,
    RoutingServiceError,
    TimeMatrix,
    approximate_time_matrix,
    road_path,
    road_time_matrix,
    straight_line_matrix,
)
from src.geocode import GeocodeError, geocode
from src.model import FEASIBLE, INFEASIBLE, INVALID, NO_SOLUTION, OPTIMAL, Constraints, Solution
from src.freshness import (
    AUTO_RUN_DEFAULT,
    CURRENT,
    NONE,
    SETTINGS_CHANGED,
    STALE,
    cache_put,
    freshness,
    run_key,
    should_auto_run,
)
from src.explain import (
    audit_report,
    distance_impact,
    distance_lines,
    explain_plan,
    metrics_csv,
    model_difference,
    plan_totals,
    run_summary,
    summary_row,
)
from src.scenario import (
    MAX_PRICE,
    MAX_SCALE,
    MIN_PRICE,
    MIN_SCALE,
    ScenarioInputs,
    attendance_diff,
    comparison_table,
    overrides_vs_source,
    scenario_attendance,
    scenario_dataset,
)
from src.mapview import bus_of_school, filter_routes, map_view, route_details, school_rows
from src.ml_travel_time import SOURCE_ML, data_readiness, load_model
from src.planner import optimise, points_of
from src.report import (
    attendance_changes,
    bus_table,
    compare_plans,
    cost_shares,
    impact_metrics,
    plan_csv,
    plan_report,
    savings,
)

try:
    import folium
    from streamlit_folium import st_folium

    HAS_MAP = True
except ImportError:  # map is optional; tables still work
    HAS_MAP = False

st.set_page_config(page_title="Campus Cruise", page_icon="🚌", layout="wide")

st.markdown(
    """
    <style>
      :root { --cc-accent: #D6453D; --cc-ink: #1F2933; --cc-muted: #5F6B7A; --cc-line: #E3E6EA;
              --cc-soft: #F5F6F8; }
      .block-container { padding-top: 3.4rem; padding-bottom: 6rem; max-width: 1320px; }
      /* Header */
      .cc-brand { display: flex; align-items: center; gap: .75rem; }
      .cc-logo { width: 42px; height: 42px; border-radius: 10px; background: var(--cc-accent);
          display: flex; align-items: center; justify-content: center; font-size: 1.35rem; }
      .cc-title { font-size: 1.45rem; font-weight: 700; color: var(--cc-ink); line-height: 1.15;
          letter-spacing: -.01em; }
      .cc-sub { color: var(--cc-muted); font-size: .9rem; }
      /* Section headings inside tabs */
      .cc-section { font-size: 1rem; font-weight: 650; color: var(--cc-ink); margin: .5rem 0 .15rem; }
      /* Metric cards: one size for every card */
      div[data-testid="stMetricValue"] { font-size: 1.7rem; font-weight: 650; }
      div[data-testid="stMetricLabel"] p { color: var(--cc-muted); font-weight: 600; }
      /* Tabs */
      .stTabs [data-baseweb="tab-list"] { gap: 1.4rem; }
      .stTabs [data-baseweb="tab"] p { font-weight: 600; }
      /* Sidebar */
      .cc-side-title { font-weight: 700; font-size: 1.05rem; color: var(--cc-ink); margin-bottom: .1rem; }
      /* Bus list beside the map */
      .cc-buslist { max-height: 470px; overflow-y: auto; border: 1px solid var(--cc-line);
          border-radius: .6rem; }
      .cc-bus { display: flex; gap: .6rem; padding: .5rem .75rem; border-bottom: 1px solid var(--cc-line);
          font-size: .86rem; line-height: 1.35; }
      .cc-bus:last-child { border-bottom: none; }
      .cc-dot { width: 10px; height: 10px; border-radius: 50%; flex: none; margin-top: .3rem; }
      .cc-bus b { color: var(--cc-ink); }
      .cc-bus span.cc-muted { color: var(--cc-muted); }
      /* Floating assistant button: the keyed container is pinned bottom-right. */
      .st-key-cc_chat_fab { position: fixed; right: 1.5rem; bottom: 1.5rem; z-index: 1000;
          width: auto !important; }
      .st-key-cc_chat_fab button { border-radius: 999px; padding: .5rem 1.05rem;
          box-shadow: 0 4px 14px rgba(31, 41, 51, .18); }
      .st-key-cc_chat_fab button:focus-visible { outline: 3px solid var(--cc-ink); outline-offset: 2px; }
      /* Panel width: compact on desktop, nearly full width on phones. */
      div[data-testid="stPopoverBody"]:has(.st-key-cc_chat_panel) { width: min(440px, calc(100vw - 2rem)); }
      /* The panel scrolls as a whole; the newest answer sits right under the typing bar. */
      div[data-testid="stPopoverBody"]:has(.st-key-cc_chat_panel) { max-height: calc(100vh - 6rem);
          overflow-y: auto; }
      @media (max-width: 640px) { .st-key-cc_chat_fab { right: .75rem; bottom: .75rem; } }
    </style>
    """,
    unsafe_allow_html=True,
)

# Muted, distinguishable colours (Tableau 10 plus two). Plans with more buses reuse them in order.
PALETTE = [
    "#4E79A7", "#F28E2B", "#E15759", "#59A14F", "#B07AA1", "#76B7B2", "#EDC948", "#9C755F",
    "#FF9DA7", "#2F4B7C", "#A05195", "#7F7F7F",
]


# --------------------------------------------------------------------------- #
# State and cached inputs
# --------------------------------------------------------------------------- #


@st.cache_resource(max_entries=8)
def get_dataset(schools_csv: bytes | None = None, event_name: str = DEFAULT_EVENT.name,
                event_lat: float = DEFAULT_EVENT.latitude, event_lon: float = DEFAULT_EVENT.longitude):
    """Validated inputs. ``schools_csv`` None = the bundled CSV; an upload is never written to disk.
    Cached per (school file, venue), so switching back to an earlier choice is instant."""
    return load_dataset(schools_csv=schools_csv, event=Event(event_name, event_lat, event_lon))


@st.cache_data(show_spinner="Fetching travel-time matrix...", max_entries=16)
def get_road_matrix(points_key: tuple, allow_online: bool) -> TimeMatrix:
    """Road times only (disk cache or OSRM). ``points_key`` holds every school and then the
    venue, so other schools or another venue never reuse this matrix. Failures raise and
    are not cached, so a later run retries OSRM instead of reusing a stored fallback."""
    return road_time_matrix([tuple(p) for p in points_key], allow_online=allow_online, allow_approximate=False)


@st.cache_resource
def get_ml_model():
    """Trained artefact from models/ (never retrained by the app). None if missing or malformed."""
    return load_model()


TIME_SOURCES = {
    "osrm": "Road times (OSRM)",
    "ml": "ML estimate of OSRM (experimental)",
    "estimate": "Fixed-speed estimate (offline)",
}
# Shown wherever the Routes tab shows driving minutes.
DRIVING_TIME_NOTE = ("Estimated driving time. Does not include live traffic, school boarding time, or delays. "
                     "Actual journeys may take longer.")


def get_time_matrix(points_key: tuple, allow_online: bool, source: str = "osrm") -> TimeMatrix:
    """Travel times for the time model, from the source chosen in the sidebar. Always labelled."""
    points = [tuple(p) for p in points_key]  # schools, then the venue
    if source == "estimate":
        return TimeMatrix(approximate_time_matrix(points), SOURCE_APPROX,
                          ["Chosen in the sidebar. Travel times are approximate, not road routing."])
    if source == "ml":
        ml = get_ml_model()
        if ml is not None and ml.usable:
            notes = ["Chosen in the sidebar. Learned from cached OSRM estimates; not road routing "
                     "and not observed journeys. See Data & validation."]
            if custom_inputs():
                notes.append(ML_REGION_NOTE)
            return TimeMatrix(ml.predict_hours(points), SOURCE_ML, notes)
        return TimeMatrix(approximate_time_matrix(points), SOURCE_APPROX,
                          ["The ML artefact is missing or did not beat the fixed-speed estimate on held-out "
                           "schools, so the fixed-speed estimate was used."])
    try:
        return get_road_matrix(points_key, allow_online)
    except RoutingServiceError as exc:
        return TimeMatrix(approximate_time_matrix(points), SOURCE_APPROX,
                          [str(exc), "Using the offline estimate. Travel times are approximate, not road routing."])


ML_REGION_NOTE = ("The ML estimate was trained only on the 60 South Wales schools and Principality Stadium. "
                  "It is not validated for other schools or venues.")


def points_key() -> tuple:
    """Every school (dataset order) and then the venue: the key of every matrix.
    Raw values, not pre-rounded: the disk cache key formats them itself, and rounding twice
    can shift the last digit and miss a cached matrix."""
    return tuple((float(a), float(b)) for a, b in points_of(dataset))


def same_venue(a: Event, b: Event) -> bool:
    """Compare venues by value. Not ``a == b``: after Streamlit reloads ``src.data`` the
    stored and the new ``Event`` are different classes and never compare equal."""
    return (a.name, a.latitude, a.longitude) == (b.name, b.latitude, b.longitude)


def custom_inputs() -> bool:
    """True when the school data or the venue differs from the bundled defaults."""
    ss = st.session_state
    return ss.get("school_source") is not None or not same_venue(ss.get("venue", DEFAULT_EVENT), DEFAULT_EVENT)


def display_hours() -> tuple:
    """Road hours for display only, from the OSRM cache. No network request. (None, None) if absent."""
    try:
        tm = get_road_matrix(points_key(), False)
        return tm.hours, tm.source
    except RoutingServiceError:
        return None, None


def init_state():
    ss = st.session_state
    ss.setdefault("must_share", [])
    ss.setdefault("cannot_share", [])
    ss.setdefault("solo", [])
    ss.setdefault("overrides", {})         # school -> edited attendance (valid values only)
    ss.setdefault("attendance_errors", [])  # rejected edits; block the optimiser
    ss.setdefault("current", None)   # dict: result, model, overrides, constraints, dataset
    ss.setdefault("previous", None)
    ss.setdefault("by_model", {})    # model -> latest run of that model (same dict shape as current)
    ss.setdefault("editor_base", {})  # overrides the attendance editor table starts from
    ss.setdefault("assistant_proposal", None)  # last validated assistant Proposal (not applied)
    ss.setdefault("assistant_log", [])  # (request, type, outcome) for this session
    ss.setdefault("assistant_msg", None)
    ss.setdefault("chat_history", [])  # floating chat: {"role", "text"} or {"role", "proposal"}
    ss.setdefault("last_valid", None)    # latest run with a validated plan (any model)
    ss.setdefault("result_cache", {})    # RunKey -> run entry; reused when inputs repeat
    ss.setdefault("last_attempt", None)  # RunKey of the latest started run (blocks duplicate auto-runs)
    ss.setdefault("run_error", None)     # dict(key, message) of the latest failed or rejected run
    ss.setdefault("solver_runs", 0)      # number of optimise() calls this session
    ss.setdefault("reused", None)        # RunKey of a result just taken from the cache
    ss.setdefault("run_log", [])         # audit trail: one row per run attempt (no personal data)
    ss.setdefault("approved_proposal", None)  # assistant proposal approved for the next run
    # Inputs. school_source None = bundled attendants_dataset.csv; else a validated upload
    # (dict: name, content, digest, warnings). The upload lives only in this session.
    ss.setdefault("school_source", None)
    ss.setdefault("upload_errors", [])     # errors of the latest rejected upload (active data unchanged)
    ss.setdefault("upload_n", 0)           # bumped to clear the file uploader widget
    ss.setdefault("venue", DEFAULT_EVENT)  # validated destination used by every calculation
    ss.setdefault("venue_errors", [])
    ss.setdefault("venue_name", DEFAULT_EVENT.name)  # destination form fields
    ss.setdefault("venue_lat", f"{DEFAULT_EVENT.latitude:.6f}")
    ss.setdefault("venue_lon", f"{DEFAULT_EVENT.longitude:.6f}")
    ss.setdefault("inputs_msg", None)


# --------------------------------------------------------------------------- #
# School data and event destination
# --------------------------------------------------------------------------- #

# Session keys that hold plans, edits or widget values for one set of schools and one venue.
PLAN_STATE = {
    "overrides": {}, "attendance_errors": [], "editor_base": {}, "must_share": [], "cannot_share": [],
    "solo": [], "current": None, "previous": None, "by_model": {}, "last_valid": None, "result_cache": {},
    "last_attempt": None, "run_error": None, "reused": None, "assistant_proposal": None,
    "approved_proposal": None, "scenario": None,
}
PLAN_WIDGETS = ("att_editor", "solo_select", "must_share_a", "must_share_b", "cannot_share_a",
                "cannot_share_b", "map_buses", "map_school", "map_plan", "scn_editor", "scn_scale",
                "scn_confirm")


def reset_plan_state():
    """Forget every plan and edit. Called when the schools or the venue change, because a plan,
    an attendance edit or a sharing rule made for other inputs is not valid for the new ones."""
    ss = st.session_state
    for k, v in PLAN_STATE.items():
        ss[k] = type(v)() if isinstance(v, (dict, list)) else v
    for k in PLAN_WIDGETS:
        ss.pop(k, None)


def school_file_key() -> str:
    return f"school_file_{st.session_state.upload_n}"


def on_school_upload():
    """Check a new file. A valid file replaces the active schools; an invalid one changes nothing."""
    ss = st.session_state
    f = ss.get(school_file_key())
    if f is None:  # file removed with the uploader's x: back to the bundled data
        if ss.school_source is not None:
            use_default_schools()
        return
    content = f.getvalue()
    check = check_school_upload(content)
    if not check.ok:
        ss.upload_errors = [f"{f.name} was not used. " + check.errors[0]] + check.errors[1:]
        return
    digest = hashlib.sha256(content).hexdigest()
    ss.upload_errors = []
    if ss.school_source is not None and ss.school_source["digest"] == digest:
        return
    ss.school_source = {"name": f.name, "content": content, "digest": digest, "warnings": check.warnings}
    reset_plan_state()
    ss.inputs_msg = (f"Loaded {f.name}: {len(check.schools)} schools, {check.total_attendees} attendees. "
                     "Previous plans and edits were cleared.")


def use_default_schools():
    ss = st.session_state
    changed = ss.school_source is not None
    ss.school_source, ss.upload_errors = None, []
    ss.upload_n += 1  # new uploader key: the widget shows no file
    if changed:
        reset_plan_state()
        ss.inputs_msg = "Using the default school data. Previous plans and edits were cleared."


def set_venue(event: Event):
    ss = st.session_state
    ss.venue_errors = []
    ss.venue_name, ss.venue_lat, ss.venue_lon = event.name, f"{event.latitude:.6f}", f"{event.longitude:.6f}"
    if not same_venue(event, ss.venue):
        ss.venue = event
        reset_plan_state()
        ss.inputs_msg = (f"Destination set to {event.short_name} ({event.latitude:.6f}, {event.longitude:.6f}). "
                         "Previous plans and edits were cleared.")


def venue_from_search(query: str) -> bool:
    """Look ``query`` up and make the best match the venue. Other matches stay selectable.
    On failure the current venue stays and the reason is shown. Returns True on success."""
    ss = st.session_state
    try:
        places = geocode(query)
    except GeocodeError as exc:
        ss.venue_errors = [f"Destination not changed. {exc}"]
        return False
    ss.venue_matches = places
    ss.venue_match = 0
    set_venue(make_event(places[0].name[:MAX_VENUE_NAME], places[0].latitude, places[0].longitude))
    return True


def search_venue():
    """Search form submit: typed name or address -> coordinates -> venue."""
    venue_from_search(st.session_state.get("venue_query", ""))


def pick_venue_match():
    """Another of the search matches chosen."""
    ss = st.session_state
    places = ss.get("venue_matches") or []
    k = ss.get("venue_match", 0)
    if 0 <= k < len(places):
        p = places[k]
        set_venue(make_event(p.name[:MAX_VENUE_NAME], p.latitude, p.longitude))


def apply_venue():
    """Manual form submit. Invalid input is reported and the current venue stays.

    If only the name was changed (coordinates still those of the current venue), the
    name is looked up, so typing a new place name does move the venue."""
    ss = st.session_state
    current = ss.venue
    name = str(ss.get("venue_name") or "").strip()
    same_coords = (str(ss.get("venue_lat", "")).strip() == f"{current.latitude:.6f}"
                   and str(ss.get("venue_lon", "")).strip() == f"{current.longitude:.6f}")
    if name and name != current.name and same_coords:
        venue_from_search(name)
        return
    try:
        event = make_event(ss.venue_name, ss.venue_lat, ss.venue_lon)
    except ValueError as exc:
        ss.venue_errors = [f"Destination not changed. {exc}"]
        return
    clear_matches()
    set_venue(event)


def clear_matches():
    """Search matches belong to one search; drop them when the venue is set another way."""
    st.session_state.pop("venue_matches", None)
    st.session_state.pop("venue_match", None)


def reset_venue():
    clear_matches()
    set_venue(DEFAULT_EVENT)


def on_venue_file():
    ss = st.session_state
    f = ss.get("venue_file")
    if f is None:
        return
    try:
        event = parse_venue_csv(f.getvalue())
        clear_matches()
        set_venue(event)
    except ValueError as exc:
        ss.venue_errors = [f"{f.name} was not used. {exc}"]


def request_run():
    """Callback of every Run button. Callbacks run before the script body."""
    st.session_state.run_requested = True


# Widget callbacks. Streamlit runs callbacks before the script body, so every
# edit is in session state before the "Run optimiser" block reads it, even
# when the edit and the button click arrive in the same rerun.


def editor_values() -> list[int]:
    """Attendance shown in the editor before any cell edit: source data plus editor_base."""
    base = st.session_state.editor_base
    return [int(base.get(sid, a)) for sid, a in zip(school_ids, dataset.schools["attendees"])]


def sync_attendance():
    ss = st.session_state
    state = ss.get("att_editor") or {}
    edits, ss.attendance_errors = attendance_edits(
        state.get("edited_rows", {}), school_ids, editor_values(), "Attendees")
    source = dict(zip(school_ids, dataset.schools["attendees"]))
    merged = {**ss.editor_base, **edits}
    ss.overrides = {k: v for k, v in merged.items() if v != source[k]}


def reset_attendance():
    ss = st.session_state
    ss.overrides, ss.attendance_errors, ss.editor_base = {}, [], {}
    ss.pop("att_editor", None)


def assistant_approve():
    """Apply the shown proposal, after re-checking it against the current state, then run."""
    ss = st.session_state
    shown = ss.assistant_proposal
    ss.assistant_proposal = None
    if shown is None or not shown.needs_approval:
        return
    if ss.attendance_errors:
        ss.assistant_msg = ("error", "Fix the attendance errors in Adjust plan first. Nothing was changed.")
        return
    p = validate_proposal(shown.raw, shown.request, dataset.with_attendance(ss.overrides),
                          current_constraints(), ss.get("model_radio", models[0]))
    if not p.needs_approval:
        ss.assistant_msg = ("error", "The inputs changed and the proposal is no longer valid: "
                            + " ".join(p.errors) + " Nothing was changed.")
        ss.assistant_log.append((p.request, p.request_type, "approved, then blocked by re-check"))
        return
    if p.attendance:
        source = dict(zip(school_ids, dataset.schools["attendees"]))
        merged = {**ss.overrides, **p.attendance}
        ss.overrides = {k: v for k, v in merged.items() if v != source[k]}
        ss.editor_base = dict(ss.overrides)
        ss.pop("att_editor", None)  # editor redraws from editor_base
    if p.rules:
        after = p.constraints_after(current_constraints())
        ss.must_share, ss.cannot_share, ss.solo = after.must_share, after.cannot_share, after.solo
        ss.solo_select = list(ss.solo)
    ss.model_radio = p.cost_model
    ss.assistant_run = True
    ss.approved_proposal = p  # the run summary reports what was applied
    ss.assistant_msg = ("success", "Approved and applied. The optimiser is running now; its result "
                                   "follows as a separate message.")
    ss.assistant_log.append((p.request, p.request_type, "approved and applied"))


def assistant_reject():
    ss = st.session_state
    p = ss.assistant_proposal
    ss.assistant_proposal = None
    if p is not None:
        ss.assistant_log.append((p.request, p.request_type, "rejected, nothing changed"))
    ss.assistant_msg = ("info", "Rejected. Nothing was changed.")


def sync_solo():
    st.session_state.solo = list(st.session_state.solo_select)


def add_pair(key: str):
    ss = st.session_state
    a, b = ss[f"{key}_a"], ss[f"{key}_b"]
    pair = tuple(sorted((a, b)))
    if a == b:
        ss[f"{key}_msg"] = "Pick two different schools."
    elif pair not in ss[key]:
        ss[key].append(pair)


def remove_pair(key: str, pair: tuple):
    if pair in st.session_state[key]:
        st.session_state[key].remove(pair)


def current_constraints() -> Constraints:
    ss = st.session_state
    return Constraints(list(ss.must_share), list(ss.cannot_share), list(ss.solo))


def matches_current_inputs(entry: dict) -> bool:
    """True when a stored run used the attendance and constraints now in the editor."""
    ss = st.session_state
    return entry["overrides"] == ss.overrides and entry["constraints"] == current_constraints()


def make_current(entry: dict):
    """Show ``entry`` as the plan. The plan it replaces becomes the comparison point."""
    ss = st.session_state
    if ss.current is not None and ss.current is not entry and ss.current["result"].solution.has_plan:
        ss.previous = ss.current
    ss.current = entry
    ss.by_model[entry["model"]] = entry


def money(x: float) -> str:
    return f"{x:,.2f}"


def fmt_money(x) -> str | None:
    return None if x is None else money(x)


def status_badge(sol) -> str:
    """Solver status as a coloured badge (Streamlit markdown). Green only when optimality is proven."""
    text = sol.status_text()
    if sol.proven_optimal:
        return f":green-badge[:material/verified: {text}]"
    if sol.has_plan:
        return f":orange-badge[{text}]"
    return f":red-badge[{text}]"


def section(title: str):
    """Small section heading used inside tabs (one style everywhere)."""
    st.markdown(f'<div class="cc-section">{title}</div>', unsafe_allow_html=True)


init_state()
ss = st.session_state
_src, _venue = ss.school_source, ss.venue
dataset = get_dataset(None if _src is None else _src["content"], _venue.name, _venue.latitude, _venue.longitude)
school_ids = dataset.schools["school_id"].tolist()
END = dataset.event.short_name  # route labels: "A → B → <venue>"
DATA_LABEL = "Default schools" if _src is None else _src["name"]

# Short names for the sidebar choice. Reports and tables keep MODEL_LABELS.
GOAL_LABELS = {"fixed": "Fixed price per bus", "distance": "Price per mile", "time": "Price per driving hour"}
GOAL_CAPTIONS = {
    "fixed": "Exact search; can prove the cheapest plan",
    "distance": "Route search on straight-line miles",
    "time": "Route search on road driving times",
}

# --------------------------------------------------------------------------- #
# Header (filled once the plan's status is known) and sidebar controls
# --------------------------------------------------------------------------- #

header = st.container()

def routing_status() -> tuple[str, str]:
    """(kind, text): where driving times for the current schools and venue will come from."""
    n = len(points_key())
    if n > OSRM_MAX_POINTS:
        return "warning", (f"{n} points (schools + venue) exceed the OSRM limit of {OSRM_MAX_POINTS}. The "
                           "driving-time goal will use the fixed-speed estimate, labelled as an estimate.")
    try:
        get_road_matrix(points_key(), False)
        return "success", "Road times (OSRM) are cached for these schools and this venue."
    except RoutingServiceError:
        return "info", ("Road times are not cached for these inputs. The driving-time goal fetches them "
                        "from OSRM on its first run; if OSRM is unreachable it uses a labelled estimate.")


with st.sidebar:
    st.markdown('<div class="cc-side-title">Inputs</div>', unsafe_allow_html=True)
    if ss.inputs_msg:
        st.success(ss.pop("inputs_msg"), icon=":material/check_circle:")

    # 1. School data -------------------------------------------------------
    n_schools, n_att = len(dataset.schools), int(dataset.schools["attendees"].sum())
    with st.expander(f"School data · {n_schools} schools", icon=":material/school:",
                     expanded=bool(ss.upload_errors)):
        st.file_uploader(
            "Upload school CSV", type=["csv"], key=school_file_key(), on_change=on_school_upload,
            help="Columns: outcode, latitude, longitude, attendants (one row per school, decimal degrees). "
                 "The file is used for this session only; the bundled data is not changed.")
        st.download_button("Download CSV template", SCHOOL_TEMPLATE_CSV, file_name="school_template.csv",
                           mime="text/csv", icon=":material/download:", width="stretch")
        for e in ss.upload_errors:
            st.error(e)
        if ss.upload_errors:
            st.caption(f"Still using: **{DATA_LABEL}**.")
        st.markdown(f"**In use:** {DATA_LABEL}  \n{n_schools} schools · {n_att:,} attendees")
        if _src is not None:
            for w in _src["warnings"]:
                st.warning(w)
            st.button("Use default school data", on_click=use_default_schools, icon=":material/restart_alt:",
                      width="stretch")
        too_big = [m for m in dataset.available_models if dataset.report.capacity_errors.get(m)]
        if too_big:
            st.error("Some schools are larger than every bus, so no plan exists for: "
                     + ", ".join(GOAL_LABELS[m] for m in too_big) + ". See **Data & checks**.")
        st.dataframe(dataset.schools.drop(columns=["name"]).rename(columns={
            "school_id": "Outcode", "latitude": "Lat", "longitude": "Lon", "attendees": "Attendees"}),
            hide_index=True, height=180, width="stretch")

    # 2. Event destination -------------------------------------------------
    venue = dataset.event
    with st.expander(f"Destination · {venue.short_name}", icon=":material/location_on:",
                     expanded=bool(ss.venue_errors) or bool(ss.get("venue_matches"))):
        with st.form("venue_search_form", border=False):
            st.text_input("Search venue or address", key="venue_query", max_chars=200,
                          placeholder="e.g. Ashton Gate Stadium, Bristol",
                          help="Name, street address or postcode. The best match becomes the destination.")
            st.form_submit_button("Find and use", on_click=search_venue, type="primary", width="stretch",
                                  icon=":material/search:")
        st.caption("Search uses OpenStreetMap Nominatim: only the typed text is sent.")
        for e in ss.venue_errors:
            st.error(e)
        matches = ss.get("venue_matches") or []
        if len(matches) > 1:
            st.selectbox("Not the right place? Other matches", range(len(matches)), key="venue_match",
                         format_func=lambda k: matches[k].name, on_change=pick_venue_match)
        st.markdown(f"**In use:** {venue.name}  \n{venue.latitude:.6f}, {venue.longitude:.6f}")
        st.map(pd.DataFrame({"lat": [venue.latitude], "lon": [venue.longitude]}), zoom=12, height=170,
               size=60, color="#D6453D")
        if not same_venue(venue, DEFAULT_EVENT):
            st.button("Reset to Principality Stadium", on_click=reset_venue, icon=":material/restart_alt:",
                      width="stretch")
        with st.popover("Enter coordinates by hand", width="stretch", icon=":material/edit_location_alt:"):
            with st.form("venue_form", border=False):
                st.text_input("Venue name", key="venue_name", max_chars=120)
                c1, c2 = st.columns(2)
                c1.text_input("Latitude", key="venue_lat", help="Decimal degrees, -90 to 90 (e.g. 51.478283).")
                c2.text_input("Longitude", key="venue_lon",
                              help="Decimal degrees, -180 to 180 (e.g. -3.182652).")
                st.form_submit_button("Apply destination", on_click=apply_venue, type="primary", width="stretch")
            st.file_uploader("Or upload a one-row venue CSV (name, latitude, longitude)", type=["csv"],
                             key="venue_file", on_change=on_venue_file)
        far = far_schools(dataset.schools, venue)
        if far:
            st.warning(f"{len(far)} school(s) are over {FAR_FROM_VENUE_MILES:g} straight-line miles from the "
                       "venue (" + ", ".join(far[:5]) + ("…" if len(far) > 5 else "") + "). Check the "
                       "coordinates, for example latitude and longitude swapped.")
    kind, text = routing_status()
    getattr(st, kind)(text, icon=":material/route:")

    st.markdown('<div class="cc-side-title">Plan settings</div>', unsafe_allow_html=True)
    models = [m for m in COST_MODELS if m in dataset.available_models]
    if not models:
        st.error("No valid cost data. See the Data & checks tab.")
        st.stop()
    model = st.radio(
        "Optimisation goal", models, format_func=GOAL_LABELS.get, key="model_radio",
        captions=[GOAL_CAPTIONS[m] for m in models],
        help="Each goal minimises the total transport cost; they differ in how a bus is priced.",
    )
    run_area = st.container()  # run options and button, drawn above the advanced settings
    default_limit = int(DEFAULT_TIME_LIMITS[model])
    allow_online = True
    time_source = "osrm"
    with st.expander("Advanced settings", icon=":material/tune:"):
        # One slider per model, so each model keeps its own default (fixed 20 s, routing 300 s).
        time_limit = st.slider(
            "Search time limit (seconds)", MIN_TIME_LIMIT, MAX_TIME_LIMIT, default_limit, step=5,
            key=f"time_limit_{model}",
            help=f"How long the optimiser may search. Default for this goal: {default_limit} s.",
        )
        if model == "fixed":
            st.caption(f"Limit sent to the solver: {time_limit} s. The search stops early once the "
                       "cheapest plan is proven.")
        else:
            st.caption(f"Limit sent to the solver: {time_limit} s. Route search uses the full limit; "
                       "more time can help but does not guarantee a cheaper plan.")
        if model == "time":
            time_source = st.selectbox(
                "Travel-time source", list(TIME_SOURCES), format_func=TIME_SOURCES.get, key="time_source_choice",
                help="OSRM: road-network estimates (online or cached). ML estimate: experimental model learned "
                     "from cached OSRM estimates, used only if it beat the fixed-speed estimate on held-out "
                     "schools. Fixed-speed: straight-line miles x 1.3 / 35 mph. None is an observed journey time.",
            )
            if time_source == "ml" and custom_inputs():
                st.warning(ML_REGION_NOTE)
            if time_source == "osrm":
                allow_online = st.checkbox(
                    "Fetch road times online (OSRM)", value=True, key="allow_online",
                    help="Public OSRM table service, cached locally. If unchecked, no network request is "
                         "made: cached road times are used if present, otherwise a labelled offline estimate.",
                )

    # Fingerprint of everything the plan depends on. Widgets that do not change
    # it (tabs, expanders, map filters, assistant...) never start a run.
    key = run_key(model, ss.overrides, current_constraints(), allow_online, time_source)
    manual_request = ss.pop("run_requested", False)  # set by a Run button callback
    assistant_request = ss.pop("assistant_run", False)  # set only by an approved assistant proposal
    hit = ss.result_cache.get(key)
    if hit is not None and ss.current is not hit and not (manual_request or assistant_request):
        # These exact inputs were optimised before: show that result, no solver run.
        make_current(hit)
        ss.reused = key
    fresh = freshness(ss.current, key, time_limit)

    with run_area:
        warm = False
        if ss.last_valid is not None:
            warm = st.checkbox(
                "Use previous plan as a starting point", value=True,
                help="The last valid plan is adjusted for your changes and the search starts from it, "
                     "so schools move only where your edits require it.",
            )
        auto = st.checkbox(
            "Update automatically when inputs change", value=AUTO_RUN_DEFAULT[model],
            key=f"auto_run_{model}",
            help="Runs once per new set of inputs, after your first run. Off by default for the "
                 "route-search goals, which use the full time limit.",
        )
        st.button("Run optimiser", type="primary", width="stretch", key="run_optimiser", on_click=request_run,
                  icon=":material/play_arrow:",
                  help="Runs now with these settings, even if the plan is up to date.")
        trigger = "manual" if manual_request else ("assistant" if assistant_request else None)
        if trigger is None and should_auto_run(fresh, auto, key, ss.last_attempt, bool(ss.attendance_errors),
                                               ss.solver_runs > 0):
            trigger = "auto"

        if ss.attendance_errors:
            st.error("Fix the attendance errors in **Adjust plan** first.")
        elif trigger is None:
            if fresh.state == CURRENT:
                st.success("Plan is up to date.", icon=":material/check_circle:")
            elif fresh.state == SETTINGS_CHANGED:
                st.info(f"This plan used a {fresh.run_limit:g} s limit; the setting is now "
                        f"{fresh.want_limit:g} s. Run again to apply it.")
            elif fresh.state == STALE:
                st.warning("Inputs changed (" + ", ".join(fresh.changes) + "). Update the plan.",
                           icon=":material/schedule:")
        if auto and model != "fixed":
            st.caption(f"Each automatic run can take up to {time_limit} s.")
    st.caption(f"Costs are in {CURRENCY_LABEL}; the data states no currency.")
    st.caption("Campus Cruise · by **Dhanush Kalapati**, Cardiff University")

# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #


def describe_entry(e: dict) -> str:
    return f"{MODEL_LABELS[e['model']]}, run #{e['run_no']} finished at {e['finished_at']}"


def log_run(trigger: str, sol=None, base=None, outcome: str = "", tm=None):
    """Audit trail row. Holds run settings and figures only: no request text, no personal data."""
    ss.run_log.append({
        "Run": ss.solver_runs,
        "Time": time.strftime("%H:%M:%S"),
        "Trigger": trigger,
        "Model": model,
        "Schools": DATA_LABEL,
        "Venue": END,
        "Attendance edits": len(ss.overrides),
        "Rules": len(current_constraints().describe()),
        "Limit (s)": float(time_limit),
        "Travel-time data": (tm.source if tm else "-") if model == "time" else "not used",
        "Status": sol.status_text() if sol else "error",
        "Runtime (s)": round(sol.runtime_s, 1) if sol else None,
        "Cost": round(sol.total_cost, 2) if sol and sol.has_plan else None,
        "Baseline": round(base.total_cost, 2) if base is not None and base.has_plan else None,
        "Buses": len(sol.routes) if sol and sol.has_plan else None,
        "Validation": ("passed" if sol.has_plan else ("failed: " + "; ".join(sol.violations) if sol.violations
                                                      else "no plan")) if sol else "not run",
        "Outcome": outcome,
    })


def assistant_note(kind: str, text: str):
    """A message from the app (not the language model) in the chat history."""
    ss.chat_history.append({"role": "assistant", "kind": kind, "text": text, "summary": True})


def record_run(result, working, tm, warm_from, trigger):
    """Store a finished run. Called straight after the solver returns, before any
    other Streamlit call, so a rerun requested meanwhile cannot lose the result."""
    sol = result.solution
    approved = ss.pop("approved_proposal", None) if trigger == "assistant" else None
    entry = {
        "result": result, "model": model, "overrides": dict(ss.overrides),
        "constraints": current_constraints(), "dataset": working,
        "time_source": tm.source if tm else None, "time_notes": tm.notes if tm else [],
        "key": key, "trigger": trigger, "run_no": ss.solver_runs,
        "finished_at": time.strftime("%H:%M:%S"),
        "settings": {"time_limit": float(time_limit), "warm_start": warm_from is not None,
                     "allow_online": allow_online if model == "time" else None},
    }
    cur = ss.current
    if (not sol.has_plan and sol.status != INFEASIBLE and cur is not None and cur["key"] == key
            and cur["result"].solution.has_plan):
        # Timeout or failed check on inputs that already have a valid plan: keep that plan.
        ss.run_error = {"key": key, "message": (
            f"The new run found no usable plan ({sol.status_text().lower()}). The existing plan for these "
            "inputs is kept and shown below."),
            "detail": f"Status {sol.status}; searched {sol.runtime_s:.1f} s of a {sol.time_limit_s:g} s limit."}
        log_run(trigger, sol, result.baseline, "no plan; previous plan for these inputs kept", tm)
        if trigger == "assistant":
            assistant_note("error", ss.run_error["message"])
        return
    ss.run_error = None
    make_current(entry)
    # Cache only conclusive results. A timeout may succeed next time, and an
    # offline estimate used because OSRM failed should be retried later.
    fallback = (model == "time" and time_source == "osrm" and allow_online and tm is not None
                and tm.source == SOURCE_APPROX)
    if (sol.has_plan or sol.status == INFEASIBLE) and not fallback:
        cache_put(ss.result_cache, key, entry)
    if sol.has_plan:
        ss.last_valid = entry
    log_run(trigger, sol, result.baseline, "shown as current plan", tm)
    if trigger == "assistant":
        totals = None
        if sol.has_plan:
            hours, hsrc = (tm.hours, tm.source) if tm is not None else display_hours()
            totals = plan_totals(sol, working, result.matrix if model != "time"
                                 else straight_line_matrix(points_of(working)), hours)
        else:
            hsrc = None
        assistant_note(*run_summary(entry, approved, totals, hsrc))


if trigger:
    ss.last_attempt = key  # an automatic run never repeats for the same inputs
    working = None
    if ss.attendance_errors:
        st.error("Not optimised. Invalid attendance: " + " ".join(ss.attendance_errors)
                 + " The results below are from the last successful run.")
    else:
        try:
            working = dataset.with_attendance(ss.overrides)
        except ValueError as exc:
            st.error(f"Attendance edit rejected: {exc}")
    if working is not None:
        shown = ss.last_valid
        st.info(f"**Optimising:** {GOAL_LABELS[model].lower()}, up to {time_limit} s ({trigger} run). "
                + ("The current plan stays on screen until the new one has passed every check."
                   if shown else ""), icon=":material/hourglass_top:")
        try:
            tm = None
            if model == "time":
                # Coordinates do not change with attendance, so one matrix serves every run.
                tm = get_time_matrix(points_key(), allow_online, time_source)  # always labelled
            warm_from = shown["result"].solution if (warm and shown is not None) else None
            ss.solver_runs += 1
            with st.spinner("Searching for the best plan..."):
                result = optimise(working, model, current_constraints(), time_limit, tm, warm_start=warm_from)
                record_run(result, working, tm, warm_from, trigger)
        except Exception as exc:  # solver or data error: keep every stored result
            ss.run_error = {"key": key, "message": (
                f"The optimiser stopped with an error ({type(exc).__name__}: {exc}). No result was replaced; "
                "the page shows the last result.")}
            log_run(trigger, outcome=f"error: {type(exc).__name__}")
            if trigger == "assistant":
                ss.pop("approved_proposal", None)
                assistant_note("error", "The approved run did not complete. " + ss.run_error["message"])
        st.rerun()

cur = ss.current
fresh = freshness(cur, key, time_limit)


def state_badge() -> str:
    """One status for the whole page: is the plan on screen current, out of date or missing?"""
    if cur is not None and not cur["result"].solution.has_plan:
        return ":red-badge[:material/error: No valid plan]"
    return {
        CURRENT: ":green-badge[:material/check_circle: Plan up to date]",
        SETTINGS_CHANGED: ":blue-badge[:material/tune: Settings changed]",
        STALE: ":orange-badge[:material/schedule: Out of date]",
        NONE: ":gray-badge[No plan yet]",
    }[fresh.state]


with header:
    brand, status = st.columns([4, 1], vertical_alignment="center")
    brand.markdown(
        '<div class="cc-brand"><div class="cc-logo">🚌</div><div><div class="cc-title">Campus Cruise</div>'
        f'<div class="cc-sub">School transport planning · {dataset.event.name}</div></div></div>',
        unsafe_allow_html=True,
    )
    with status.container(horizontal_alignment="right"):
        st.markdown(state_badge())

if ss.run_error is not None and ss.run_error["key"] == key:
    st.error(ss.run_error["message"], icon=":material/error:")
    if ss.run_error.get("detail"):
        with st.expander("Error details"):
            st.code(ss.run_error["detail"], language=None)
if ss.reused is not None:
    if ss.reused == key and cur is not None:
        st.info(f"Reused the stored result for these inputs ({describe_entry(cur)}). No new search was needed.",
                icon=":material/history:")
    ss.reused = None
if fresh.state == STALE:
    text_col, button_col = st.columns([5, 1], vertical_alignment="center")
    text_col.warning(
        "**Optimisation required.** The results below do not include your changes to: "
        + ", ".join(fresh.changes) + ". Update the plan before using it for booking.",
        icon=":material/schedule:")
    button_col.button("Update plan", type="primary", key="run_optimiser_main", on_click=request_run,
                      disabled=bool(ss.attendance_errors), width="stretch")
    if auto and ss.last_attempt == key:
        st.caption("An automatic run was already tried for these inputs. Press **Update plan** to try again.")


def stale_note():
    """Shown on every tab that displays plan figures while the plan is out of date."""
    if fresh.state == STALE:
        st.warning("Out of date: this is the last run, before your changes to "
                   + ", ".join(fresh.changes) + ".", icon=":material/schedule:")


def run_caption(entry: dict) -> str:
    """When and how a run was made. The runtime lives here, not in a headline card."""
    sol = entry["result"].solution
    return (f"From run #{entry['run_no']} at {entry['finished_at']} ({entry['trigger']}) · finished in "
            f"{sol.runtime_s:.1f} s, time limit {entry['settings']['time_limit']:g} s")


@st.cache_data(max_entries=16)
def miles_for(pts: tuple) -> "np.ndarray":
    """Straight-line miles between schools (dataset order) and the venue (last point of ``pts``)."""
    return straight_line_matrix([tuple(p) for p in pts])


def own_matrix(entry: dict):
    """The matrix the run was costed with (hours for the time model, miles otherwise)."""
    return entry["result"].matrix if entry["model"] == "time" else miles_for(points_key())


def entry_hours(entry: dict) -> tuple:
    """Driving hours for a run: its own matrix for the time model, else cached OSRM (no request)."""
    if entry["model"] == "time":
        return entry["result"].matrix, entry["result"].solution.matrix_source
    return display_hours()


def entry_distance(entry: dict):
    """Bus-miles and driving hours of a validated run against its baseline, or None."""
    res = entry["result"]
    if not (res.solution.has_plan and res.baseline.has_plan):
        return None
    hours, hsrc = entry_hours(entry)
    try:
        return distance_impact(res.solution, res.baseline, entry["dataset"], miles_for(points_key()), hours, hsrc)
    except ValueError:
        return None


def session_summary() -> pd.DataFrame:
    """All models run this session, one row each, from stored validated results."""
    hours, hsrc = display_hours()
    rows = [summary_row(e, matches_current_inputs(e), miles_for(points_key()), hours, hsrc)
            for m, e in ss.by_model.items()]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# AI Assistant: the floating chat at the bottom-right of every section
# --------------------------------------------------------------------------- #

TYPE_LABELS = {
    "explain": "Explanation", "compare": "Comparison", "run_optimiser": "Run optimiser",
    "attendance_change": "Attendance change", "sharing_rule": "Sharing rule",
    "grouping_preference": "Soft preference (not applied)", "clarify": "Needs clarification",
    "unsupported": "Not supported",
}
assert set(TYPE_LABELS) == set(REQUEST_TYPES)


def assistant_inputs():
    """Current (edited) dataset and the context snapshot for the model."""
    working = dataset.with_attendance(ss.overrides)
    runs = {m: {"solution": e["result"].solution, "baseline": e["result"].baseline,
                "current_inputs": matches_current_inputs(e)} for m, e in ss.by_model.items()}
    plan = cur["result"].solution if cur is not None and matches_current_inputs(cur) else None
    return working, build_context(working, current_constraints(), runs, model, plan)


def get_provider():
    try:
        secrets = dict(st.secrets) if hasattr(st, "secrets") else {}
    except Exception:  # no secrets.toml: that is fine
        secrets = {}
    try:
        return provider_from_config(secrets), None
    except AssistantUnavailable as exc:
        return None, str(exc)


def ask_assistant(text: str, provider) -> None:
    """One request from the chat. ``assist`` only proposes and
    validates: nothing changes and the optimiser does not run."""
    working, context = assistant_inputs()
    p = assist(text, provider, working, current_constraints(), context, model)
    ss.assistant_proposal = p  # a newer request replaces an undecided proposal
    if not p.needs_approval:
        ss.assistant_log.append((p.request, p.request_type,
                                 "rejected by validation" if p.errors else "answered, nothing changed"))
    ss.chat_history.append({"role": "user", "text": p.request or text})
    ss.chat_history.append({"role": "assistant", "proposal": p})


def decide(callback):
    """Approve or Reject. The outcome is posted to the chat."""
    st.session_state.assistant_msg = None
    callback()
    msg = st.session_state.pop("assistant_msg", None)
    if msg:
        st.session_state.chat_history.append({"role": "assistant", "kind": msg[0], "text": msg[1]})


def queue_chat_input():
    st.session_state.chat_queued = st.session_state.get("chat_input")


def clear_chat():
    ss = st.session_state
    p = ss.assistant_proposal
    if p is not None and p.needs_approval:
        ss.assistant_log.append((p.request, p.request_type, "discarded when the chat was cleared"))
    ss.assistant_proposal = None
    ss.chat_history = []


def show_proposal(p, prefix: str = "assistant", actions: bool = True, show_request: bool = True):
    """Render a proposal. Approve/Reject buttons only when ``actions`` (the pending proposal)."""
    if show_request:
        st.markdown(f"**Request:** {p.request}")
    st.markdown(f"**Understood as:** {TYPE_LABELS.get(p.request_type, p.request_type)}"
                + (f" — {p.summary}" if p.summary else ""))
    if p.answer and not p.errors:
        st.markdown(p.answer)
        st.caption("Generated by a language model from this session's plan data. Figures were checked "
                   "against the plan data; the solver status is authoritative.")
    if p.clarification:
        st.info(p.clarification)
    for note in p.notes:
        st.info(note)
    if p.errors:
        st.error("**Not applied.** " + " ".join(p.errors))
    if not p.needs_approval:
        return
    with st.container(border=True):
        st.markdown("**Proposed changes — nothing has changed yet**" if actions
                    else "**Proposed changes** (no longer pending; the outcome is shown below)")
        rows = [{"Change": "Attendance", "Detail": f"{s}: {p.current_attendance[s]} → {v}"}
                for s, v in p.attendance.items()]
        for kind, schools in p.rules:
            text = {"must_share": "{} and {} must share a bus (hard rule)",
                    "cannot_share": "{} and {} must not share a bus (hard rule)",
                    "solo": "{} travels on its own bus (hard rule)"}[kind].format(*schools)
            rows.append({"Change": "Sharing rule", "Detail": text})
        limit = ss.get(f"time_limit_{p.cost_model}", int(DEFAULT_TIME_LIMITS[p.cost_model]))
        rows.append({"Change": "Then run", "Detail": f"{MODEL_LABELS[p.cost_model]}, time limit {limit} s "
                     "(change it in the sidebar)"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        if not actions:
            return
        st.caption("On approval the app re-checks the proposal, applies it, runs the OR-Tools optimiser "
                   "and the independent validator. Hard rules can raise the cost.")
        a, r = st.columns(2)
        a.button("Approve and run", type="primary", key=f"{prefix}_approve", on_click=decide,
                 args=(assistant_approve,), width="stretch")
        r.button("Reject", key=f"{prefix}_reject", on_click=decide, args=(assistant_reject,), width="stretch")


# Floating chat ---------------------------------------------------------------
# A native st.popover inside a keyed container. Streamlit gives the container
# the CSS class "st-key-cc_chat_fab"; the style block pins it bottom-right.
# Opening and closing a popover happens in the browser only (no script rerun),
# so it never touches the dashboard state or the optimiser.

def chat_turns(history: list[dict]) -> list[list[dict]]:
    """Group the history into turns (a request and the replies that follow it),
    newest turn first. Messages inside a turn keep their order."""
    turns: list[list[dict]] = []
    for msg in history:
        if msg["role"] == "user" or not turns:
            turns.append([])
        turns[-1].append(msg)
    return turns[::-1]


def render_chat(provider, unavailable: str | None):
    with st.container(key="cc_chat_panel"):
        # Handle a queued request first, so every widget below reflects it in this run.
        text = ss.pop("chat_queued", None)
        if text and provider is not None:
            with st.spinner("Asking the assistant..."):
                ask_assistant(text, provider)
        head, clear = st.columns([3, 1], vertical_alignment="center")
        head.markdown("**Transport planning assistant**")
        clear.button("Clear", key="chat_clear", on_click=clear_chat, disabled=not ss.chat_history,
                     help="Clears the conversation. A proposal waiting for approval is discarded.")
        if provider is None:
            st.info(f"**Assistant not configured.** {unavailable} Every other part of Campus Cruise works "
                    "without it.")
        else:
            st.caption(f"{provider.label}. Answers use this session's plan data. Changes to attendance, "
                       "rules or the plan need your approval.")

        # One typing bar at the top; answers appear below it, newest first.
        st.chat_input("Ask about the plan...", key="chat_input", on_submit=queue_chat_input,
                      max_chars=MAX_REQUEST_CHARS, disabled=provider is None)

        with st.container(key="cc_chat_log"):
            if not ss.chat_history:
                st.caption("For example: \"Explain the current plan\" or \"NP11 now brings 15 people.\" "
                           "Nothing changes until you approve it.")
            for k, turn in enumerate(chat_turns(ss.chat_history)):
                if k:
                    st.divider()
                for msg in turn:
                    with st.chat_message(msg["role"]):
                        if "proposal" in msg:
                            p = msg["proposal"]
                            show_proposal(p, prefix="chat", actions=p is ss.assistant_proposal,
                                          show_request=False)
                        elif msg.get("kind"):
                            getattr(st, msg["kind"])(msg["text"])
                        else:
                            st.markdown(msg["text"])

        with st.expander("What is sent to the language model"):
            st.markdown(
                "- Your request text.\n"
                "- School outcodes (for example *CF15*) with head counts and, if a plan exists, each school's bus.\n"
                "- Bus sizes and prices, organiser rules, and the figures and status of each plan you ran.\n"
                "- **Not sent:** coordinates, names, or the API key in any message. No student-level data "
                "exists in the app.\n\n"
                + (f"Provider: {provider.label}. The request leaves this computer."
                   if provider is not None else
                   "No provider configured. Set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` to enable the assistant.")
            )
        if ss.assistant_log:
            with st.expander(f"Assistant history this session ({len(ss.assistant_log)})"):
                st.dataframe(pd.DataFrame(ss.assistant_log, columns=["Request", "Type", "Outcome"]),
                             hide_index=True, width="stretch")


provider, unavailable = get_provider()
with st.container(key="cc_chat_fab"):
    # No hover tooltip: it would cover the open panel. Opening or closing it changes nothing.
    with st.popover("Assistant", icon=":material/forum:", type="primary"):
        render_chat(provider, unavailable)


tabs = st.tabs(["Overview", "Routes", "Adjust plan", "Compare", "Scenarios", "Responsible AI", "Data & checks",
               "Reports"])

HARD_RULES = (
    "- Each school with attendees rides exactly one bus. Schools are never split.\n"
    "- A bus never carries more people than its seats.\n"
    "- Only the bus sizes in the price file are used.\n"
    "- Organiser rules (must share, must not share, own bus) always hold. If they cannot, "
    "the app says why instead of ignoring them.\n"
    "- A bus starts at its first pickup and ends at the event venue."
)
INDEPENDENT_CHECK = (
    "Separate code (`src/validation.py`) re-checks each plan without trusting the solver:\n"
    "- every school with attendees is on exactly one bus, and no school without attendees is;\n"
    "- attendees are recounted, and no bus is over capacity;\n"
    "- every route length and cost is recalculated from the price file and route legs;\n"
    "- every organiser rule is re-checked.\n\n"
    "A plan that fails is labelled *Plan failed validation* and cannot be shown or exported."
)


def plan_meaning(sol) -> str:
    """One plain sentence on what the solver status means for this plan."""
    if sol.proven_optimal:
        return "Optimal solution found: no valid plan costs less under this pricing."
    return (f"Feasible plan found. The {sol.time_limit_s:g} s search limit was reached without a proof "
            "that it is the cheapest; a longer search may find a cheaper plan.")


# --------------------------------------------------------------------------- #
# Overview
# --------------------------------------------------------------------------- #

with tabs[0]:
    if cur is None:
        with st.container(border=True):
            st.markdown("**No plan yet**")
            st.caption("Choose an optimisation goal in the sidebar, then generate a plan. "
                       "The fixed-price goal takes about 20 seconds.")
            if not ss.attendance_errors:
                st.button("Generate plan", type="primary", key="run_optimiser_main", on_click=request_run,
                          icon=":material/play_arrow:")
        section("Event data")
        c = st.columns(3)
        c[0].metric("Schools", len(dataset.schools), border=True)
        source_total = int(dataset.schools["attendees"].sum())
        edited_total = int(dataset.with_attendance(ss.overrides).schools["attendees"].sum())
        c[1].metric("Attendees (with your edits)" if ss.overrides else "Attendees", edited_total,
                    f"{edited_total - source_total:+d} vs source data" if ss.overrides else None, border=True)
        c[2].metric("Bus sizes", ", ".join(str(s) for s in dataset.bus_sizes(models[0])), border=True)
    else:
        res = cur["result"]
        sol, base = res.solution, res.baseline
        st.markdown(f"**{MODEL_LABELS[cur['model']]}** &nbsp; {status_badge(sol)}")
        st.caption(run_caption(cur))
        if not sol.has_plan:
            st.error("No valid plan. " + " ".join(sol.notes + sol.violations))
            lv = ss.last_valid
            if lv is not None and lv is not cur:
                st.info(f"Last valid plan kept for comparison, not shown as current: {describe_entry(lv)}, "
                        f"cost {money(lv['result'].solution.total_cost)}. It used different inputs "
                        "(see the Compare tab).")
        else:
            im = impact_metrics(base, sol)
            pct = "n/a" if im.saving_percent is None else f"{im.saving_percent:.1f}%"
            c = st.columns(4)
            c[0].metric("Attendees", sol.total_attendees, f"{im.schools_served} schools", delta_color="off",
                        delta_arrow="off", border=True,
                        help="Everyone who travels, from the data plus your edits.")
            c[1].metric("Buses", len(sol.routes), f"{len(sol.routes) - len(base.routes):+d} vs baseline",
                        delta_color="inverse", border=True)
            c[2].metric("Total cost", money(sol.total_cost), f"baseline {money(base.total_cost)}",
                        delta_color="off", delta_arrow="off", border=True, help=f"In {CURRENCY_LABEL}.")
            c[3].metric("Savings", money(im.saving), pct, border=True,
                        help="Compared with every school renting its own cheapest bus, same pricing.")

            left, right = st.columns([3, 2], gap="large")
            with left:
                section("Baseline vs optimised")
                rows = [
                    ("Total cost", money(im.baseline_cost), money(im.optimised_cost), f"{-im.saving:+,.2f}"),
                    ("Buses", str(im.baseline_buses), str(im.optimised_buses), f"{-im.buses_removed:+d}"),
                    ("Seat utilisation", f"{im.baseline_utilisation * 100:.1f}%",
                     f"{im.optimised_utilisation * 100:.1f}%", ""),
                    ("Cost per attendee", fmt_money(im.cost_per_student_before) or "n/a",
                     fmt_money(im.cost_per_student_after) or "n/a", ""),
                ]
                di = entry_distance(cur)
                if di is not None:
                    mp = "" if di.miles_saved_percent is None else f" ({-di.miles_saved_percent:+.1f}%)"
                    rows.append(("Bus-miles (straight line)", f"{di.baseline_miles:,.1f}", f"{di.plan_miles:,.1f}",
                                 f"{-di.miles_saved:+,.1f}{mp}"))
                    if di.hours_saved is not None:
                        hp = "" if di.hours_saved_percent is None else f" ({-di.hours_saved_percent:+.1f}%)"
                        rows.append(("Bus driving hours", f"{di.baseline_hours:,.1f}", f"{di.plan_hours:,.1f}",
                                     f"{-di.hours_saved:+,.1f}{hp}"))
                st.dataframe(pd.DataFrame(rows, columns=["Measure", "One bus per school", "Optimised plan",
                                                         "Change"]), hide_index=True, width="stretch")
                if di is not None:
                    st.caption("Bus-miles and hours are summed over all buses, first pickup to the event venue. "
                               "Miles are straight-line, not road mileage"
                               + (f"; hours from {di.hours_source}" if di.hours_saved is not None else
                                  "; no travel-time data cached, so hours are not shown")
                               + ". Fuel and emissions are not estimated.")
            with right:
                section("What this plan means")
                mix = pd.Series([r.bus_size for r in sol.routes]).value_counts().sort_index()
                lines = [
                    ("**Cheapest possible plan** for this pricing." if sol.proven_optimal else
                     f"**Best plan found** in the {sol.time_limit_s:g} s search. A longer search may find "
                     "a cheaper one."),
                    "**Checked independently:** every school on one bus, no bus over capacity, every cost "
                    "recalculated.",
                    "**Bus mix:** " + ", ".join(f"{n} × {s}-seat" for s, n in mix.items()) + ".",
                ]
                if cur["model"] == "fixed":
                    lines.append("Priced per bus, so distance does not change the cost.")
                if sol.excluded_schools:
                    lines.append(f"No bus needed (zero attendees): {', '.join(sol.excluded_schools)}.")
                st.markdown("\n".join(f"- {x}" for x in lines))

            with st.expander("Why this plan? Explanation from the plan data", icon=":material/lightbulb:"):
                st.caption("Facts are computed from this plan, its baseline and the input data. Lines marked "
                           "*Interpretation* are readings of those facts, not proven causes: the solver does "
                           "not report why it chose a plan.")
                for kind, text in explain_plan(sol, base, cur["dataset"], cur["constraints"], own_matrix(cur)):
                    st.markdown(f"- {'*Interpretation:* ' if kind == 'interpretation' else ''}{text}")
            with st.expander("How this plan was created", icon=":material/settings:"):
                st.markdown(f"- **Method:** {sol.method}")
                st.markdown(f"- **Status:** {sol.status_text()}. {plan_meaning(sol)}")
                st.markdown(f"- **Search time:** {sol.runtime_s:.1f} s of a {sol.time_limit_s:g} s limit; "
                            f"started from the previous plan: {'yes' if cur['settings']['warm_start'] else 'no'}.")
                if cur["model"] != "fixed":
                    st.markdown(f"- **Route data:** {sol.matrix_source}")
                for note in sol.notes + cur["time_notes"]:
                    st.markdown(f"- {note}")
            with st.expander("Validation and constraints", icon=":material/rule:"):
                st.markdown("**Hard rules, never relaxed**")
                st.markdown(HARD_RULES)
                rules = cur["constraints"].describe()
                st.markdown("**Organiser rules in this run:** " + ("; ".join(rules) if rules else "none"))
                st.markdown("**Independent check of every plan**")
                st.markdown(INDEPENDENT_CHECK)

# --------------------------------------------------------------------------- #
# Routes: map, buses and schools
# --------------------------------------------------------------------------- #


def load_road_paths(sol, ds) -> tuple[dict, list[str]]:
    """Road path per bus from cache or OSRM (1 request/s). Missing paths stay straight lines."""
    store = ss.setdefault("road_paths", {})
    info = ds.schools.set_index("school_id")
    e = ds.event
    paths, errors = {}, []
    todo = []
    for r in sol.routes:
        pts = tuple((round(float(info.loc[x, "latitude"]), 6), round(float(info.loc[x, "longitude"]), 6))
                    for x in r.stops) + ((e.latitude, e.longitude),)
        todo.append((r.bus_id, pts))
    missing = [t for t in todo if t[1] not in store]
    bar = st.progress(0.0, text="Fetching road paths from OSRM...") if missing else None
    for k, (bus_id, pts) in enumerate(missing, start=1):
        try:
            store[pts] = road_path(list(pts))
        except RoutingServiceError as exc:
            store[pts] = str(exc)
        bar.progress(k / len(missing), text=f"Fetching road paths from OSRM... {k}/{len(missing)}")
    if bar:
        bar.empty()
    for bus_id, pts in todo:
        value = store.get(pts)
        if isinstance(value, str):
            errors.append(f"{bus_id}: {value}")
            store.pop(pts, None)  # retry next time
        else:
            paths[bus_id] = value
    return paths, errors


def bus_color(sol, bus_id: str) -> str:
    """A bus keeps its colour (its position in the full plan) when the map is filtered."""
    k = next(i for i, r in enumerate(sol.routes) if r.bus_id == bus_id)
    return PALETTE[k % len(PALETTE)]


MAP_HEIGHT = 520


def draw_map(sol, ds, paths=None, show=None, highlight=None):
    """``show``: routes to draw (default all). ``highlight``: a school id to ring in black.
    The view is computed from the points drawn (map_view), not with Leaflet's fitBounds,
    which zooms to its maximum when the map is drawn inside a hidden tab."""
    paths = paths or {}
    routes = show if show is not None else sol.routes
    e = ds.event
    info = ds.schools.set_index("school_id")
    points = [(e.latitude, e.longitude)] + [
        (float(info.loc[s, "latitude"]), float(info.loc[s, "longitude"])) for r in routes for s in r.stops]
    center, zoom = map_view(points, 640, MAP_HEIGHT)
    fmap = folium.Map(location=center, zoom_start=zoom, tiles="OpenStreetMap", scrollWheelZoom=False,
                      control_scale=True)
    numbered = len(routes) <= 3  # pickup numbers only when few buses are shown, so they stay readable
    for r in routes:
        color = bus_color(sol, r.bus_id)
        coords = [(float(info.loc[s, "latitude"]), float(info.loc[s, "longitude"])) for s in r.stops]
        if highlight in r.stops:
            folium.CircleMarker(location=coords[r.stops.index(highlight)], radius=16, color="#1F2933",
                                weight=3, fill=False, tooltip=f"Selected: {highlight}").add_to(fmap)
        road = paths.get(r.bus_id)
        if road is not None:
            folium.PolyLine(road.coords, color=color, weight=4, opacity=.85,
                            tooltip=f"{r.bus_id}: road path (OSRM), {road.miles:.1f} mi, "
                                    f"{road.hours * 60:.0f} min").add_to(fmap)
        else:
            folium.PolyLine(coords + [(e.latitude, e.longitude)], color=color, weight=2.5, opacity=.8,
                            dash_array="6 6",
                            tooltip=f"{r.bus_id}: straight line between stops (not a road path)").add_to(fmap)
        for n, s in enumerate(r.stops, start=1):
            a = int(info.loc[s, "attendees"])
            folium.CircleMarker(
                location=coords[n - 1], radius=4 + math.sqrt(a), color="#FFFFFF", fill=True,
                fill_color=color, fill_opacity=.9, weight=1.5,
                tooltip=f"{s}: {a} attendees · {r.bus_id}, pickup {n} of {len(r.stops)}",
                popup=folium.Popup(
                    f"<b>{s}</b><br>{a} attendees<br>{r.bus_id} ({r.bus_size}-seat), pickup {n} of "
                    f"{len(r.stops)}<br>Route: {' → '.join(r.stops)} → {END}", max_width=260),
            ).add_to(fmap)
            if numbered:
                folium.Marker(coords[n - 1], icon=folium.DivIcon(
                    icon_size=(18, 18), icon_anchor=(-6, 22),
                    html=f'<div style="background:{color};color:#fff;border-radius:9px;width:18px;height:18px;'
                         f'font:600 11px/18px sans-serif;text-align:center">{n}</div>')).add_to(fmap)
    folium.Marker([e.latitude, e.longitude], tooltip=f"{e.name} (event)",
                  icon=folium.Icon(color="red", icon="star")).add_to(fmap)
    return fmap


def bus_list_html(sol, routes, details: pd.DataFrame) -> str:
    """Compact list of the buses shown, with the same colours as the map."""
    rows = []
    by_bus = details.set_index("Bus")
    for r in routes:
        d = by_bus.loc[r.bus_id]
        minutes = d["Driving minutes"]
        extra = "" if minutes is None or pd.isna(minutes) else f" · {int(minutes)} min"
        rows.append(
            f'<div class="cc-bus"><span class="cc-dot" style="background:{bus_color(sol, r.bus_id)}"></span>'
            f'<div><b>{r.bus_id}</b> <span class="cc-muted">· {r.attendees}/{r.bus_size} seats · '
            f'{d["Straight-line miles"]:.1f} mi{extra}</span><br>'
            f'<span class="cc-muted">{" → ".join(r.stops)} → {END}</span></div></div>')
    return '<div class="cc-buslist">' + "".join(rows) + "</div>"


def clear_map_filters():
    st.session_state.map_buses = []
    st.session_state.map_school = ""


def map_plan_options() -> dict:
    """Plans the map can show: the current plan first, then other models' latest valid runs."""
    opts = {}
    if cur is not None and cur["result"].solution.has_plan:
        opts["current"] = cur
    for m, e in ss.by_model.items():
        if e is not cur and e["result"].solution.has_plan:
            opts[m] = e
    return opts


with tabs[1]:
    plan_opts = map_plan_options()
    if not plan_opts:
        st.info("Generate a plan to see its routes.", icon=":material/map:")
    else:
        pick = "current"
        if len(plan_opts) > 1 or "current" not in plan_opts:
            pick = st.selectbox(
                "Plan shown", list(plan_opts), key="map_plan",
                # Labels depend only on the option key, so they stay stable across runs.
                format_func=lambda k: "Current plan" if k == "current" else f"Other model: {MODEL_LABELS[k]}")
        shown_entry = plan_opts.get(pick, next(iter(plan_opts.values())))
        sol, ds = shown_entry["result"].solution, shown_entry["dataset"]
        st.markdown(f"**Showing:** {describe_entry(shown_entry)}"
                    + ("" if matches_current_inputs(shown_entry) else " (older inputs)")
                    + f" &nbsp; {status_badge(sol)}")
        if shown_entry is cur:
            stale_note()
        else:
            st.info("This is not the current plan. It is shown for comparison; costs from different models "
                    "are not comparable.")

        # Filters. Stale selections (from another plan) are dropped before the widgets are drawn.
        bus_ids = [r.bus_id for r in sol.routes]
        stop_ids = sorted(s for r in sol.routes for s in r.stops)
        if any(b not in bus_ids for b in ss.get("map_buses", [])):
            ss.map_buses = [b for b in ss.map_buses if b in bus_ids]
        if ss.get("map_school") and ss.map_school not in stop_ids:
            ss.map_school = ""
        f1, f2, f3, f4 = st.columns([3, 2, 1, 2], vertical_alignment="bottom")
        chosen_buses = f1.multiselect("Buses", bus_ids, key="map_buses", placeholder="All buses",
                                      help="Pick one or more buses. Empty shows the whole plan.")
        chosen_school = f2.selectbox("Find a school", [""] + stop_ids, key="map_school",
                                     format_func=lambda s: s or "Any school")
        f3.button("Show all", key="map_clear", on_click=clear_map_filters, width="stretch")
        show_roads = f4.toggle(
            "Road paths (OSRM)", value=False,
            help="Fetches real driving paths for the buses shown from the public OSRM server "
                 "(1 request per second, cached on disk). Display only: plan costs do not change.",
        )
        shown = filter_routes(sol, chosen_buses, chosen_school or None)
        hours, hsrc = (shown_entry["result"].matrix, sol.matrix_source) if shown_entry["model"] == "time" \
            else display_hours()
        details = route_details(shown, sol, ds, miles_for(points_key()), hours, hsrc)
        paths = {}
        if show_roads:
            only = type(sol)(model=sol.model, status=sol.status, routes=shown)  # fetch shown buses only
            paths, road_errors = load_road_paths(only, ds)
            if road_errors:
                st.warning("Shown as straight lines (no road path available): " + "; ".join(road_errors))

        map_col, side_col = st.columns([3, 2], gap="medium")
        with map_col:
            if HAS_MAP:
                try:
                    st_folium(draw_map(sol, ds, paths, show=shown, highlight=chosen_school or None),
                              height=MAP_HEIGHT, use_container_width=True, returned_objects=[])
                except Exception as exc:  # never let the map take the page down
                    st.warning(f"Map unavailable ({exc}). The bus list and table show the full plan.")
            else:
                st.warning("Map libraries not installed. The bus list and table show the full plan.")
            st.caption("Marker size = attendees · ★ event venue · solid line = road path (OSRM) · dashed line = "
                       "straight line between stops, not a road route"
                       + (f" · road paths drawn for {len(paths)} of {len(shown)} buses" if show_roads else "")
                       + (" · costs use the model's own measure, not these paths" if show_roads else "") + ".")
        with side_col:
            if chosen_school:
                bus = bus_of_school(sol, chosen_school)
                r = next(x for x in sol.routes if x.bus_id == bus)
                if chosen_buses and bus not in chosen_buses:
                    st.warning(f"{chosen_school} rides {bus}, which is not in the selected buses.")
                st.caption(f"{chosen_school} rides **{bus}** ({r.bus_size}-seat), pickup "
                           f"{r.stops.index(chosen_school) + 1} of {len(r.stops)}: {' → '.join(r.stops)} → {END}.")
            if len(shown) == 1:
                r, d = shown[0], details.iloc[0]
                section(f"{r.bus_id} · {r.bus_size}-seat bus")
                m = st.columns(2)
                m[0].metric("Passengers", f"{r.attendees} / {r.bus_size}", border=True)
                m[1].metric("Seats left", r.empty_seats, border=True)
                m = st.columns(2)
                m[0].metric("Distance", f"{d['Straight-line miles']:.1f} mi", border=True,
                            help="Straight-line miles, first pickup to the event venue.")
                m[1].metric("Driving time", "n/a" if d["Driving minutes"] is None or pd.isna(d["Driving minutes"])
                            else f"{int(d['Driving minutes'])} min", border=True, help=f"Source: {d['Time source']}")
                st.caption(DRIVING_TIME_NOTE)
                st.markdown(f"**Pickup order ({len(r.stops)} schools):** " + " → ".join(r.stops) + f" → {END}")
            else:
                section(f"Buses shown ({len(shown)} of {len(sol.routes)})")
                st.markdown(bus_list_html(sol, shown, details), unsafe_allow_html=True)
                st.caption(DRIVING_TIME_NOTE)

        section("Bus details")
        st.caption("Same buses as the map. Straight-line miles for every model; driving minutes from "
                   + (hsrc or "no travel-time data") + ". " + DRIVING_TIME_NOTE)
        if paths:
            details["Road miles (map)"] = details["Bus"].map(lambda b: round(paths[b].miles, 1) if b in paths else None)
            details["Road minutes (map)"] = details["Bus"].map(lambda b: round(paths[b].hours * 60) if b in paths else None)
        st.dataframe(details, hide_index=True, width="stretch",
                     column_config={"Utilisation %": st.column_config.ProgressColumn(
                         "Utilisation %", min_value=0, max_value=100, format="%.1f%%")})
        if shown_entry["model"] == "fixed":
            st.caption("This model charges per bus, not per mile. The pickup order is the shortest "
                       "straight-line route.")
        with st.expander("School cost shares", icon=":material/payments:"):
            st.caption("Estimate, not an invoice: each bus's cost is split in proportion to attendees.")
            shares = cost_shares(sol, ds)
            shown_ids = {r.bus_id for r in shown}
            shares_shown = shares[shares["Bus"].isin(shown_ids)]
            st.dataframe(shares_shown, hide_index=True, width="stretch",
                         column_config={c: st.column_config.NumberColumn(format="%.2f")
                                        for c in ("Bus cost", "Cost share", "Cost per attendee")})
            shown_cost = sum(r.cost for r in shown)
            st.caption(f"Sum of shares shown: {money(shares_shown['Cost share'].sum())} = cost of the buses "
                       f"shown {money(shown_cost)} (plan total {money(sol.total_cost)}; values rounded).")

# --------------------------------------------------------------------------- #
# Adjust plan: attendance and sharing constraints
# --------------------------------------------------------------------------- #

with tabs[2]:
    if fresh.state == STALE:
        a1, a2 = st.columns([5, 1], vertical_alignment="center")
        a1.info("Your changes are not in the plan yet. **Update plan** applies them.", icon=":material/info:")
        a2.button("Update plan", type="primary", key="run_optimiser_adjust", on_click=request_run,
                  disabled=bool(ss.attendance_errors), width="stretch")
    else:
        st.caption("Edits apply on the next optimiser run: automatically when *Update automatically* is on, "
                   "otherwise when you press **Run optimiser**. The source data files are never changed.")
    col_a, col_c = st.columns([2, 3], gap="large")
    with col_a:
        section("Attendance")
        st.caption("Whole numbers only. 0 means the school needs no bus.")
        base_table = dataset.schools[["school_id", "attendees"]].copy()
        base_table["edited"] = editor_values()  # cell edits live in the editor's own state
        st.data_editor(
            base_table.rename(columns={"school_id": "School", "attendees": "Original", "edited": "Attendees"}),
            hide_index=True, width="stretch", height=380, key="att_editor", on_change=sync_attendance,
            disabled=["School", "Original"],
            column_config={"Attendees": st.column_config.NumberColumn(min_value=0, max_value=500, step=1)},
        )
        for msg in ss.attendance_errors:
            st.error(msg)
        if ss.overrides:
            source_total = int(dataset.schools["attendees"].sum())
            new_total = int(dataset.with_attendance(ss.overrides).schools["attendees"].sum())
            st.markdown("**Edited:** " + ", ".join(f"{k} → {v}" for k, v in ss.overrides.items())
                        + f"  \nTotal attendees: {source_total} in the source data → {new_total} with your edits.")
        if ss.overrides or ss.attendance_errors:
            st.button("Reset attendance to source data", key="reset_attendance", on_click=reset_attendance,
                      icon=":material/undo:")

    with col_c:
        section("Sharing rules")
        st.caption("Hard rules. If they conflict with each other or with bus capacity, the optimiser says "
                   "why instead of ignoring them.")

        def pair_editor(label: str, key: str, help_text: str):
            st.markdown(f"**{label}**")
            c1, c2, c3 = st.columns([2, 2, 1])
            c1.selectbox("School", school_ids, key=f"{key}_a", label_visibility="collapsed")
            c2.selectbox("School", school_ids, index=min(1, len(school_ids) - 1), key=f"{key}_b", label_visibility="collapsed")
            c3.button("Add", key=f"{key}_add", help=help_text, on_click=add_pair, args=(key,), width="stretch")
            if ss.pop(f"{key}_msg", None):
                st.warning("Pick two different schools.")
            for x, y in list(ss[key]):
                r1, r2 = st.columns([4, 1], vertical_alignment="center")
                r1.write(f"{x} & {y}")
                r2.button("Remove", key=f"{key}_rm_{x}_{y}", on_click=remove_pair, args=(key, (x, y)),
                          width="stretch")

        pair_editor("Must share a bus", "must_share", "Both schools ride the same bus.")
        pair_editor("Must not share a bus", "cannot_share", "The two schools ride different buses.")
        st.markdown("**Own bus (no sharing)**")
        ss.setdefault("solo_select", list(ss.solo))
        st.multiselect("Schools that travel alone", school_ids, key="solo_select", on_change=sync_solo,
                       label_visibility="collapsed", placeholder="Choose schools")

# --------------------------------------------------------------------------- #
# Compare
# --------------------------------------------------------------------------- #


def model_row(m: str) -> dict:
    """One row of the all-models table. Only runs on the current inputs show figures."""
    entry = ss.by_model.get(m)
    row = {"Cost model": MODEL_LABELS[m]}
    if entry is None:
        return row | {"Status": "Not evaluated yet"}  # not shown in the table; listed below it
    if not matches_current_inputs(entry):
        return row | {"Status": "Not run with current inputs (rerun to update)"}
    sol, base = entry["result"].solution, entry["result"].baseline
    if not sol.has_plan:
        return row | {"Status": sol.status_text()}
    im = impact_metrics(base, sol)
    return row | {
        "Status": im.status_text,
        "Optimised cost": im.optimised_cost,
        "Baseline cost": im.baseline_cost,
        "Saving": im.saving,
        "Saving %": im.saving_percent,
        "Buses": f"{im.baseline_buses} → {im.optimised_buses}",
        "Cost per attendee": f"{im.cost_per_student_before:,.2f} → {im.cost_per_student_after:,.2f}"
        if im.cost_per_student_after is not None else None,
        "Seat utilisation": f"{im.optimised_utilisation * 100:.1f}%",
    }


with tabs[3]:
    section("All cost models, current inputs")
    st.caption("Each model against its own baseline (every school rents its own cheapest bus, same pricing). "
               f"Costs are in {CURRENCY_LABEL} and are not comparable across models. Only models you have run "
               "show figures; choose a goal in the sidebar and run it to add a row.")
    evaluated = [m for m in models if m in ss.by_model]
    if evaluated:
        st.dataframe(pd.DataFrame([model_row(m) for m in evaluated]), hide_index=True, width="stretch",
                     column_config={
                         **{col: st.column_config.NumberColumn(format="%.2f")
                            for col in ("Optimised cost", "Baseline cost", "Saving")},
                         "Saving %": st.column_config.NumberColumn(format="%.1f%%"),
                     })
    not_run = [MODEL_LABELS[m] for m in models if m not in ss.by_model]
    if not_run:
        st.caption("Not evaluated yet: " + "; ".join(not_run) + ". Expensive searches never run on their own.")

    prev = ss.previous
    section("Previous vs revised plan")
    if cur is None or prev is None or not cur["result"].solution.has_plan:
        st.caption("Change attendance, rules or the goal and run again: the last two valid plans are "
                   "compared here.")
    else:
        a_sol, b_sol = prev["result"].solution, cur["result"].solution
        same_model = prev["model"] == cur["model"]
        st.caption(f"Previous: {MODEL_LABELS[prev['model']]} → revised (latest run): {MODEL_LABELS[cur['model']]}.")
        if not same_model:
            st.warning("Different cost models: costs are not directly comparable.")
        c = st.columns(4)
        diff = b_sol.total_cost - a_sol.total_cost
        c[0].metric("Previous plan cost", money(a_sol.total_cost), border=True)
        c[1].metric("Revised plan cost", money(b_sol.total_cost), f"{diff:+,.2f}", delta_color="inverse",
                    border=True)
        c[2].metric("Buses", len(b_sol.routes), f"{len(b_sol.routes) - len(a_sol.routes):+d}",
                    delta_color="inverse", border=True)
        pct = (diff / a_sol.total_cost * 100) if a_sol.total_cost else 0.0
        c[3].metric("Cost change", f"{pct:+.1f}%", border=True)

        reasons = []
        att = attendance_changes(prev["dataset"], cur["dataset"])
        if att:
            reasons.append("Attendance changed: " + "; ".join(att))
        before_c, after_c = set(prev["constraints"].describe()), set(cur["constraints"].describe())
        if after_c - before_c:
            reasons.append("Rules added: " + "; ".join(sorted(after_c - before_c)))
        if before_c - after_c:
            reasons.append("Rules removed: " + "; ".join(sorted(before_c - after_c)))
        if not same_model:
            reasons.append("Cost model changed.")
        if not reasons:
            reasons.append("Inputs unchanged. Differences come from the search itself "
                           "(route search is heuristic, so repeated runs can find different plans of similar cost).")
        left, right = st.columns([2, 3], gap="large")
        with left:
            st.markdown("**What changed in the inputs**")
            st.markdown("\n".join(f"- {r}" for r in reasons))
        with right:
            st.markdown("**Schools whose bus changed**")
            changes = compare_plans(a_sol, b_sol)
            if changes.empty:
                st.success("Every school keeps the same bus-mates, bus size and pickup order.")
            else:
                st.dataframe(changes, hide_index=True, width="stretch")

    if ss.by_model:
        with st.expander("More metrics and solver details", icon=":material/query_stats:"):
            st.caption("Each model's latest run, including runs on older inputs. Compare buses, utilisation, "
                       "miles and hours across models, not costs. Driving hours use cached OSRM road times "
                       "when present (no network request).")
            st.dataframe(session_summary(), hide_index=True, width="stretch")
            if cur is not None and cur["result"].solution.has_plan:
                for m, e in ss.by_model.items():
                    if e is not cur and e["result"].solution.has_plan:
                        st.markdown(f"**{MODEL_LABELS[cur['model']]} vs {MODEL_LABELS[m]}:** "
                                    + " ".join(model_difference(cur["result"].solution, e["result"].solution)))

# --------------------------------------------------------------------------- #
# Scenarios: a separate scenario, compared with the current plan
# --------------------------------------------------------------------------- #


def scenario_base():
    """The plan a scenario starts from: the current plan, only if it is valid and up to date."""
    if cur is None or not cur["result"].solution.has_plan or fresh.state == STALE:
        return None
    return cur


def discard_scenario():
    ss = st.session_state
    ss.scenario = None
    ss.scn_confirm = False


def adopt_scenario():
    """Make the scenario's attendance the plan's attendance. Prices cannot be adopted (they are data).

    The scenario result is stored under the run key of the new inputs, so the
    next rerun shows it as the current plan without a new search. If the rules
    or settings changed meanwhile, the key differs and the page asks for a run.
    """
    ss = st.session_state
    sc = ss.get("scenario")
    if sc is None or sc["inputs"].changes_prices or not ss.get("scn_confirm"):
        return
    entry = sc["entry"]
    ss.overrides = dict(entry["overrides"])
    ss.editor_base = dict(ss.overrides)
    ss.attendance_errors = []
    ss.pop("att_editor", None)  # the attendance editor redraws from editor_base
    cache_put(ss.result_cache, entry["key"], entry)
    ss.last_valid = entry
    ss.scenario = None
    ss.scn_confirm = False
    ss.scenario_msg = (f"Scenario adopted: its attendance is now in **Adjust plan**, and its result ({entry['run_no']}) "
                       "is the current plan. The plan it replaced is in **Compare**.")


def cell(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    return f"{v:,g}" if isinstance(v, (int, float)) else str(v)


with tabs[4]:
    section("Test a scenario")
    st.caption("Test a change, such as more attendees or higher bus prices, without touching the current plan. "
               "The scenario uses the same optimiser, rules and independent check. Nothing is predicted: you "
               "enter the numbers.")
    if msg := ss.pop("scenario_msg", None):
        st.success(msg, icon=":material/check_circle:")
    base_entry = scenario_base()
    if base_entry is None:
        st.info("Generate an up-to-date plan first. A scenario starts from the current plan's attendance, "
                "rules and cost model.", icon=":material/science:")
    else:
        m = base_entry["model"]
        base_ds = base_entry["dataset"]
        base_att = dict(zip(base_ds.schools["school_id"], (int(a) for a in base_ds.schools["attendees"])))
        st.markdown(f"**Starts from:** {describe_entry(base_entry)} &nbsp; {status_badge(base_entry['result'].solution)}")
        w1, w2 = st.columns(2)
        scale = w1.number_input(
            "Change every school's attendance by (%)", MIN_SCALE, MAX_SCALE, 0, step=5, key="scn_scale",
            help="Applied to each school's current attendance and rounded up to a whole person, so there "
                 "are always enough seats. 0 = no change.")
        price = w2.number_input(
            f"Change bus prices ({MODEL_UNITS[m]}) by (%)", MIN_PRICE, MAX_PRICE, 0, step=5, key="scn_price",
            help="Applied to every bus size of this cost model, and to its baseline. For testing only: a scenario "
                 "with a price change cannot become the plan.")
        with st.expander("Set attendance for individual schools (optional)", icon=":material/edit:"):
            st.caption("Leave a cell empty to use the value above. A value here replaces it.")
            edited = st.data_editor(
                pd.DataFrame({"School": list(base_att), "Current plan": list(base_att.values()),
                              "Scenario": [float("nan")] * len(base_att)}),
                hide_index=True, width="stretch", height=300, key="scn_editor", disabled=["School", "Current plan"],
                column_config={"Scenario": st.column_config.NumberColumn(min_value=0, max_value=500, step=1)})
        edits = tuple((s, v) for s, v in zip(edited["School"], edited["Scenario"])
                      if v is not None and not (isinstance(v, float) and math.isnan(v)))
        limit = base_entry["settings"]["time_limit"]
        if m != "fixed":
            st.caption(f"This route-search scenario runs for the plan's full time limit ({limit:g} s).")
        if st.button("Run scenario", type="primary", key="scn_run", icon=":material/science:"):
            inputs = ScenarioInputs(int(scale), edits, int(price))
            check = scenario_attendance(base_att, inputs)
            if check.errors:
                for e in check.errors:
                    st.error(e)
            else:
                try:
                    working = scenario_dataset(dataset, check.attendance, m, inputs.price_percent)
                    tm = get_time_matrix(points_key(), allow_online, time_source) if m == "time" else None
                    with st.spinner("Optimising the scenario..."):
                        result = optimise(working, m, base_entry["constraints"], limit, tm,
                                          warm_start=base_entry["result"].solution)
                    ss.scenario_runs = ss.get("scenario_runs", 0) + 1
                    s_over = overrides_vs_source(dataset, check.attendance)
                    ss.scenario = {
                        "inputs": inputs, "base": base_entry, "base_attendance": base_att,
                        "entry": {
                            "result": result, "model": m, "overrides": s_over,
                            "constraints": base_entry["constraints"], "dataset": working,
                            "time_source": tm.source if tm else None, "time_notes": tm.notes if tm else [],
                            "key": None if inputs.changes_prices else run_key(
                                m, s_over, base_entry["constraints"], allow_online, time_source),
                            "trigger": "scenario", "run_no": f"S{ss.scenario_runs}",
                            "finished_at": time.strftime("%H:%M:%S"),
                            "settings": {"time_limit": float(limit), "warm_start": True,
                                         "allow_online": allow_online if m == "time" else None},
                        },
                    }
                    ss.scn_confirm = False
                except Exception as exc:  # never lose the current plan because of a scenario
                    st.error(f"The scenario could not be optimised ({type(exc).__name__}: {exc}). "
                             "The current plan is unchanged.")

    sc = ss.get("scenario")
    if sc is not None:
        s_entry, s_base = sc["entry"], sc["base"]
        s_sol, s_baseline = s_entry["result"].solution, s_entry["result"].baseline
        st.divider()
        st.markdown(f"**Scenario {s_entry['run_no']}** (finished {s_entry['finished_at']}) — not the current plan "
                    f"&nbsp; {status_badge(s_sol)}")
        changes = attendance_diff(sc["base_attendance"], {s: int(a) for s, a in zip(
            s_entry["dataset"].schools["school_id"], s_entry["dataset"].schools["attendees"])})
        what = []
        if sc["inputs"].scale_percent:
            what.append(f"attendance {sc['inputs'].scale_percent:+d}% for every school (rounded up)")
        if sc["inputs"].edits:
            what.append(f"{len(sc['inputs'].edits)} school(s) set individually")
        if sc["inputs"].changes_prices:
            what.append(f"bus prices {sc['inputs'].price_percent:+d}%")
        st.caption("Changes tested: " + ("; ".join(what) if what else "none (same inputs as the plan)") + ". "
                   + (f"{len(changes)} school(s) with different attendance." if changes else ""))
        if s_base is not cur:
            st.warning("The current plan changed after this scenario ran. The comparison below is with the plan "
                       "the scenario started from.", icon=":material/schedule:")
        if not s_sol.has_plan:
            st.error("No valid plan for this scenario. " + " ".join(s_sol.notes + s_sol.violations))
        else:
            table = comparison_table((s_base["result"].solution, s_base["result"].baseline), (s_sol, s_baseline),
                                     entry_distance(s_base), entry_distance(s_entry))
            show = table.copy()
            for col in ("Current plan", "Scenario"):
                show[col] = show[col].map(cell)
            show["Difference"] = show["Difference"].map(lambda v: "" if cell(v) == "" else f"{v:+,g}")
            st.dataframe(show, hide_index=True, width="stretch")
            st.caption("Same cost model on both sides, so costs are comparable. "
                       + ("The baseline is re-priced with the scenario's prices. " if sc["inputs"].changes_prices
                          else "")
                       + "Bus-miles are straight-line; fuel and emissions are not estimated.")
            moved = compare_plans(s_base["result"].solution, s_sol)
            with st.expander(f"Schools whose bus changed ({len(moved)})", icon=":material/swap_horiz:"):
                if moved.empty:
                    st.caption("Every school keeps the same bus-mates, bus size and pickup order.")
                else:
                    st.dataframe(moved, hide_index=True, width="stretch")
        a1, a2 = st.columns([3, 1], vertical_alignment="bottom")
        if sc["inputs"].changes_prices:
            a1.info("Price changes are for testing only: bus prices come from the supplied price files, so this "
                    "scenario can be compared but not used as the plan.")
        elif s_sol.has_plan:
            with a1:
                st.checkbox("Replace the current attendance with this scenario's attendance", key="scn_confirm")
                st.button("Use this scenario as the plan", key="scn_adopt", on_click=adopt_scenario,
                          disabled=not ss.get("scn_confirm"), icon=":material/done_all:")
        a2.button("Discard scenario", key="scn_discard", on_click=discard_scenario, width="stretch")

# --------------------------------------------------------------------------- #
# Responsible AI
# --------------------------------------------------------------------------- #

STRATEGY_URL = "https://www.gov.wales/national-events-strategy-wales-2022-2030-html"
STATUS_MEANINGS = [
    (OPTIMAL, "The solver proved that no valid plan costs less."),
    (FEASIBLE, "A valid plan that passed every check, but there is no proof it is the cheapest. "
               "All routing results (distance, time) are in this group."),
    (INFEASIBLE, "The inputs cannot be met, for example a school larger than the biggest bus or "
                 "conflicting constraints. No plan is shown, and the reason is given."),
    (NO_SOLUTION, "The time limit ran out before any plan was found. No plan is shown."),
    (INVALID, "The solver returned a plan that failed the independent check. It is blocked from "
              "display and export."),
]
MODEL_EXPLAINERS = {
    "fixed": ("Total rental price. Each bus has a fixed price by size, so location does not change "
              "the cost.", "Exact (CP-SAT). Can prove the plan is the cheapest."),
    "distance": ("Rate per mile x straight-line miles from the first pickup, through each stop, to the "
                 "event venue.", "Heuristic (guided local search). Valid plan, no proof of the minimum."),
    "time": ("Rate per hour x driving hours on the same route. Road times come from OSRM; if they are "
             "unavailable, a labelled offline estimate is used.",
             "Heuristic (guided local search). Valid plan, no proof of the minimum."),
}

with tabs[5]:
    section("Decision support, not automation")
    p1, p2, p3 = st.columns(3)
    with p1.container(border=True):
        st.markdown("**:material/person: People decide**")
        st.caption("The optimiser proposes; organisers inspect, adjust, compare and approve. Assistant "
                   "proposals change nothing until approved.")
    with p2.container(border=True):
        st.markdown("**:material/verified: Every plan is checked**")
        st.caption("Separate code re-checks capacity, splits, rules and costs. A failed plan is never "
                   "shown or exported.")
    with p3.container(border=True):
        st.markdown("**:material/lock: School-level data only**")
        st.caption("Head counts per pickup point, no student names. Online services are optional.")
    st.info("**Not measured:** fuel use, carbon emissions, planning time and money saved for a real event. "
            "Fewer buses means fewer vehicles to book, but emissions were not calculated. "
            "The proposed pilot below would measure these.", icon=":material/info:")

    with st.expander("Six Responsible AI principles", icon=":material/balance:", expanded=True):
        st.markdown(pd.DataFrame([
            ("Transparency", "Every plan shows its status (proven optimal or best found), method, buses, costs and "
             "each school's share. Estimates are labelled. Reports state assumptions and limits.",
             "The solver does not say why it chose a grouping; explanations mark readings as interpretation."),
            ("Privacy", "Head counts per postcode outcode only: no student or school names. Coordinates go only to "
             "OSRM; the language model gets outcodes and counts, never coordinates. Both are optional.",
             "Public OSRM and LLM APIs are third-party services; self-host for sensitive locations."),
            ("Accountability", "The organiser approves every AI-proposed change. A run log and audit report record "
             "each run, its trigger, settings and outcome.",
             "Final booking sign-off is a recommended procedure, not a built-in step."),
            ("Safety", "Hard rules (no split school, no bus over capacity, organiser rules) are enforced by the "
             "optimiser and re-checked by separate code. A plan that fails is never shown or exported.",
             "No real-world safety guarantee: drivers, roads and supervision are outside the model."),
            ("Security", "API keys come from the environment or a git-ignored secrets file and are never shown or "
             "logged. The language model has no tools and returns JSON that code validates.",
             "This prototype has no login; put it behind access control before sharing."),
            ("Fairness", "Every school pays in proportion to its attendees on its bus, whatever its pickup position. "
             "Each school is planned by the same rules regardless of size or location.",
             "Accessible-transport needs are not in the data; see Inclusion and accessibility."),
        ], columns=["Principle", "How Campus Cruise applies it", "Limit"]).to_markdown(index=False))
        st.caption("A description of the prototype's design, not a certification.")

    with st.expander("Who could benefit", icon=":material/diversity_3:"):
        st.caption("Potential benefits. No group has tested or endorsed Campus Cruise yet; the pilot below would "
                   "measure them.")
        st.markdown(pd.DataFrame([
            ("Event organisers and event SMEs", "A checked shared-bus plan in seconds instead of manual grouping; "
             "scenario tests before booking."),
            ("Schools and teachers", "A lower transport cost and a clear, attendee-based cost share per school; one "
             "group never split from its supervisor."),
            ("Pupils", "A lower cost to attend, which can make a trip affordable for more pupils."),
            ("Welsh coach and bus operators", "A clear booking list: bus sizes, stops and pickup order per bus."),
            ("Venues and their transport teams", "Fewer vehicles to park and marshal (the bus count is calculated)."),
            ("Audiences and communities", "Fewer vehicles on roads near the event; emissions are not measured."),
        ], columns=["Group", "Potential benefit"]).to_markdown(index=False))

    with st.expander("How a plan is made and checked", icon=":material/settings:"):
        cols = st.columns(len(models))
        for col, m in zip(cols, models):
            with col.container(border=True):
                goal, method = MODEL_EXPLAINERS[m]
                st.markdown(f"**{MODEL_LABELS[m]}**")
                st.markdown(f"*Minimises:* {goal}")
                st.markdown(f"*Method:* {method}")
        left, right = st.columns(2)
        with left:
            st.markdown("**Hard rules, never relaxed**")
            st.markdown(HARD_RULES)
            st.markdown("**How schools are assigned**")
            st.markdown(
                "The solver chooses which schools share each bus and which bus size each group uses, "
                "to make the total cost of the selected model as low as possible. For the distance and time "
                "models it also chooses the pickup order."
            )
        with right:
            st.markdown("**Independent check of every plan**")
            st.markdown(INDEPENDENT_CHECK)
        current_status = cur["result"].solution.status if cur is not None else None
        st.markdown("**What the solver status means**")
        st.dataframe(pd.DataFrame([
            {"Status": Solution(model="", status=s).status_text(), "Meaning": text,
             "Current plan": "◀ this plan" if s == current_status else ""}
            for s, text in STATUS_MEANINGS
        ]), hide_index=True, width="stretch")

    with st.expander("Human oversight", icon=":material/groups:"):
        st.markdown(
            "1. **The optimiser proposes** a plan for the chosen goal and time limit.\n"
            "2. **Organisers inspect it:** every bus, school, pickup order, cost and cost share is shown "
            "(Routes tab).\n"
            "3. **Organisers adjust** attendance or sharing rules (Adjust plan tab) and rerun. "
            "Until they rerun, the page says the results are out of date.\n"
            "4. **Organisers compare** the old and new plan (Compare tab) before choosing.\n"
            "5. **AI Assistant proposals need approval (implemented):** a language model only proposes; the "
            "organiser sees the exact changes and clicks Approve or Reject. Nothing changes before approval."
        )
        st.warning("**Final sign-off is a recommended procedure, not a built-in feature.** Campus Cruise has no "
                   "step for approving the final transport plan. An organiser should review the exported plan "
                   "and confirm it with schools and the bus company before booking.")

    with st.expander("Data protection", icon=":material/lock:"):
        d1, d2 = st.columns(2)
        with d1:
            st.markdown("**What the app processes**")
            st.markdown(
                "- One row per pickup point: a postcode outcode (for example *CF15*), one coordinate inside "
                "that outcode, and a head count.\n"
                "- Bus prices by size.\n"
                "- **No student names, ages or individual records.** No school names either: the data "
                "identifies schools by outcode only.\n"
                "- Attendance edits stay in the app session. The source CSV files are never changed."
            )
            st.markdown("**What leaves this computer**")
            st.markdown(
                "- **Time model with OSRM ticked:** the coordinates of the pickup points and the event venue go "
                "to the public OSRM server (router.project-osrm.org). Head counts are not sent.\n"
                "- **Road paths (Routes tab):** each bus's stop coordinates go to the same server.\n"
                "- **Destination search (sidebar, only when used):** the typed venue name or address goes to "
                "OpenStreetMap Nominatim (nominatim.openstreetmap.org). No school data.\n"
                "- **Map display:** the viewer's browser loads map tiles from OpenStreetMap and "
                "map scripts from public CDNs.\n"
                "- **AI Assistant (only when used and configured):** the request text, school outcodes with head "
                "counts and bus assignments, bus prices, rules and plan figures go to the configured provider's "
                "API (OpenAI or Anthropic). No coordinates.\n"
                "- Untick OSRM to make no routing request: cached road times or a labelled estimate are used."
            )
        with d2:
            st.markdown("**What is stored**")
            st.markdown(
                "- OSRM answers are cached in `data/cache/` (coordinates and travel times only); destination "
                "search answers in `data/cache/geocode/`.\n"
                "- Downloads (CSV, Markdown) are created only when a user clicks them.\n"
                "- The only credential is the optional `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` for the assistant, "
                "read from the environment or `.streamlit/secrets.toml` (git-ignored); it is never shown or "
                "logged. Streamlit usage statistics are switched off in `.streamlit/config.toml`."
            )
            st.markdown("**Recommendations for real use**")
            st.markdown(
                "- Keep data at school level. Do not add student-level personal data; the optimiser does not need it.\n"
                "- Put the app behind a login before sharing it: this version has no access control.\n"
                "- Delete `data/cache/` and exports after the event.\n"
                "- Use a self-hosted routing server if pickup locations are sensitive."
            )

    with st.expander("Inclusion and accessibility", icon=":material/accessible:"):
        st.info(
            "**Proposed future enhancement, not implemented: accessible-bus constraint.** "
            "The supplied data has no information about passengers who need accessible transport, or about "
            "which buses are accessible, so this version cannot plan for it. Proposal: record, per school, "
            "the number of passengers needing a wheelchair space, and, per bus type, its accessible spaces. "
            "The optimiser would then place those schools only on buses with enough accessible spaces, and "
            "the independent check would re-test this rule."
        )

    with st.expander("Relevance to events in Wales", icon=":material/flag:"):
        st.markdown(
            f"The [National Events Strategy for Wales 2022–2030]({STRATEGY_URL}) states, among others:\n\n"
            "> *\"A balanced portfolio will consider age, region, ethnicity, religion, cost to attend and so on "
            "to ensure the events on offer are widely inclusive and accessible.\"*\n\n"
            "> Success measure: *\"Percentage of events encouraging or offering active and sustainable "
            "transport options.\"*\n\n"
            "> *\"Research and evaluation are critical to understanding progress within the industry and what "
            "it is achieving both for itself and for Wales.\"*"
        )
        st.markdown(
            "How Campus Cruise could help:\n"
            "- **Cost to attend:** it shows organisers and schools the lowest transport cost it can find and "
            "each school's share, which can lower one cost of bringing school groups to an event.\n"
            "- **Shared transport:** it groups schools onto shared coaches, so fewer vehicles are booked "
            "(see the bus counts on the Overview). Emissions are not measured.\n"
            "- **Evidence:** every plan is exported with its assumptions, status and costs, so results can "
            "be evaluated and compared."
        )
        st.markdown(
            "**Well-being of Future Generations (Wales) Act 2015.** The strategy says: *\"The starting point for "
            "what is 'good for Wales' is the Well-being Act.\"* and *\"Sustainability is at the heart of the "
            "Well-being act\"*. Two of the Act's seven well-being goals relate to this tool:\n"
            "- *A more equal Wales:* a lower, fairly shared cost to attend can help pupils from every school "
            "take part.\n"
            "- *A globally responsible Wales:* sharing coaches reduces the number of vehicles. Bus-miles saved "
            "are calculated (straight-line); fuel and emissions are not.\n\n"
            "**Welsh language (not implemented).** The strategy measures *\"Promotion and use of Welsh language at "
            "events\"*. The interface is English only; a bilingual (Cymraeg) interface is a possible next step."
        )
        st.caption("This is the team's reading of the strategy. Campus Cruise is not endorsed by the Welsh "
                   "Government, and no benefit for a real Welsh event has been measured yet. "
                   "Figures come from the supplied challenge data for a fictional event.")

    with st.expander("Proposed pilot", icon=":material/science:"):
        st.caption("Proposed, not yet run. No pilot results exist, and no stakeholder feedback has been collected.")
        st.markdown(
            "- **Partner:** one Welsh event organiser, or a local-authority school transport coordinator, "
            "planning a school trip to one event.\n"
            "- **Information needed:** pickup points (postcode outcode or school address), head count per group, "
            "the bus company's prices per size, and any sharing or accessibility needs. No pupil names.\n"
            "- **Process:** the organiser plans as usual; in parallel Campus Cruise produces a plan from the same "
            "inputs; both are compared before booking, and the organiser chooses which to book.\n"
            "- **Feedback:** a short survey and a 20-minute interview with the organiser after the event.\n"
            "- **Privacy and consent:** written consent from the partner; school-level data only; data deleted "
            "after the evaluation; results published only in aggregate and with permission.\n\n"
            "Measures to compare:"
        )
        st.dataframe(pd.DataFrame([
            ("Transport cost", "Quoted cost of the usual plan vs the Campus Cruise plan, same bus company prices"),
            ("Buses needed", "Number of vehicles booked in each plan"),
            ("Bus mileage", "Road miles of each plan, from the bus company's trip records"),
            ("Attendee allocation", "Every group seated on one bus, empty seats per bus"),
            ("Time to produce a plan", "Organiser hours for the usual method vs with the tool"),
            ("Manual corrections", "Changes the organiser had to make to the proposed plan, and why"),
            ("Constraint problems found", "Capacity, split-school or sharing-rule problems caught before booking"),
            ("Accessibility needs", "Requests for accessible transport and whether each plan met them"),
            ("Organiser feedback", "Short survey: clarity, trust, what was missing"),
        ], columns=["Measure", "How to measure"]), hide_index=True, width="stretch")

# --------------------------------------------------------------------------- #
# Data & checks
# --------------------------------------------------------------------------- #

with tabs[6]:
    rep = dataset.report
    data_ok = rep.ok and not any(rep.capacity_errors.values())
    if data_ok:
        st.success("All input files are valid. Every school fits on at least one available bus.",
                   icon=":material/check_circle:")
    for e in rep.errors:
        st.error(e)
    for w in rep.warnings:
        st.warning(w)
    for m in dataset.available_models:
        if rep.capacity_errors.get(m):
            for msg in rep.capacity_errors[m]:
                st.error(f"[{m}] {msg}")
    c = st.columns(4)
    c[0].metric("Schools", len(dataset.schools), border=True)
    c[1].metric("Attendees (source data)", int(dataset.schools["attendees"].sum()), border=True)
    c[2].metric("Largest school (source data)", int(dataset.schools["attendees"].max()) if len(dataset.schools)
                else 0, border=True)
    c[3].metric("Cost models available", len(dataset.available_models), border=True)
    st.caption(f"School data: **{DATA_LABEL}**"
               + (" (uploaded this session; the bundled CSV is unchanged)" if _src is not None else "")
               + ". Attendance edits from **Adjust plan** are not shown here.")

    left, right = st.columns([1, 1], gap="large")
    with left:
        section("Schools (pickup points)")
        st.caption("The data identifies schools by UK postcode outcode only; no school names are supplied.")
        st.dataframe(dataset.schools.drop(columns=["name"]), hide_index=True, width="stretch", height=320)
    with right:
        section("Bus options")
        sizes = sorted({s for r in dataset.rates.values() for s in r})
        bus_df = pd.DataFrame({"Seats": sizes})
        for m in COST_MODELS:
            if m in dataset.rates:
                header_name = {"fixed": "Per bus", "distance": "Per mile", "time": "Per hour"}[m]
                bus_df[header_name] = [dataset.rates[m].get(s) for s in sizes]
        st.dataframe(bus_df, hide_index=True, width="stretch")
        section("Event")
        st.markdown(f"{dataset.event.name}  \n{dataset.event.latitude}, {dataset.event.longitude} "
                    + ("(from the challenge data README)" if same_venue(dataset.event, DEFAULT_EVENT)
                       else "(set in the sidebar, **Destination**)"))

    with st.expander("Assumptions and limitations", icon=":material/list:"):
        st.markdown(
            "- Bus size = passenger seats; attendance counts everyone who travels.\n"
            "- Unlimited buses of each size are available.\n"
            "- A bus starts at its first pickup and ends at the event venue; no return trip.\n"
            "- Distance model: straight-line (haversine) miles between consecutive stops.\n"
            "- Time model: OSRM road driving times when available; no dwell time at stops.\n"
            f"- Currency not stated in data; shown as {CURRENCY_LABEL}."
        )

    with st.expander("Travel-time data and machine learning", icon=":material/model_training:"):
        ready = data_readiness()
        st.markdown(
            f"- **Observed journey records:** {ready['observed_journey_records']}. The data has no trip logs, "
            "timestamps, traffic or weather.\n"
            f"- **OSRM road-time estimates cached:** {ready['osrm_matrix_pairs']} origin-destination pairs; "
            f"{ready['osrm_route_paths']} road paths for the map.\n"
            f"- **Verdict:** {ready['verdict']}"
        )
        ml = get_ml_model()
        if ml is None:
            st.info("No trained ML artefact found (`models/travel_time_ml.json`). Run `python train_travel_time.py` "
                    "to train one. The time model works without it.")
        else:
            art = ml.artefact
            ev = art["evaluation"]
            st.markdown(f"**Experimental model:** {art['model']['description']} (trained {art['created']} on "
                        f"{art['training_data']['pairs']} pairs from `{art['training_data']['file']}`). "
                        f"{art['intended_use']}")
            st.dataframe(pd.DataFrame([
                {"Method": ev["candidates"][k], "MAE (min)": round(v["mae_min"], 2),
                 "RMSE (min)": round(v["rmse_min"], 2),
                 "MAPE % (pairs > 5 min)": None if v["mape_pct"] is None else round(v["mape_pct"], 1),
                 "Test pairs": v["n"]}
                for k, v in ev["metrics"].items()]), hide_index=True, width="stretch")
            st.caption(f"Evaluation: {ev['method']}. Target: OSRM estimates, so these errors measure how well "
                       "each method reproduces OSRM, not real journeys. "
                       + ("The model beat the fixed-speed estimate and can be selected as the time model's "
                          "travel-time source (sidebar, Advanced settings)." if ml.usable else
                          "The model did not beat the fixed-speed estimate, so the app will not use it."))
            for lim in art["limitations"]:
                st.markdown(f"- {lim}")

# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #

with tabs[7]:
    if cur is None or not cur["result"].solution.has_plan:
        st.info("Only a validated plan can be exported. Generate a plan first.", icon=":material/description:")
    else:
        res, ds = cur["result"], cur["dataset"]
        stale_note()
        overrides_text = attendance_changes(dataset, ds)
        cur_distance = entry_distance(cur)
        report_md = plan_report(res.solution, res.baseline, ds, cur["constraints"], overrides_text,
                                distance_lines(cur_distance))
        summary = session_summary()
        audit_md = audit_report(cur, dataset, ss.run_log, summary, own_matrix(cur),
                                current=fresh.state != STALE, distance=cur_distance)
        r1, r2 = st.columns(2, gap="medium")
        with r1.container(border=True):
            st.markdown("**:material/directions_bus: Transport plan**")
            st.caption("Every bus, its schools and pickup order, for booking and sharing with schools.")
            st.download_button("Download plan (CSV)", plan_csv(res.solution, ds), icon=":material/download:",
                               file_name=f"campus_cruise_{cur['model']}_plan.csv", mime="text/csv")
            st.download_button("Download report (Markdown)", report_md, icon=":material/download:",
                               file_name=f"campus_cruise_{cur['model']}_report.md", mime="text/markdown")
        with r2.container(border=True):
            st.markdown("**:material/fact_check: Audit report**")
            st.caption("Summary, data, baseline method, approach, results, a fresh independent validation, "
                       "solver status, limitations and this session's run log.")
            st.download_button("Download audit report (Markdown)", audit_md, key="dl_audit",
                               icon=":material/download:",
                               file_name=f"campus_cruise_{cur['model']}_audit.md", mime="text/markdown")
            st.download_button("Download model metrics (CSV)", metrics_csv(summary), key="dl_metrics",
                               icon=":material/download:",
                               file_name="campus_cruise_model_metrics.csv", mime="text/csv")
        with st.expander("Preview report"):
            st.markdown(report_md)
        with st.expander("Preview audit report"):
            st.markdown(audit_md)

    if ss.run_log:
        with st.expander(f"Run log this session ({len(ss.run_log)})", icon=":material/history:"):
            st.caption("Every optimiser run attempt, including failures. Settings and figures only: no "
                       "request text or personal data.")
            st.dataframe(pd.DataFrame(ss.run_log), hide_index=True, width="stretch")
