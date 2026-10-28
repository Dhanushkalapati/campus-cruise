"""Campus Cruise: Streamlit dashboard.

Run:  streamlit run app.py
All optimisation logic lives in ``src/``; this file only handles the UI.
"""

from __future__ import annotations

import math

import pandas as pd
import streamlit as st

from src.data import COST_MODELS, CURRENCY_LABEL, MODEL_LABELS, attendance_edits, load_dataset
from src.distances import SOURCE_APPROX, RoutingServiceError, TimeMatrix, approximate_time_matrix, road_path
from src.model import Constraints
from src.planner import optimise, time_matrix_for
from src.report import (
    attendance_changes,
    bus_table,
    compare_plans,
    cost_shares,
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
      .block-container { padding-top: 3.5rem; max-width: 1400px; }
      .cc-sub { color: #5b6573; margin-top: .1rem; margin-bottom: 1rem; }
      div[data-testid="stMetric"] { border: 1px solid rgba(128,128,128,.25);
          border-radius: 10px; padding: .7rem .9rem; }
      .cc-pill { display:inline-block; padding:.15rem .6rem; border-radius:999px;
          font-size:.85rem; font-weight:600; }
      .cc-ok { background:#e3f4e8; color:#1d6b35; }
      .cc-warn { background:#fff3d6; color:#7a5300; }
      .cc-bad { background:#fde2e1; color:#8f1d18; }
    </style>
    """,
    unsafe_allow_html=True,
)

PALETTE = [
    "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b",
    "#e377c2", "#bcbd22", "#393b79", "#637939", "#843c39", "#7b4173", "#3182bd",
    "#e6550d", "#31a354", "#756bb1", "#636363",
]


# --------------------------------------------------------------------------- #
# State and cached inputs
# --------------------------------------------------------------------------- #


@st.cache_resource
def get_dataset():
    return load_dataset()


@st.cache_data(show_spinner="Fetching travel-time matrix...")
def get_road_matrix(points_key: tuple, allow_online: bool) -> TimeMatrix:
    """Road times only (disk cache or OSRM). Failures raise and are not cached,
    so a later run retries OSRM instead of reusing a stored fallback."""
    return time_matrix_for(get_dataset(), allow_online=allow_online, allow_approximate=False)


def get_time_matrix(points_key: tuple, allow_online: bool) -> TimeMatrix:
    try:
        return get_road_matrix(points_key, allow_online)
    except RoutingServiceError as exc:
        points = [tuple(p) for p in points_key] + [(dataset.event.latitude, dataset.event.longitude)]
        return TimeMatrix(approximate_time_matrix(points), SOURCE_APPROX,
                          [str(exc), "Using the offline estimate. Travel times are approximate, not road routing."])


def init_state():
    ss = st.session_state
    ss.setdefault("must_share", [])
    ss.setdefault("cannot_share", [])
    ss.setdefault("solo", [])
    ss.setdefault("overrides", {})         # school -> edited attendance (valid values only)
    ss.setdefault("attendance_errors", [])  # rejected edits; block the optimiser
    ss.setdefault("current", None)   # dict: result, model, overrides, constraints, dataset
    ss.setdefault("previous", None)


# Widget callbacks. Streamlit runs callbacks before the script body, so every
# edit is in session state before the "Run optimiser" block reads it, even
# when the edit and the button click arrive in the same rerun.


def sync_attendance():
    ss = st.session_state
    state = ss.get("att_editor") or {}
    ss.overrides, ss.attendance_errors = attendance_edits(
        state.get("edited_rows", {}), school_ids, dataset.schools["attendees"].tolist(), "Attendees")


def reset_attendance():
    ss = st.session_state
    ss.overrides, ss.attendance_errors = {}, []
    ss.pop("att_editor", None)


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


def money(x: float) -> str:
    return f"{x:,.2f}"


def status_pill(sol) -> str:
    cls = "cc-ok" if sol.proven_optimal else ("cc-warn" if sol.has_plan else "cc-bad")
    return f'<span class="cc-pill {cls}">{sol.status_text()}</span>'


init_state()
dataset = get_dataset()
ss = st.session_state
school_ids = dataset.schools["school_id"].tolist()

# --------------------------------------------------------------------------- #
# Header and sidebar controls
# --------------------------------------------------------------------------- #

st.title("🚌 Campus Cruise")
st.markdown(
    f'<p class="cc-sub">Where travel gets smarter. School bus sharing to {dataset.event.name}.</p>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Optimiser")
    models = [m for m in COST_MODELS if m in dataset.available_models]
    if not models:
        st.error("No valid cost data. See the Data tab.")
        st.stop()
    model = st.radio("Cost model", models, format_func=lambda m: MODEL_LABELS[m])
    time_limit = st.slider("Solver time limit (seconds)", 5, 120, 20, step=5)
    allow_online = True
    if model == "time":
        allow_online = st.checkbox(
            "Use real road times from OSRM (online)", value=True,
            help="Public OSRM table service. Results are cached locally. If unchecked, no network "
                 "request is made: cached road times are used if present. If no road times are "
                 "available, an approximate offline estimate is used and labelled.",
        )

    pending = []
    if ss.current is not None:
        if ss.current["model"] != model:
            pending.append("cost model")
        if ss.current["overrides"] != ss.overrides:
            pending.append("attendance")
        if ss.current["constraints"] != current_constraints():
            pending.append("constraints")
    else:
        pending.append("no plan yet")
    warm = False
    if ss.current is not None and ss.current["result"].solution.has_plan:
        warm = st.checkbox(
            "Start from previous plan", value=True,
            help="Warm start: the last valid plan is repaired for your changes (new attendance, "
                 "constraints) and the search starts from it. Routing then only improves on it, "
                 "so results change only where your edits require it.",
        )
    run = st.button("Run optimiser", type="primary", width="stretch", key="run_optimiser")
    if ss.current is not None and pending:
        st.info("Not applied yet: " + ", ".join(pending) + ". Run the optimiser to apply.")
    if ss.attendance_errors:
        st.error("Fix the attendance errors in **Adjust plan** before running.")
    st.caption(f"Costs are in {CURRENCY_LABEL}: the supplied data does not state a currency.")

# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #

if run:
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
        tm = None
        if model == "time":
            # Coordinates do not change with attendance, so one matrix serves every run.
            pts = tuple(map(tuple, dataset.schools[["latitude", "longitude"]].to_numpy().round(6)))
            tm = get_time_matrix(pts, allow_online)  # never fails: falls back to the labelled estimate
        if model != "time" or tm is not None:
            with st.spinner(f"Optimising ({MODEL_LABELS[model]}, up to {time_limit} s)..."):
                start = ss.current["result"].solution if warm else None
                result = optimise(working, model, current_constraints(), time_limit, tm, warm_start=start)
            if ss.current is not None and ss.current["result"].solution.has_plan:
                ss.previous = ss.current
            ss.current = {
                "result": result, "model": model, "overrides": dict(ss.overrides),
                "constraints": current_constraints(), "dataset": working,
                "time_source": tm.source if tm else None, "time_notes": tm.notes if tm else [],
            }
            st.rerun()

cur = ss.current
if cur is not None and [p for p in pending if p != "no plan yet"]:
    st.warning(
        "The results below come from the last run. They do not include your changes to: "
        + ", ".join(pending) + ". Press **Run optimiser** to apply them."
    )
tabs = st.tabs(["Overview", "Plan & map", "Adjust plan", "Compare", "Data & validation", "Export"])

# --------------------------------------------------------------------------- #
# Overview
# --------------------------------------------------------------------------- #

with tabs[0]:
    if cur is None:
        st.info("Choose a cost model in the sidebar and press **Run optimiser**.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Schools", len(dataset.schools))
        source_total = int(dataset.schools["attendees"].sum())
        edited_total = int(dataset.with_attendance(ss.overrides).schools["attendees"].sum())
        c2.metric("Attendees (with your edits)" if ss.overrides else "Attendees", edited_total,
                  f"{edited_total - source_total:+d} vs source data" if ss.overrides else None)
        c3.metric("Bus sizes", ", ".join(str(s) for s in dataset.bus_sizes(models[0])))
    else:
        res = cur["result"]
        sol, base = res.solution, res.baseline
        st.markdown(
            f"**{MODEL_LABELS[cur['model']]}** &nbsp; {status_pill(sol)}", unsafe_allow_html=True
        )
        if not sol.has_plan:
            st.error("No valid plan. " + " ".join(sol.notes + sol.violations))
        else:
            sv = savings(base, sol)
            pct = "n/a" if sv.percent is None else f"{sv.percent:.1f}%"
            c = st.columns(3) + st.columns(3)
            c[0].metric("Total cost", money(sol.total_cost))
            c[1].metric("Savings", money(sv.absolute), pct,
                        help="Compared with every school renting its own cheapest bus.")
            c[2].metric("Buses", len(sol.routes), f"{len(sol.routes) - len(base.routes):+d} vs baseline",
                        delta_color="inverse")
            c[3].metric("Attendees", sol.total_attendees)
            c[4].metric("Seat utilisation", f"{sol.utilisation * 100:.1f}%",
                        f"{(sol.utilisation - base.utilisation) * 100:+.1f} pts")
            c[5].metric("Solver runtime", f"{sol.runtime_s:.1f} s of {sol.time_limit_s:g} s",
                        help="Time the solver used, out of the time limit set in the sidebar.")

            left, right = st.columns([3, 2])
            with left:
                st.subheader("Optimised vs baseline")
                def column(s):
                    return [money(s.total_cost), str(len(s.routes)), f"{s.utilisation * 100:.1f}%",
                            str(sum(len(r.stops) for r in s.routes))]

                summary = pd.DataFrame({
                    "": ["Total cost", "Buses", "Seat utilisation", "Schools served"],
                    "Baseline (one bus per school)": column(base),
                    "Optimised": column(sol),
                })
                st.dataframe(summary, hide_index=True, width="stretch")
                mix = pd.Series([r.bus_size for r in sol.routes]).value_counts().sort_index()
                st.caption("Bus mix: " + ", ".join(f"{n} x {s}-seat" for s, n in mix.items()))
            with right:
                st.subheader("How this was solved")
                st.markdown(f"- **Method:** {sol.method}")
                if cur["model"] != "fixed":
                    st.markdown(f"- **Route data:** {sol.matrix_source}")
                for note in sol.notes + cur["time_notes"]:
                    st.markdown(f"- {note}")
                st.markdown("- **Independent validation:** passed (capacity, no split schools, "
                            "every school served, every cost recalculated from route legs).")
                if sol.excluded_schools:
                    st.markdown(f"- Zero attendees, no bus: {', '.join(sol.excluded_schools)}")

# --------------------------------------------------------------------------- #
# Plan & map
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


def draw_map(sol, ds, paths=None):
    paths = paths or {}
    e = ds.event
    fmap = folium.Map(location=[51.66, -3.5], zoom_start=9, tiles="OpenStreetMap", scrollWheelZoom=False)
    info = ds.schools.set_index("school_id")
    legend_rows = []
    for k, r in enumerate(sol.routes):
        color = PALETTE[k % len(PALETTE)]
        coords = [(info.loc[s, "latitude"], info.loc[s, "longitude"]) for s in r.stops]
        road = paths.get(r.bus_id)
        if road is not None:
            folium.PolyLine(road.coords, color=color, weight=4, opacity=.85,
                            tooltip=f"{r.bus_id}: road path (OSRM), {road.miles:.1f} mi, "
                                    f"{road.hours * 60:.0f} min").add_to(fmap)
        else:
            folium.PolyLine(coords + [(e.latitude, e.longitude)], color=color, weight=3, opacity=.8,
                            dash_array="6 6",
                            tooltip=f"{r.bus_id}: straight line between stops (not road path)").add_to(fmap)
        for n, s in enumerate(r.stops, start=1):
            a = int(info.loc[s, "attendees"])
            folium.CircleMarker(
                location=coords[n - 1], radius=4 + math.sqrt(a), color=color, fill=True,
                fill_color=color, fill_opacity=.85, weight=1,
                tooltip=f"{s}: {a} attendees, {r.bus_id} pickup {n}/{len(r.stops)}",
                popup=folium.Popup(
                    f"<b>{s}</b><br>{a} attendees<br>{r.bus_id} ({r.bus_size}-seat), pickup {n} of "
                    f"{len(r.stops)}<br>Route: {' → '.join(r.stops)} → Stadium", max_width=260),
            ).add_to(fmap)
        legend_rows.append(
            f'<div><span style="display:inline-block;width:10px;height:10px;border-radius:50%;'
            f'background:{color};margin-right:6px"></span>{r.bus_id} · {r.bus_size} seats · {", ".join(r.stops)}</div>'
        )
    folium.Marker([e.latitude, e.longitude], tooltip="Principality Stadium (event)",
                  icon=folium.Icon(color="black", icon="star")).add_to(fmap)
    legend = (
        '<details style="position:fixed;top:12px;right:12px;z-index:9999;background:white;color:#222;'
        'padding:8px 10px;border-radius:8px;box-shadow:0 1px 4px rgba(0,0,0,.3);font-size:11px;'
        'max-height:420px;overflow-y:auto;max-width:320px">'
        f'<summary style="cursor:pointer;font-weight:600">Legend: {len(sol.routes)} buses</summary>'
        '<div style="margin:4px 0">Marker size = attendees · ★ Stadium<br>'
        'Solid line = road path (OSRM) · dashed = straight line, not road route</div>'
        + "".join(legend_rows) + "</details>"
    )
    fmap.get_root().html.add_child(folium.Element(legend))
    return fmap


with tabs[1]:
    if cur is None or not cur["result"].solution.has_plan:
        st.info("Run the optimiser to see a plan.")
    else:
        sol, ds = cur["result"].solution, cur["dataset"]
        if cur["model"] == "fixed":
            st.caption("Fixed model: cost ignores location. Pickup order shown is the shortest straight-line order, for display only.")
        paths = {}
        show_roads = st.toggle(
            "Draw road paths (OSRM)", value=False,
            help="Fetches real driving paths for each bus from the public OSRM server "
                 "(1 request per second, cached on disk). Display only: plan costs do not change.",
        )
        if show_roads:
            paths, road_errors = load_road_paths(sol, ds)
            st.caption(f"Road paths drawn for {len(paths)} of {len(sol.routes)} buses. "
                       "Road miles and times are shown for information; costs use the selected model's own measure.")
            if road_errors:
                st.warning("Shown as straight lines (no road path): " + "; ".join(road_errors))
        if HAS_MAP:
            try:
                st_folium(draw_map(sol, ds, paths), height=560, use_container_width=True, returned_objects=[])
            except Exception as exc:  # never let the map take the page down
                st.warning(f"Map unavailable ({exc}). Tables below show the full plan.")
        else:
            st.warning("Map libraries not installed. Tables below show the full plan.")

        st.subheader("Buses")
        buses = bus_table(sol)
        if paths:
            buses["Road miles (map)"] = buses["Bus"].map(lambda b: round(paths[b].miles, 1) if b in paths else None)
            buses["Road minutes (map)"] = buses["Bus"].map(lambda b: round(paths[b].hours * 60) if b in paths else None)
        st.dataframe(buses, hide_index=True, width="stretch",
                     column_config={"Utilisation %": st.column_config.ProgressColumn(
                         "Utilisation %", min_value=0, max_value=100, format="%.1f%%")})
        st.subheader("School cost shares")
        st.caption("Estimate, not an invoice: each bus's cost is split in proportion to attendees.")
        shares = cost_shares(sol, ds)
        st.dataframe(shares, hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.2f")
                                    for c in ("Bus cost", "Cost share", "Cost per attendee")})
        st.caption(f"Sum of shares: {money(shares['Cost share'].sum())} = total cost {money(sol.total_cost)} "
                   "(displayed values rounded to 2 decimals).")

# --------------------------------------------------------------------------- #
# Adjust plan: attendance and sharing constraints
# --------------------------------------------------------------------------- #

with tabs[2]:
    st.caption("Changes here are applied only when you press **Run optimiser**. The source CSV is never modified.")
    col_a, col_c = st.columns([2, 3])
    with col_a:
        st.subheader("Attendance")
        st.caption("Whole numbers only. 0 means the school needs no bus.")
        base_table = dataset.schools[["school_id", "attendees"]].copy()
        base_table["edited"] = base_table["attendees"]  # edits live in the editor's own state
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
            st.write("Edited: " + ", ".join(f"{k} → {v}" for k, v in ss.overrides.items()))
            st.write(f"Total attendees: {source_total} in the source data → {new_total} with your edits.")
        if ss.overrides or ss.attendance_errors:
            st.button("Reset attendance to source data", key="reset_attendance", on_click=reset_attendance)

    with col_c:
        st.subheader("Sharing constraints")

        def pair_editor(label: str, key: str, help_text: str):
            st.markdown(f"**{label}**")
            c1, c2, c3 = st.columns([2, 2, 1])
            c1.selectbox("School", school_ids, key=f"{key}_a", label_visibility="collapsed")
            c2.selectbox("School", school_ids, index=1, key=f"{key}_b", label_visibility="collapsed")
            c3.button("Add", key=f"{key}_add", help=help_text, on_click=add_pair, args=(key,))
            if ss.pop(f"{key}_msg", None):
                st.warning("Pick two different schools.")
            for x, y in list(ss[key]):
                r1, r2 = st.columns([4, 1])
                r1.write(f"{x} & {y}")
                r2.button("Remove", key=f"{key}_rm_{x}_{y}", on_click=remove_pair, args=(key, (x, y)))

        pair_editor("Must share a bus", "must_share", "Both schools ride the same bus.")
        pair_editor("Must NOT share a bus", "cannot_share", "The two schools ride different buses.")
        st.markdown("**Own bus (no sharing)**")
        ss.setdefault("solo_select", list(ss.solo))
        st.multiselect("Schools that travel alone", school_ids, key="solo_select", on_change=sync_solo,
                       label_visibility="collapsed")
        st.caption("Constraints are hard rules. If they conflict with each other or with bus capacity, "
                   "the optimiser reports why instead of ignoring them.")

# --------------------------------------------------------------------------- #
# Compare
# --------------------------------------------------------------------------- #

with tabs[3]:
    prev = ss.previous
    if cur is None or prev is None or not cur["result"].solution.has_plan:
        st.info("Run the optimiser, change attendance, constraints or the cost model, then run again. "
                "The last two valid plans are compared here.")
    else:
        a_sol, b_sol = prev["result"].solution, cur["result"].solution
        same_model = prev["model"] == cur["model"]
        st.markdown(f"**Previous valid plan:** {MODEL_LABELS[prev['model']]} &nbsp;→&nbsp; "
                    f"**Revised plan (latest run):** {MODEL_LABELS[cur['model']]}")
        st.caption("This compares two optimised plans. The one-bus-per-school baseline is on the Overview tab.")
        if not same_model:
            st.warning("Different cost models: costs are not directly comparable (the challenge notes this too).")
        c = st.columns(4)
        diff = b_sol.total_cost - a_sol.total_cost
        c[0].metric("Previous plan cost", money(a_sol.total_cost))
        c[1].metric("Revised plan cost", money(b_sol.total_cost), f"{diff:+,.2f}", delta_color="inverse")
        c[2].metric("Buses", len(b_sol.routes), f"{len(b_sol.routes) - len(a_sol.routes):+d}", delta_color="inverse")
        pct = (diff / a_sol.total_cost * 100) if a_sol.total_cost else 0.0
        c[3].metric("Cost change", f"{pct:+.1f}%")

        reasons = []
        att = attendance_changes(prev["dataset"], cur["dataset"])
        if att:
            reasons.append("Attendance changed: " + "; ".join(att))
        before_c, after_c = set(prev["constraints"].describe()), set(cur["constraints"].describe())
        if after_c - before_c:
            reasons.append("Constraints added: " + "; ".join(sorted(after_c - before_c)))
        if before_c - after_c:
            reasons.append("Constraints removed: " + "; ".join(sorted(before_c - after_c)))
        if not same_model:
            reasons.append("Cost model changed.")
        if not reasons:
            reasons.append("Inputs unchanged. Differences come from the solver search "
                           "(routing is heuristic, so repeated runs can find different plans of similar cost).")
        st.subheader("What changed in the inputs")
        for r in reasons:
            st.markdown(f"- {r}")
        st.subheader("Schools whose bus changed")
        changes = compare_plans(a_sol, b_sol)
        if changes.empty:
            st.success("Every school keeps the same bus-mates, bus size and pickup order.")
        else:
            st.dataframe(changes, hide_index=True, width="stretch")

# --------------------------------------------------------------------------- #
# Data & validation
# --------------------------------------------------------------------------- #

with tabs[4]:
    rep = dataset.report
    st.caption("Source data as loaded from the CSV files. Attendance edits from **Adjust plan** are not shown here.")
    c = st.columns(4)
    c[0].metric("Schools", len(dataset.schools))
    c[1].metric("Attendees (source data)", int(dataset.schools["attendees"].sum()))
    c[2].metric("Largest school (source data)", int(dataset.schools["attendees"].max()) if len(dataset.schools) else 0)
    c[3].metric("Cost models available", len(dataset.available_models))
    for e in rep.errors:
        st.error(e)
    for w in rep.warnings:
        st.warning(w)
    for m in dataset.available_models:
        if rep.capacity_errors.get(m):
            for msg in rep.capacity_errors[m]:
                st.error(f"[{m}] {msg}")
    if rep.ok and not any(rep.capacity_errors.values()):
        st.success("All files valid. Every school fits on at least one available bus.")

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Schools (pickup points)")
        st.caption("The data identifies schools by UK postcode outcode only; no school names are supplied.")
        st.dataframe(dataset.schools.drop(columns=["name"]), hide_index=True, width="stretch", height=360)
    with right:
        st.subheader("Bus options")
        sizes = sorted({s for r in dataset.rates.values() for s in r})
        bus_df = pd.DataFrame({"Seats": sizes})
        for m in COST_MODELS:
            if m in dataset.rates:
                header = {"fixed": "Per bus", "distance": "Per mile", "time": "Per hour"}[m]
                bus_df[header] = [dataset.rates[m].get(s) for s in sizes]
        st.dataframe(bus_df, hide_index=True, width="stretch")
        st.subheader("Event")
        st.write(f"{dataset.event.name}  \n{dataset.event.latitude}, {dataset.event.longitude} (from the challenge data README)")
        st.subheader("Assumptions")
        st.markdown(
            "- Bus size = passenger seats; attendance counts everyone who travels.\n"
            "- Unlimited buses of each size are available.\n"
            "- A bus starts at its first pickup and ends at the stadium; no return trip.\n"
            "- Distance model: straight-line (haversine) miles between consecutive stops.\n"
            "- Time model: OSRM road driving times when available; no dwell time at stops.\n"
            f"- Currency not stated in data; shown as {CURRENCY_LABEL}."
        )

# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #

with tabs[5]:
    if cur is None or not cur["result"].solution.has_plan:
        st.info("Only a validated plan can be exported. Run the optimiser first.")
    else:
        res, ds = cur["result"], cur["dataset"]
        overrides_text = attendance_changes(dataset, ds)
        st.download_button("Download plan (CSV)", plan_csv(res.solution, ds),
                           file_name=f"campus_cruise_{cur['model']}_plan.csv", mime="text/csv")
        report_md = plan_report(res.solution, res.baseline, ds, cur["constraints"], overrides_text)
        st.download_button("Download report (Markdown)", report_md,
                           file_name=f"campus_cruise_{cur['model']}_report.md", mime="text/markdown")
        with st.expander("Preview report"):
            st.markdown(report_md)
