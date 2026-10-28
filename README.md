# Campus Cruise — Where Travel Gets Smarter

Optimisation tool for **Technical Challenge 2: Travel Organizer**. Schools send students to an
event at Principality Stadium, Cardiff. Campus Cruise decides which schools share which bus, and
which bus size each one uses, so the total transport cost is as low as possible. A school is never
split between buses.

The intelligence is mathematical optimisation (Google OR-Tools). The project has no language model
or machine-learning component.

## Results on the supplied data

There are 60 schools and 1,244 attendees. Each row compares the optimised plan with a baseline
where every school rents its own cheapest bus. The runs used a 60 s time limit (`python demo.py --time-limit 60`).

| Cost model | Optimised | Buses | Baseline | Buses | Savings | Status |
|---|---|---|---|---|---|---|
| Fixed cost per bus | 1,875.00 | 25 | 3,330.00 | 60 | 1,455.00 (43.7%) | **Proven optimal** |
| Straight-line distance | 3,035.38 | 39 | 3,903.87 | 60 | 868.49 (22.2%) | Feasible, heuristic |
| Travel time (OSRM road times) | 7,048.85 | 43 | 8,220.26 | 60 | 1,171.41 (14.3%) | Feasible, heuristic |

Why 1,875 is optimal: 1,244 attendees need at least 25 buses (25 × 50 = 1,250 seats). A plan with
25 buses has enough seats only if all of them are 50-seaters, and a plan with 26 or more buses costs
more. The solver proves the same result.

The routing results (distance and time) come from a time-limited heuristic search. Each run can
return a slightly different plan, and longer time limits usually find cheaper ones. They are valid
plans, but they are not proven to be the cheapest possible.

The data does not state a currency, so all costs are shown in **cost units**.

## Problem formulation

**Inputs:**
- Schools *i* with attendees *aᵢ* and coordinates.
- Bus sizes *t* ∈ {15, 35, 50} with a rate for each cost model.
- The event location: 51.478283, -3.182652, taken from the challenge README.

**Hard rules:**
- Every school with attendees > 0 rides exactly one bus. Schools are never split.
- The attendees on a bus never exceed its size.
- Organiser constraints (must share, must not share, own bus) are always satisfied, or the app
  explains why they cannot be.

**Route accounting** (from the challenge data README):
- A bus starts at its first pickup. Nothing is charged before it.
- The bus may stop at more schools, then ends at the stadium.
- No return trip is charged. Dwell time at stops is not modelled.
- Route cost = rate(bus size) × the sum of the legs `first school → … → last school → stadium`,
  measured in miles or hours.
- Leg direction matters, because road times are asymmetric.

### Model 1: fixed cost per bus (exact)

This is mixed-size bin packing, solved with CP-SAT using a symmetry-free "leader" formulation.
Schools are sorted largest first. Bus slot *j* exists only when school *j* is its lowest-index
passenger, so `x[i,j]` is defined only for *j ≤ i*.

```
min  Σⱼ Σₜ costₜ · z[j,t]
s.t. Σ_{j≤i} x[i,j] = 1                       ∀i   (each school on exactly one bus)
     Σₜ z[j,t] = x[j,j]                       ∀j   (slot used iff its leader rides it)
     x[i,j] ≤ x[j,j]
     Σ_{i≥j} aᵢ x[i,j] ≤ Σₜ t · z[j,t]        ∀j   (capacity)
     Σₜ t · countₜ ≥ Σᵢ aᵢ,  cost ≥ LB        (redundant cuts)
```

**LB** is the seat-cover lower bound: the cheapest set of buses that seats everyone, computed
exactly by dynamic programming. When a plan reaches LB, it is optimal and the search stops.

A second, lexicographic stage then keeps the optimal cost fixed and minimises the miles between
each school and the leader of its bus. This groups nearby schools without changing the cost.

The status is OPTIMAL only when it is proven. A plan found at the time limit is labelled
FEASIBLE and shows the best lower bound.

### Models 2 and 3: distance and travel time (heuristic routing)

These are open capacitated vehicle routing problems with mixed bus sizes, solved with OR-Tools
Routing.

- **Virtual start node.** Every bus leaves a virtual start node through zero-cost arcs to every
  school. This is not a real depot: it is the exact encoding of "the bus starts at its first
  pickup". All routes end at the stadium.
- **Arc costs.** Each bus size has its own cost callback, rate × leg.
- **Fleet.** The number of buses of each size is not limited in practice: there is one vehicle per
  school that fits that size.
- **Constraints.** Must share / must not share / own bus are equalities and inequalities on the
  vehicle variables.
- **Search.** The search starts from a valid plan, with each must-share group on its own cheapest
  bus. Guided local search then improves it until the time limit.
- **Status.** The result is always reported as FEASIBLE (heuristic), never as optimal.
- **Distance metric.** Haversine miles, which are straight-line distances, not road distances.
- **Time metric.** OSRM public `table` service gives real road driving times in a single 61×61
  request, cached in `data/cache/`. If OSRM is unreachable or returns a bad matrix, the app can use
  an estimate instead: straight-line miles × 1.3 ÷ 35 mph. This estimate is always labelled
  **APPROXIMATE (not road routing)**.

### Re-optimising from the previous plan (warm start)

After you change attendance or constraints, the next run can start from the last valid plan instead
of from scratch. This is the **Start from previous plan** option in the sidebar, on by default.
`repair_plan` in `src/solvers.py` first makes the old plan valid for the new inputs:
- drops schools that now have 0 attendees
- gives each must-share group, and each solo school that was sharing, its own bus
- splits must-not-share pairs
- drops the last pickups from buses that are now over capacity
- gives every new or removed school its own bus

The repair note shows how many schools kept their bus and which were re-placed.

- **Routing models:** the repaired plan is the starting solution. The search keeps the best plan it
  finds, so the result never costs more than the repaired start. Schools change bus only where the
  edit requires it or where the search finds a real saving. With unchanged inputs, a re-run never
  gets worse.
- **Fixed model:** the repaired plan is a CP-SAT hint. It speeds up the search but never changes
  the optimum or the status rules.

Warm start makes consecutive runs consistent. It does not make the heuristic exact: a re-run can
still find savings that the previous run missed.

### Independent validation

Every plan is re-checked by `src/validation.py` without trusting the solver:
- every school is served once and none is split
- capacities are respected
- bus sizes exist
- organiser constraints hold
- every route cost is recomputed from its legs, and the total matches

A plan that fails validation is marked INVALID and cannot be exported.

### Baseline, savings, cost sharing

- **Baseline:** every school rents its own cheapest bus that fits, under the same cost model, and
  drives straight to the stadium.
- **Savings %** = (baseline − optimised) / baseline × 100. Negative savings are shown as they are.
  A zero baseline shows "n/a".
- **School share** = bus cost × school attendees / bus attendees. This applies to per-mile and
  per-hour buses too, so being picked up first is not penalised. Shares are estimates, not
  invoices. They sum exactly to each bus cost; displayed values are rounded to 2 decimals.

## Data schema and units

| File | Columns | Units |
|---|---|---|
| `attendants_dataset.csv` | outcode, latitude, longitude, attendants | WGS84 degrees, people |
| `fixed_costs.csv` | Bus size, Fixed price | seats, cost units per bus |
| `direct_distance_costs.csv` | Bus size, Price per mile | seats, cost units per straight-line mile |
| `time_costs.csv` | Bus size, Price per hour | seats, cost units per hour |

Schools are identified by postcode outcode, because the data has no school names. The source CSVs
are read-only. Attendance edits in the app apply to a copy.

**Validation rules:**
- Rows with missing or invalid coordinates, or with negative or fractional attendance, are
  excluded with a warning.
- Duplicate school IDs block optimisation.
- Negative or invalid prices and bus sizes are excluded.
- A school larger than every bus makes the model infeasible, and the app explains why.

## Assumptions and limitations

- Bus size = passenger seats, and attendance counts everyone who travels.
- An unlimited number of buses of each size is available.
- Schools with 0 attendees need no bus.
- The routing models are heuristic: valid plans, improved for the full time limit, but not proven
  minimal. Repeated runs can differ slightly.
- The fixed model can return FEASIBLE when the time limit is too short to prove optimality. In
  that case the gap to the lower bound is reported. Example: the demo attendance change gives
  1,950 against a bound of 1,935 within 60 s.
- By default, map lines are straight dashed lines between stops. The **Draw road paths (OSRM)**
  toggle fetches real driving paths for each bus and draws them as solid lines.
  - The first fetch takes about 1 s per bus, because the public server allows 1 request per second.
    Paths are cached in `data/cache/routes/` after that.
  - Road miles and minutes for each path are added to the bus table for information only. Plan
    costs always use the selected model's own measure.
  - A bus whose path cannot be fetched stays a dashed straight line, and the app shows a warning.
- OSRM's public demo server is rate-limited and best-effort. Results are cached after the first
  successful call.
- Costs from different cost models are not comparable with each other (the challenge notes this
  too).

## Setup (Windows PowerShell)

The project uses its own virtual environment, `.venv`. Nothing is installed globally.

```powershell
cd C:\Users\kalap\OneDrive\Desktop\2026_ai_pollination_student_competition_challenge_2
python -m venv .venv                      # skip if .venv already exists
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks the activation script, run this once:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.
You can also skip activation and call `.\.venv\Scripts\python.exe` directly.

Linux/macOS: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`

## Run

```powershell
streamlit run app.py          # dashboard at http://localhost:8501
python demo.py                # CLI demo: all models, attendance change, re-optimisation
python demo.py --offline      # demo without OSRM (labelled estimate for the time model)
python -m pytest -q           # 104 tests
```

`demo.py` writes CSV plans and Markdown reports to `demo_output/`.

## Using the dashboard

1. **Sidebar.**
   - Pick a cost model and a time limit.
   - For the time model, choose whether to use OSRM.
   - After the first run, **Start from previous plan** appears (warm start).
   - Press **Run optimiser**.
2. **Overview.** Shows total cost, savings against the baseline, buses, attendees, seat
   utilisation, solver status, method and notes.
3. **Plan & map.**
   - Each bus route has its own colour. Marker size shows attendees.
   - The **Draw road paths** toggle replaces straight lines with real road routes.
   - Clicking a marker shows the school, its bus and its pickup number.
   - The legend lists every bus.
   - The tab also has the bus table and the school cost-share table.
4. **Adjust plan.**
   - Edit attendance.
   - Add "must share", "must not share" and "own bus" constraints.
   - The sidebar shows which changes are not applied yet. Nothing changes until you rerun.
5. **Compare.** Shows the previous and current valid plans: cost difference, input changes and
   every school whose bus changed.
6. **Data & validation.** Shows file checks, warnings, bus options, the event and assumptions.
7. **Export.** Download the validated plan as CSV, or a full Markdown report with assumptions,
   status, allocation, savings and cost shares.

**Demonstration workflow:**
1. Run the Fixed model: 1,875, proven optimal.
2. In **Adjust plan**, set CF15 to 25 and SA18 to 20, then rerun. The result is 1,950 with 26 buses.
3. Open **Compare** to see +75 and the regrouped schools.
4. Repeat with the distance or time model.

## Project structure

```
app.py               Streamlit UI only
demo.py              CLI demonstration on the real data
src/data.py          loading, Dataset, attendance overrides
src/validation.py    input validation and independent solution checks
src/distances.py     haversine, OSRM matrix + cache, labelled offline estimate
src/solvers.py       CP-SAT fixed model, OR-Tools routing, baseline, constraint checks
src/planner.py       one entry point: matrices -> solve -> baseline
src/report.py        savings, cost shares, plan comparison, CSV/Markdown export
src/model.py         Solution, BusRoute, Constraints types
tests/               pytest suite, including warm start and road paths (synthetic data is labelled as such)
```

Tests cover these areas:
- brute-force checks of the fixed solver on random small instances, with and without constraints
- a brute-force check of the routing solver on a 5-school instance
- route-leg accounting, including asymmetric legs
- baseline and savings maths
- cost shares
- malformed, empty and invalid data
- capacity infeasibility and conflicting constraints
- timeouts that never claim optimality
- OSRM failures and fallback, using fake sessions with no network
- export of valid plans, and refusal to export invalid ones
