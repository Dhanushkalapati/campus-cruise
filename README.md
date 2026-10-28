# Campus Cruise — Where Travel Gets Smarter

Optimisation tool for **Technical Challenge 2: Travel Organizer**. Schools send students to an
event at Principality Stadium, Cardiff. Campus Cruise decides which schools share which bus, and
which bus size each one uses, so the total transport cost is as low as possible. A school is never
split between buses.

Several kinds of AI work together, each doing what it is good at:

- **Constraint-based optimisation (Google OR-Tools)** makes every plan: an exact constraint solver
  (CP-SAT) and a vehicle-routing search (guided local search). It can prove the fixed-cost plan is
  the cheapest.
- **An optional language-model assistant (OpenAI or Anthropic, whichever key is configured)** turns
  an organiser's plain-English request into a structured proposal, which the app validates and shows
  for approval before anything changes. It explains results using only the app's own figures. It
  never makes or edits a plan itself.
- **An experimental machine-learning estimate** (ridge regression) of OSRM road times, for when road
  data is unavailable. The repository has **no observed journey times**, so it is trained and
  evaluated against OSRM's estimates and is never presented as a real journey-time predictor.

Campus Cruise is decision support: it proposes and checks plans, and organisers decide.

For a full technical walkthrough (algorithms, maths, data flow, results, test evidence and judge
questions), see [`PROJECT_TECHNICAL_REPORT.md`](PROJECT_TECHNICAL_REPORT.md).

## Results on the supplied data

There are 60 schools and 1,244 attendees. Each row compares the optimised plan with a baseline
where every school rents its own cheapest bus. Measured with `python benchmark.py` on 2026-10-24
(default limits: fixed 20 s, distance and time 300 s; Windows 11, Python 3.11.5):

| Cost model | Optimised | Buses | Baseline | Buses | Savings | Cost per attendee | Status | Runtime |
|---|---|---|---|---|---|---|---|---|
| Fixed cost per bus | 1,875.00 | 25 | 3,330.00 | 60 | 1,455.00 (43.7%) | 2.68 → 1.51 | **Proven optimal** | 18.8 s |
| Straight-line distance | 2,787.25 | 33 | 3,903.87 | 60 | 1,116.62 (28.6%) | 3.14 → 2.24 | Feasible, heuristic | 300.0 s |
| Travel time (OSRM road times, cached) | 6,667.72 | 37 | 8,220.26 | 60 | 1,552.54 (18.9%) | 6.61 → 5.36 | Feasible, heuristic | 300.0 s |

Bus distance (`python benchmark.py --offline`, 2026-10-25; the costs above were reproduced
exactly). Straight-line miles summed over buses, first pickup to stadium (not road mileage); hours
from cached OSRM road times; no fuel or emissions estimate:

| Cost model | Bus-miles baseline → plan | Saved | Bus-hours baseline → plan |
|---|---|---|---|
| Fixed cost per bus | 1,289.4 → 785.3 | 504.1 (39.1%) | 45.36 → 34.07 |
| Straight-line distance | 1,289.4 → 762.7 | 526.7 (40.8%) | 45.36 → 32.25 |
| Travel time | 1,289.4 → 856.6 | 432.8 (33.6%) | 45.36 → 32.34 |

The fixed plan's miles vary slightly between runs (785–792 seen): distance is only its tie-break.

Earlier 60 s runs gave 3,035.38 (22.2%) for distance and 7,048.85 (14.3%) for travel time. The
fixed run uses most of its 20 s because, after the cost is proven, the remaining time groups nearby
schools at the same cost.

Why 1,875 is optimal: 1,244 attendees need at least 25 buses (25 × 50 = 1,250 seats). A plan with
25 buses has enough seats only if all of them are 50-seaters, and a plan with 26 or more buses costs
more. The solver proves the same result.

The routing results (distance and time) come from a time-limited heuristic search. Each run can
return a slightly different plan. More time gave cheaper plans here, but that is not guaranteed.
They are valid plans, but they are not proven to be the cheapest possible.

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
of from scratch. This is the **Use previous plan as a starting point** option in the sidebar, on by default.
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

### Solver time limits

Defaults live in `DEFAULT_TIME_LIMITS` in `src/data.py` and are used by the dashboard, `demo.py` and
`benchmark.py` unless you pass your own limit:

| Model | Default | Behaviour at the limit |
|---|---|---|
| Fixed cost per bus | 20 s | Exact. The cost stage (70% of the limit) stops once optimality is proven; the time left groups nearby schools at the same cost, so a run usually uses most of the limit. If time runs out before the proof, the best plan found is returned as FEASIBLE with the lower bound. |
| Distance, travel time | 300 s | Heuristic. Guided local search uses the full limit, then returns the best valid plan found (FEASIBLE). |

- The dashboard has one slider per model (5–600 s). It starts at that model's default, and the value
  shown is the value passed to the solver. The sidebar caption repeats it.
- `optimise(..., time_limit=None)` uses the default; a limit of 0 or less is rejected.
- A longer limit gives the search more time. It does not guarantee a cheaper plan, and the reported
  results are the measured results of each run.
- For a quick live demo, lower the slider (for example to 30 s) and keep **Use previous plan as a starting point**
  ticked: the search then starts from the last valid plan and can only improve on its repaired version.

### Baseline, savings, cost sharing

- **Baseline:** every school rents its own cheapest bus that fits, under the same cost model, and
  drives straight to the stadium.
- **Savings %** = (baseline − optimised) / baseline × 100. Negative savings are shown as they are.
  A zero baseline shows "n/a".
- **School share** = bus cost × school attendees / bus attendees. This applies to per-mile and
  per-hour buses too, so being picked up first is not penalised. Shares are estimates, not
  invoices. They sum exactly to each bus cost; displayed values are rounded to 2 decimals.

### Impact metrics (Overview and Compare tabs)

All figures come from the latest validated run (`impact_metrics` in `src/report.py`):

- **Cost saving** = baseline cost − optimised cost, same model and inputs; **%** of the baseline.
- **Cost per attendee** = total cost ÷ attendees transported (everyone counted in the data). Shown
  for the baseline and the optimised plan.
- **Buses removed** = baseline buses − optimised buses. The baseline has one bus per school with
  attendees, so its size is counted from the data, not assumed to be 60.
- **Seat utilisation** = attendees ÷ seats on the buses used.
- The all-models table shows figures only for runs on the current attendance and constraints. Other
  runs are marked "Not run with current inputs".
- Impact figures are refused for an invalid plan, or when baseline and plan come from different inputs.
- **Bus-distance saved** = baseline total bus-miles − optimised total bus-miles (`distance_impact` in
  `src/explain.py`). Each bus is measured from its first pickup, through its stops, to the stadium,
  and the miles of all buses are summed. Both sides use the same straight-line (haversine) matrix, so
  they are comparable; they are **not road miles**. **%** is of the baseline; a zero baseline shows
  "n/a".
- **Bus driving hours** are summed the same way from the time model's own matrix, or from cached
  OSRM road times for the other models (no network request). If no travel-time data is available,
  hours are not shown rather than estimated.
- Fuel use and emissions are **not** estimated: the data has no vehicle or fuel information.

## Scenarios (Scenarios tab)

Test a change without touching the current plan (`src/scenario.py`):

- A scenario starts from the current, up-to-date plan: its attendance, organiser rules and cost model.
- Inputs: a percentage change of every school's attendance (whole percent, −90 to +200, each school
  rounded **up** to a whole person so there are always enough seats), optional per-school values, and a
  percentage change of the model's bus prices (−90 to +300, applied to the baseline as well).
- The scenario runs the same optimiser (warm-started from the current plan) and the same independent
  validator. Its result is stored separately and labelled *Scenario … — not the current plan*.
- The comparison table shows status, attendees, buses, total and baseline cost, cost per attendee,
  utilisation, bus-miles and driving hours for both, plus the schools whose bus changed.
- **Use this scenario as the plan** needs a confirmation tick. It copies the scenario's attendance into
  *Adjust plan* and shows the stored scenario result as the current plan, with no new search. The plan it
  replaced moves to *Compare*. A scenario with a **price change cannot be adopted**: prices are supplied
  data.
- Nothing is predicted: the organiser enters every number. Invalid inputs (decimals, unknown schools,
  values out of range) are refused with a reason; a school larger than the biggest bus gives *no plan*.

**Limits.** These are model outputs for the supplied (fictional) event, priced in cost units. Fuel,
carbon emissions, planning time and real money saved are not measured; they are listed as measures
for a proposed pilot. Costs from different models are not comparable.

## AI Transport Planning Assistant (optional)

The assistant accepts requests such as "Find the cheapest valid transport plan",
"CF15 now brings 25 people and SA18 brings 20", "CF15 and CF38 must share a bus", "Explain why this
plan is not proven optimal" or "Compare the cost and travel-time plans".

Open it with the **Assistant** button fixed at the bottom-right of every section (a native
`st.popover`; opening it never reruns anything). The panel also says what is sent to the language
model and lists this session's requests and their outcomes.

**Workflow** (`src/assistant.py`, UI in `app.py`):
1. The organiser types a request.
2. The configured model (OpenAI `gpt-4.1-mini` by default, or Anthropic `claude-opus-5-5`; strict
   JSON-schema structured output, no tools) returns a proposal: request type, summary, answer, cost
   model, attendance changes, sharing rules, clarification question, soft preferences.
   - Objectives map to the existing models: cheapest/most cost-effective → fixed; shortest journey →
     distance; fastest/travel time → time.
   - Soft preferences come from an allowlist and are never turned into hard rules.
     `group_nearby` is applied only for the fixed model, where it is the solver's real tie-break
     stage (nearby schools grouped among equally cheap plans; cost never rises). For routing models
     it is reported as not applied. `fewer_buses` is always reported as not applied: there is no
     bus-count objective, and the minimum possible bus count is shown instead.
3. `validate_proposal` checks it in code against the loaded data: request type in an allowlist; every
   school id exists; attendance is a whole number from 0 to the largest bus (a school is never split);
   rules have the right number of distinct schools and do not conflict with capacity or existing rules
   (`check_constraints`); the cost model exists.
4. Valid changes are shown as an exact table with **Approve and run** / **Reject**. Nothing changes before
   approval. Read-only answers need no approval.
5. On approval the proposal is re-checked against the current inputs, applied (attendance, rules, cost
   model), and the existing optimiser runs. The existing validator checks the plan; every tab refreshes.
6. The chat then gets a run summary **built by the app from the stored result, not by the language
   model**: objective, optimiser and settings, status and whether optimality was proven, buses, cost
   against the baseline, utilisation, total straight-line miles and driving hours (with their source),
   preferences applied or not, and approximation warnings. A run that errors, finds no plan or fails
   validation is reported as such; the approval message never claims the optimiser already ran.

| Request type | What happens |
|---|---|
| `run_optimiser` | Proposal to run a cost model on the current inputs |
| `attendance_change` | Proposal to set new head counts for named schools, then run |
| `sharing_rule` | Proposal to add hard rules (must share, must not share, own bus), then run |
| `explain`, `compare` | Answer from the app's figures. Withheld if it contains a number not in the plan data, or calls an unproven plan optimal |
| `grouping_preference` | Explained, never applied: the optimiser has no soft preferences, and a preference is never turned into a hard rule silently |
| `clarify` | The assistant asks a question (for example "which schools gain 50 students?") |
| `unsupported` | Refused with a reason (booking, emails, ignoring capacity, ...) |

Changes the model suggests for a read-only request are ignored and reported. The model cannot run
code, call tools or edit state; it only returns JSON.

**Configuration.** `openai` and `anthropic` are in `requirements.txt`. Set `OPENAI_API_KEY` (or
`ANTHROPIC_API_KEY`) in the environment, or copy `.streamlit/secrets.example.toml` to
`.streamlit/secrets.toml` (git-ignored). With both keys set, OpenAI is used unless
`CAMPUS_CRUISE_LLM_PROVIDER=anthropic`. `CAMPUS_CRUISE_LLM_MODEL` overrides the model. Without a key,
the chat and the tab say so and everything else works.

**Errors.** Missing key, rejected key, rate limit, timeout, no connection, refusal, cut-off answer and
invalid JSON each give a plain message; the app keeps working.

**Evaluation.** `tests/assistant_eval_cases.py` has 30 organiser requests: clear, ambiguous, unsupported,
invalid ids, invalid values, capacity breaches, rule conflicts, read-only questions, preferences,
attendance scenarios and adversarial model outputs.
- Offline (in the test suite, no key): scripted model outputs, including deliberately wrong ones, go
  through the real validation. Result: 30/30 handled as expected.
- Live: `python eval_assistant.py` sends the 30 requests to the configured model and reports
  request-type accuracy, correct approval offers and unsafe approvals. Last saved run
  (`demo_output/assistant_eval.json`, 2026-10-24 02:36, OpenAI `gpt-4.1-mini`): request type correct
  24/30, approval offered exactly when expected 29/30, **1 approval counted as unsafe** ("CF15 and CF38
  travel alone": the case expects a refusal; the live model's raw output was not saved, so what it
  proposed cannot be checked; any proposal shown still needed organiser approval). That run predates
  the soft-preference field; rerun it to measure the current version.

**Limits.** The grounding check is a heuristic (numbers and "optimal" wording), not a proof. The
assistant does not plan "what if" scenarios without named schools; it asks for them. It has no memory
between requests.

## When the optimiser runs (result freshness)

`src/freshness.py` fingerprints every input a plan depends on: cost model, attendance edits, sharing
rules and, for the time model, the travel-time source. The sidebar and every results tab say whether
the plan shown is **up to date**, **out of date** (inputs changed) or used a different time limit.

- Results are cached by fingerprint (last 20). Returning to inputs that were already optimised shows
  the stored result with no solver run.
- **Fixed model:** re-optimises automatically when inputs change (about 6 s), once per new set of inputs,
  after your first manual run. **Routing models:** show *Optimisation required* and wait for a click
  (300 s each). Both behaviours can be switched in the sidebar.
- Tabs, expanders, map filters and the chat never start a run. Solver errors and timeouts never
  replace the last valid plan.

## Explainability and audit

- **Overview → "Why this plan?"** (`src/explain.py: explain_plan`). Each line is either a *fact*
  computed from the plan and data (bus sizes vs the cheapest size that fits, bus mix and empty seats,
  price per seat, the seat-cover lower bound and whether the plan reaches it, the cost of shared
  schools if each had its own bus, organiser rules re-checked, savings) or an *interpretation*,
  labelled as such. The solver does not report why it chose a plan, so no causal claim is made as fact.
- **Compare → "All cost models run this session"** (`summary_row`): per model, baseline and optimised
  cost, change and %, buses before/after, attendees, seats, utilisation, status, proven optimal,
  runtime, limit, travel-time data, whether the data is approximate, total straight-line miles, and
  total driving hours from cached OSRM times (no network request). Costs across models are flagged
  as not comparable.
- **Export → audit report (Markdown) and model metrics (CSV)** (`audit_report`): executive summary,
  dataset and scenario, baseline method, optimisation approach, results and the explanation, a fresh
  run of the independent validator, solver status and optimality, limitations, responsible-AI and data
  notes, and the session's run log. Built from the same stored result as the dashboard; an out-of-date
  plan is marked as such; an invalid plan cannot be exported.
- **Run log** (Reports tab): every run attempt, including errors, with trigger (manual, auto,
  assistant), settings, status, runtime, cost, validation result and outcome. No request text or
  personal data.

## Interactive route map

Routes tab (`src/mapview.py` for the data, `draw_map` in `app.py` for folium):
- **Plan shown:** the current plan, or another model's latest valid run for comparison (labelled).
- **Filters:** pick one or more buses, or find a school (its bus is shown and the school is ringed).
  *Show all* returns to the full plan. Filters never run the optimiser.
- The map, the bus table and the cost-share table are built from the same filtered rows. A bus keeps
  its colour when filtered. With one bus selected: passengers/capacity, seats left, schools,
  straight-line miles and driving minutes (with source).
- Lines are dashed straight lines between stops unless **Road paths (OSRM)** is on (only the
  buses shown are fetched, 1 request/s, cached). If the map libraries fail, the tables still show
  the full plan.

## Machine learning: travel-time estimate (experimental)

**Data audit** (`data_readiness` in `src/ml_travel_time.py`, shown in the Data & checks tab):
the data has 60 pickup points with coordinates and head counts. There are **no observed journey
times, timestamps, traffic or weather records**. The only travel-time data is the cached OSRM table
(61 × 61 road-network *estimates*; 3,600 usable school → school/stadium pairs) and 68 cached road
paths. So no model of real journey time can be trained or evaluated.

**What is implemented:** a model that approximates OSRM's estimates from coordinates, as a better
offline fallback than the fixed-speed formula (miles × 1.3 / 35 mph).
- Features (all known before a trip): straight-line miles, √miles, each endpoint's straight-line
  miles to the stadium, destination-is-stadium flag. No feature uses OSRM output (no leakage).
- Candidates: the existing formula (baseline, not fitted); ridge on miles only; ridge on all features.
  numpy only, no new dependency; the artefact is plain JSON (`models/travel_time_ml.json`).
- Evaluation: 5-fold cross-validation **grouped by school** (held-out schools appear in no training
  pair), on all 3,600 pairs. Measured with `python train_travel_time.py`:

| Method | MAE (min) | RMSE (min) | MAPE (pairs > 5 min) |
|---|---|---|---|
| Fixed-speed formula (existing fallback) | 10.34 | 14.44 | 22.5% |
| Ridge, miles only | 7.63 | 11.49 | 19.6% |
| Ridge, all features (selected) | 7.25 | 11.17 | 17.0% |

  The target is OSRM's estimate, so these figures measure how well each method reproduces OSRM, not
  real journeys.
- **Use:** the time model's sidebar has a *Travel-time source* choice: OSRM (default), *ML estimate of
  OSRM (experimental)*, or the fixed-speed estimate. The ML option is used only if the saved artefact
  beat the formula on held-out schools; otherwise the formula is used and labelled. Every result
  states its source, and the validator re-checks every plan.
- **Retrain:** `python train_travel_time.py` (reads `data/cache/`, no network). The app never retrains.
- **Limits:** no traffic, time of day, weather or dwell time; tested only on this region's 60 points.
  Real journey logs would be needed for a genuine travel-time predictor.

## Data flows and external services

Verified from the code:

| Data | Where it goes |
|---|---|
| CSV inputs (outcode, one coordinate per outcode, head count; bus prices) | Read locally, never modified. No student names or individual records; no school names. |
| Attendance edits, constraints | Streamlit session memory only. |
| Pickup and stadium coordinates (time model, OSRM ticked, no cache) | Public OSRM server `router.project-osrm.org` (table service). Head counts are not sent. |
| Each bus's stop coordinates (map, **Road paths (OSRM)**) | Same OSRM server (route service). |
| Typed venue name or address (sidebar **Destination** search, only when used) | OpenStreetMap Nominatim `nominatim.openstreetmap.org` (max 1 request/s; answers cached in `data/cache/geocode/`). No school data. |
| OSRM answers | Cached in `data/cache/` (coordinates and times only; git-ignored). |
| Map tiles and map scripts | Loaded by the viewer's browser from OpenStreetMap and public CDNs. |
| Request text, school outcodes with head counts and bus assignments, bus prices, rules, plan figures (AI Assistant, only when used) | The configured LLM API (OpenAI or Anthropic). No coordinates, no names. |
| CSV / Markdown exports | Created when the user clicks download; `demo.py` and `benchmark.py` write to `demo_output/`. |

The only credential is the optional `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`, read from the environment or
`.streamlit/secrets.toml`; it is never displayed or logged. Streamlit's anonymous usage statistics are
switched off in `.streamlit/config.toml`. Untick OSRM in the sidebar to make no routing request.

## Responsible use, limitations and future work

- **Human oversight.** Organisers review, adjust and compare plans. Assistant proposals need explicit
  approval in the app (implemented). There is no built-in sign-off for the final transport plan:
  confirming the exported plan with schools and the bus company is a recommended procedure.
- **Explainability.** Every plan shows its status (proven optimal vs feasible), method, notes, each
  bus, and each school's cost share. Invalid plans are blocked from display and export.
- **Data protection.** Keep data at school level; the optimiser needs no personal data. This version
  has no access control, so put it behind a login before sharing real data. Delete `data/cache/` and
  exports after the event. A self-hosted routing server avoids sending coordinates to a public one.
- **Accessibility (proposed, not implemented).** An accessible-bus constraint needs data the challenge
  does not supply: passengers needing a wheelchair space per school, and accessible spaces per bus
  type. The optimiser and the independent check would then enforce it.
- **Proposed pilot.** Run alongside an organiser's usual planning for one event and compare cost,
  buses, time to plan, manual corrections, constraint problems caught, accessibility needs met and
  organiser feedback. No pilot has been run.
- **Wales.** The Responsible AI tab quotes three passages of the
  [National Events Strategy for Wales 2022–2030](https://www.gov.wales/national-events-strategy-wales-2022-2030-html)
  (cost to attend and inclusion; sustainable transport options; research and evaluation). This is the
  team's reading; there is no endorsement and no measured benefit for a real Welsh event.
- **Well-being of Future Generations (Wales) Act 2015.** The strategy says *"The starting point for what
  is 'good for Wales' is the Well-being Act."* Campus Cruise relates to two of the Act's seven goals:
  *A more equal Wales* (a lower, fairly shared cost to attend) and *A globally responsible Wales* (fewer
  vehicles; bus-miles saved are calculated, emissions are not).
- **Welsh language (not implemented).** The strategy measures *"Promotion and use of Welsh language at
  events"*. The interface is English only; a bilingual interface is a possible next step.

### Six Responsible AI principles

| Principle | How Campus Cruise applies it | Limit |
|---|---|---|
| Transparency | Status (proven optimal or best found), method, buses, costs and cost shares shown for every plan; estimates labelled; reports list assumptions | The solver does not say why it chose a grouping; readings are marked *Interpretation* |
| Privacy | Head counts per postcode outcode only, no names; coordinates only to OSRM; the LLM gets outcodes and counts, never coordinates; both optional | Public OSRM and LLM APIs are third-party services |
| Accountability | The organiser approves every AI-proposed change; run log and audit report record every run | Final booking sign-off is a recommended procedure, not a built-in step |
| Safety | Hard rules enforced by the optimiser and re-checked by separate code; failed plans never shown or exported | No real-world safety guarantee (drivers, roads, supervision are outside the model) |
| Security | Keys from the environment or a git-ignored secrets file, never shown or logged; the LLM has no tools and returns JSON that code validates | No login in this prototype |
| Fairness | Cost shared in proportion to attendees, whatever the pickup position; the same rules for every school | Accessible-transport needs are not in the data |

This describes the prototype's design; it is not a certification.

### Who could benefit (potential, not measured)

- **Event organisers and event SMEs:** a checked shared-bus plan in seconds; scenario tests before booking.
- **Schools and teachers:** lower transport cost and a clear per-school share; a group is never split from
  its supervisor.
- **Pupils:** a lower cost to attend.
- **Welsh coach and bus operators:** a clear booking list (sizes, stops, pickup order).
- **Venues and transport teams:** fewer vehicles to park and marshal.
- **Audiences and communities:** fewer vehicles near the event (emissions not measured).

No group has tested or endorsed Campus Cruise; no stakeholder feedback has been collected yet.

### Proposed pilot (not run)

- **Partner:** one Welsh event organiser or a local-authority school transport coordinator, for one event.
- **Information needed:** pickup points, head count per group, the bus company's prices per size,
  sharing and accessibility needs. No pupil names.
- **Process:** plan as usual and with Campus Cruise from the same inputs; compare before booking; the
  organiser chooses.
- **Measures:** transport cost, buses booked, road mileage, attendee allocation, planning time, manual
  corrections, constraint problems caught, accessibility needs met, organiser satisfaction.
- **Feedback:** short survey and a 20-minute interview after the event.
- **Privacy and consent:** written consent; school-level data only; deleted after the evaluation;
  results published only in aggregate and with permission.

## Generative AI use disclosure

The competition asks entrants to declare generative AI tools used to draft the submission or
presentation.

- **Claude Code (Anthropic; Claude Opus models)** was used to write and refactor code, write tests,
  audit the project against the brief, and draft documentation (this README,
  `PROJECT_TECHNICAL_REPORT.md`, the pitch deck). The team reviewed, ran and tested the output.
- **TEAM TO CONFIRM:** list any other tools used (for example for the EOI, slides or images) and how.

Separately, the app itself can call a language model (OpenAI or Anthropic) at run time; that is a
feature of the product, described in *AI Transport Planning Assistant*.

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

### Other schools and other venues

The optimiser is a general optimisation engine. It needs no retraining for new schools or a new venue.

- **School data (sidebar).** Upload a CSV with the same four columns as `attendants_dataset.csv`.
  A template can be downloaded beside the uploader.
  - The upload is used for this browser session only. The bundled CSV is never overwritten.
  - Uploads are stricter than the bundled file: any invalid row (missing or out-of-range coordinate,
    missing, negative, fractional or non-numeric attendance, missing outcode) rejects the whole file,
    with row numbers. No row is dropped silently. The data in use stays as it was.
  - Duplicate outcodes are an error. Two outcodes with identical coordinates are a warning.
  - **Use default school data**, or removing the file, returns to the bundled data.
- **Destination (sidebar).** Type a venue name, address or postcode and press **Find and use**. The
  best OpenStreetMap match becomes the destination, its coordinates are filled in, and a small map
  shows it. Other matches can be picked from a list. **Enter coordinates by hand** takes a name,
  latitude and longitude, or a one-row CSV (`name,latitude,longitude`). Changing only the name there
  also looks the name up. **Reset to Principality Stadium** restores the default.
- The venue is the last point of every matrix. Distance costs, OSRM requests, the time model,
  the baseline, independent validation, the map, the reports and the plan CSV all use it. The plan
  CSV gains `Destination`, `Destination latitude` and `Destination longitude` columns.
- Changing the schools or the venue clears plans, attendance edits and sharing rules, because they
  were made for the old inputs.
- **Routing.** The OSRM cache key is a hash of every point in order, including the venue, so other
  coordinates never reuse a cached matrix. A cache file whose stored points differ is ignored. With
  more than 99 schools (over 100 points with the venue) no OSRM request is sent. The time goal then
  uses the fixed-speed estimate, labelled as an estimate, and the sidebar says so.
- The experimental ML travel-time estimate was trained only on the 60 South Wales schools and
  Principality Stadium. With other inputs the app shows a warning that it is not validated there.

## Assumptions and limitations

- Bus size = passenger seats, and attendance counts everyone who travels.
- An unlimited number of buses of each size is available.
- Schools with 0 attendees need no bus.
- The routing models are heuristic: valid plans, improved for the full time limit, but not proven
  minimal. Repeated runs can differ slightly.
- The fixed model can return FEASIBLE when the time limit is too short to prove optimality. In
  that case the gap to the lower bound is reported. Example: the demo attendance change gives
  1,950 against a bound of 1,935 within 60 s.
- By default, map lines are straight dashed lines between stops. The **Road paths (OSRM)**
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
python benchmark.py           # one run per model at the default limits (about 11 min)
python eval_assistant.py      # live assistant evaluation (needs an API key; ~30 API calls)
python train_travel_time.py   # retrain the experimental ML travel-time estimate (no network)
python demo.py                # CLI demo: all models, attendance change, re-optimisation (about 21 min)
python demo.py --offline      # demo without OSRM (labelled estimate for the time model)
python demo.py --time-limit 30  # same limit for every model, faster
python -m pytest -q           # full test suite
```

`benchmark.py` writes `demo_output/benchmark.md` and `benchmark.json` (status, limit, runtime, costs,
buses, cost per attendee for every model). `demo.py` writes CSV plans and Markdown reports to
`demo_output/`. To reproduce the results table above, run `python benchmark.py`; routing results can
differ slightly between runs and machines.

## Using the dashboard

The header shows one status for the whole page: *Plan up to date*, *Out of date*, *Settings changed*,
*No valid plan* or *No plan yet*. Technical detail (method, solver status meaning, search time, route
data, validation rules) is in expandable sections, so the first screen shows results only.

1. **Sidebar (Inputs, then Plan settings).**
   - **School data:** upload a school CSV or keep the default data. A preview and counts are shown.
   - **Destination:** venue name and coordinates (default Principality Stadium). Below it, a line
     says whether road times are cached, will be fetched from OSRM, or will be estimated.
   - Pick an **optimisation goal**: fixed price per bus, price per mile, or price per driving hour.
   - **Use previous plan as a starting point** (warm start) appears after the first run.
   - **Update automatically when inputs change** (on for the fixed-price goal).
   - Press **Run optimiser**.
   - **Advanced settings:** search time limit and, for the time goal, the travel-time source and OSRM.
2. **Overview.** Four cards: attendees, buses, total cost (with the baseline) and savings. Below
   them, a baseline-vs-optimised table (cost, buses, seat utilisation, cost per attendee, bus-miles and,
   when travel-time data is available, bus driving hours) and a short
   "what this plan means". The search time appears only in the small run caption and in
   *How this plan was created*. Expanders: *Why this plan?*, *How this plan was created*,
   *Validation and constraints*.
3. **Routes.**
   - The map opens centred on the plan's schools and the stadium. The view is computed in Python
     (`map_view` in `src/mapview.py`), because Leaflet's own fitting zooms to street level when
     the map sits in a hidden tab.
   - Each bus has its own colour; marker size shows attendees. Filter by bus or school.
   - Beside the map: the list of buses shown, or the details of one selected bus. With three
     buses or fewer, pickups are numbered on the map.
   - **Road paths (OSRM)** replaces dashed straight lines with real road routes.
   - Below: the bus details table and, in an expander, the school cost shares.
4. **Adjust plan.**
   - Edit attendance. Values must be whole numbers ≥ 0; 0 means the school needs no bus. An empty
     cell, a decimal or a negative value is shown as an error, and the optimiser will not run
     until it is fixed or reset.
   - Edits reach the optimiser even if you click **Run optimiser** straight after typing.
   - Add "must share", "must not share" and "own bus" constraints.
   - Fixed model: changes are re-optimised automatically. Routing models: an *Optimisation required*
     banner says the results shown do not include your edits until you run again.
5. **Compare.** One table of every cost model you have run (status, costs, saving, buses, cost per
   attendee); models not run are listed as *Not evaluated yet* and never run on their own. Then the
   previous and current valid plans: cost difference, input changes and every school whose bus
   changed. Solver details for every run are in *More metrics and solver details*.
6. **Scenarios.** Test more attendees or different prices as a separate scenario; compare it with the
   current plan; adopt it (attendance only, after a confirmation tick) or discard it.
7. **Assistant** (the button at the bottom-right of every section). Plain-English requests
   become validated proposals; approve or reject them. Optional: needs an OpenAI or Anthropic key.
8. **Responsible AI.** Three principles at a glance, the six Responsible AI principles table, then
   expanders: who could benefit, how plans are made and checked, what each solver status means, human
   oversight, data protection, the proposed accessibility constraint, relevance to events in Wales (the
   National Events Strategy and the Well-being of Future Generations Act) and the proposed pilot.
9. **Data & checks.** File checks, source-data figures, schools and bus options; expanders for
   assumptions and the travel-time / machine-learning audit.
10. **Reports.** Download the plan (CSV), plan report, audit report and model metrics; preview them;
   see the run log. Both reports include bus-miles, approximation warnings and Responsible AI notes.

**Hackathon demonstration (about 5 minutes, all on the supplied data):**
1. **Optimiser.** Fixed model, 20 s, **Run optimiser** → 1,875.00, 25 buses, *Proven optimal*. Open
   **Why this plan?**: the cost equals the seat-cover lower bound, so no plan can cost less.
2. **Map.** Routes → *Find a school* → CF15: its bus is selected, ringed and summarised; the
   tables match the map. *Show all* returns to the full plan.
3. **Change + freshness.** Adjust plan → set CF15 to 25 and SA18 to 20. The fixed model re-optimises by
   itself; Compare shows the cost change and the schools that moved.
4. **Assistant** (needs a key; check it beforehand). Bottom-right **Assistant** → type "NP11 now brings
   15 people." → proposal table → **Approve and run** → the app's run summary with real
   figures. Then ask "NP11 brings 80 people" → refused (a school group cannot be split).
5. **Audit.** Reports → download the audit report; show the run log.
6. **ML honesty.** Data & checks → *Travel-time data and machine learning*: 0 observed journeys;
   the experimental model approximates OSRM (MAE 7.25 vs 10.34 min for the old formula, held-out
   schools). Do not run a 300 s routing search live; show `demo_output/benchmark.md` instead.

Measured claims you can make: the fixed optimum (proven), the benchmark table above (one run each),
the ML evaluation table (against OSRM, not real journeys), and the test count from `pytest`. Not
measured: real-world cost, emissions, traffic accuracy, planning time.

## Project structure

```
app.py               Streamlit UI only
demo.py              CLI demonstration on the real data
benchmark.py         reproducible one-run-per-model benchmark -> demo_output/benchmark.md
eval_assistant.py    live evaluation of the AI Assistant (needs an API key)
train_travel_time.py retrain the experimental ML travel-time estimate -> models/travel_time_ml.json
src/assistant.py     AI Assistant: OpenAI/Anthropic providers, context, proposal validation, grounding check
src/freshness.py     input fingerprints, stale-result detection, auto-run rules, result cache
src/explain.py       plan explanations, all-model summary, audit report, assistant run summary
src/mapview.py       map filters and per-bus details (map and table share the same rows)
src/ml_travel_time.py  data-readiness audit, OSRM-approximating ridge model, grouped CV, safe loading
models/              trained ML artefact (JSON, no pickled code)
PROJECT_TECHNICAL_REPORT.md  full technical analysis for judges
presentation/        pitch deck (ai project.pptx, speaker notes inside) and its backup copy;
                     build/build_ai_project.js rebuilds it
.streamlit/          Streamlit settings (usage statistics off)
src/data.py          loading, Dataset, attendance overrides
src/validation.py    input validation and independent solution checks
src/distances.py     haversine, OSRM matrix + cache, labelled offline estimate
src/solvers.py       CP-SAT fixed model, OR-Tools routing, baseline, constraint checks
src/planner.py       one entry point: matrices -> solve -> baseline
src/report.py        savings, cost shares, plan comparison, CSV/Markdown export
src/scenario.py      scenarios: attendance and price changes, comparison table
src/model.py         Solution, BusRoute, Constraints types
tests/               pytest suite, including warm start, road paths, attendance re-optimisation and a
                     headless dashboard flow (AppTest); synthetic data is labelled as such
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
- solver time limits reaching each solver; impact metrics; the Impact tab after attendance edits
- bus-distance savings, price what-ifs, scenario inputs and the Scenarios tab (isolation, adoption,
  price scenarios that cannot be adopted)
- the AI Assistant: 30-case evaluation set, malformed model output, provider errors, approve and
  reject flows in the dashboard, and the dashboard without an API key (all with a fake model)
