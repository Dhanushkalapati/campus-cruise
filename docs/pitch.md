# Campus Cruise: 7-minute final pitch

Figures come from `python benchmark.py` on the supplied challenge data, run 2026-10-24 (fixed
model 20 s limit, routing models 300 s limit). Costs are in cost units: the data states no
currency. Rerun the benchmark before the pitch and update this file if the numbers change.

| Model | Status | Optimised | Buses | Baseline | Buses | Saving | Cost per attendee |
|---|---|---|---|---|---|---|---|
| Fixed price per bus | **Proven optimal** | 1,875.00 | 25 | 3,330.00 | 60 | 1,455.00 (43.7%) | 2.68 → 1.51 |
| Straight-line distance | Best found in 300 s | 2,787.25 | 33 | 3,903.87 | 60 | 1,116.62 (28.6%) | 3.14 → 2.24 |
| Road travel time (OSRM) | Best found in 300 s | 6,667.72 | 37 | 8,220.26 | 60 | 1,552.54 (18.9%) | 6.61 → 5.36 |

Baseline = every school rents its own cheapest bus. Costs from different models are not comparable.

Bus distance from the 2026-10-25 benchmark run (`python benchmark.py --offline`, cached OSRM hours;
the costs above were reproduced exactly). Miles are straight-line, summed over buses, first pickup
to stadium; not road mileage; no emissions figure.

| Model | Bus-miles baseline → plan | Saved | Bus-hours baseline → plan |
|---|---|---|---|
| Fixed price per bus | 1,289.4 → 785.3 | 504.1 (39.1%) | 45.36 → 34.07 |
| Straight-line distance | 1,289.4 → 762.7 | 526.7 (40.8%) | 45.36 → 32.25 |
| Road travel time | 1,289.4 → 856.6 | 432.8 (33.6%) | 45.36 → 32.34 |

The fixed plan's bus-miles vary slightly between runs (785–792 seen), because its time-limited
second stage only tie-breaks on distance; its cost (1,875) never changes.

---

## Slides (10) and speaker script

Timing adds up to 7:00. Speak about 130 words a minute.

### Slide 1 — Campus Cruise (0:00–0:20)

**On screen:** name, tagline "Where travel gets smarter", team names, map of the 60 pickup points.

**Say:** "We're Campus Cruise. We help event organisers decide which schools share a bus, so that
getting pupils to an event costs less and takes less guesswork."

### Slide 2 — The problem (0:20–0:55)

**On screen:** 60 schools · 1,244 attendees · buses of 15, 35 and 50 seats · one stadium in Cardiff.
Rules: a school group is never split; no bus over capacity.

**Say:** "Sixty schools are sending 1,244 people to Principality Stadium. The simple plan is one bus
per school: 60 buses. Sharing is cheaper, but there are far too many ways to group 60 schools to try
by hand, and each group must stay together because one adult supervises it."

### Slide 3 — Campus Cruise and three cost models (0:55–1:35)

**On screen:** three columns.
- Fixed price per bus → exact solver (CP-SAT) → can prove the cheapest plan.
- Price per straight-line mile → vehicle-routing search.
- Price per hour of road driving time (OSRM) → vehicle-routing search.

**Say:** "The challenge prices buses three ways. For a fixed price per bus we use an exact constraint
solver that can prove no cheaper plan exists. When the price depends on miles or driving time, the
pickup order matters, so a vehicle-routing search improves the plan for five minutes. We label those
honestly: best found, not proven cheapest."

### Slide 4 — Live demo (1:35–3:05)

**On screen:** the dashboard (see the demo checklist).

**Say while clicking:**
1. "Fixed-price plan: 25 buses instead of 60, status *proven optimal*."
2. "On the map each colour is one bus. Click a school: its bus, pickup order and cost share."
3. "Now a change: CF15 brings 25 people and SA18 brings 20. The fixed model re-optimises by itself
   in a few seconds, and Compare shows the new cost and which schools moved." (Routing models instead
   show *Optimisation required* and wait for a click.)
4. "Scenarios: plus 10% attendance at every school, as a separate scenario. The current plan stays
   untouched until I tick the box and adopt it."

### Slide 5 — The AI Assistant: request → approval → optimiser (3:05–3:50)

**On screen:** flow diagram: organiser request → language model (OpenAI GPT-4.1 mini) proposes (structured JSON) → app validates
against the data → organiser approves or rejects → OR-Tools optimises → independent validator → results.

**Say (with the live assistant, or clip D):** "Organisers can also just type: 'NP11 now brings 15
people.' A language model turns that into a proposal, but it changes nothing. The app checks
every school and number against the data, shows the exact change, and waits. Only when I approve does
our optimiser run, and its plan is checked like any other. Ask it to put 60 people on one bus and it
refuses, because a school group can't be split."

### Slide 6 — Results (3:50–4:30)

**On screen:** the results table above, fixed row highlighted.

**Say:** "With a fixed price per bus, the plan costs 1,875 instead of 3,330, a 43.7% saving, with 25
buses instead of 60. That is provably the cheapest: 1,244 people need at least 25 fifty-seat buses.
For the distance model we save 28.6%, and for road travel time 18.9%. Those two are the best plans
found in five minutes, not proven minimums."

### Slide 7 — Trust: optimality and validation (4:30–5:05)

**On screen:** "Proven optimal" vs "Best found" · independent validator · warm start · 293 automated tests.

**Say:** "Every plan, whoever proposes it, is re-checked by separate code: every school on exactly one
bus, no bus over capacity, every cost recalculated. A plan that fails can't be shown or exported. And
when inputs change, the search starts from the previous plan, so schools only move when they must."

### Slide 8 — Responsible AI (5:05–5:45)

**On screen:** four boxes.
- **Human oversight:** assistant proposals need approval; organisers adjust, compare and sign off.
- **Explainable:** every plan shows its status, method, buses and each school's cost share; assistant answers are checked against the plan figures.
- **Privacy:** school-level head counts only, no student names. Routing gets coordinates; the assistant gets outcodes and counts, never coordinates. Both are optional.
- **Accessibility (next step):** accessible-bus constraint, needs data the challenge does not include.

Small strip under the boxes: the six principles the brief names (Transparency, Privacy,
Accountability, Safety, Security, Fairness), each ticked to one feature. The full table is in the
Responsible AI tab and the README.

**Say:** "Campus Cruise is decision support, not an autonomous system. The language model interprets
and explains; the optimiser decides; people approve. We only use school-level counts, and we tell
users exactly what leaves the computer. Next: if organisers record who needs a wheelchair space,
the solver can guarantee those groups an accessible bus."

### Slide 9 — Who benefits, Wales, pilot and limits (5:45–6:40)

**On screen:** beneficiaries row (event SMEs · schools and teachers · pupils · Welsh coach operators ·
venues · communities); two quotes from the National Events Strategy for Wales 2022–2030 (below) and
the Well-being Act link; bus-miles saved (fixed plan: 1,289 → 785 straight-line bus-miles, 39% fewer); pilot plan; limits.

**Say:** "Who gains? Event organisers get a checked plan in seconds, schools and pupils a lower cost
to attend, Welsh coach operators a clear booking list, venues fewer coaches to park. The National
Events Strategy for Wales wants events that consider *cost to attend*, tracks *sustainable transport
options*, and starts from the Well-being of Future Generations Act. On the challenge data the fixed
plan also cuts total bus-miles by about 39%, from about 1,290 to about 790 — straight-line miles, and we don't claim an emissions
figure. Nobody has tested this at a real event yet, so we propose a pilot with one Welsh organiser:
same inputs, both plans, compare cost, buses, mileage, planning time and feedback, with consent and
school-level data only."

### Slide 10 — Close (6:40–7:00)

**On screen:** "25 buses instead of 60 · proven cheapest · every plan checked · you approve."

**Footer (small):** AI-use disclosure — see the slide notes and README (*Generative AI use disclosure*).

**Say:** "Campus Cruise: fewer buses, lower cost, and plans organisers can trust. Thank you."

**Quotes (verified on gov.wales on 2026-10-24):**
- "A balanced portfolio will consider age, region, ethnicity, religion, cost to attend and so on to
  ensure the events on offer are widely inclusive and accessible."
- Success measure: "Percentage of events encouraging or offering active and sustainable transport options."

Source: https://www.gov.wales/national-events-strategy-wales-2022-2030-html. Do not imply Welsh
Government endorsement.

---

## Likely judge questions

- **"Where is the AI?"** Two parts. Constraint optimisation and search, core AI and operations-research
  methods, make the plans. A language model (OpenAI GPT-4.1 mini) understands organisers' requests and explains
  results, inside strict guardrails. Neither works unchecked.
- **"Can the language model break a rule?"** No. It returns JSON only, with no tools. Code checks every
  id, number and rule; the organiser approves; the optimiser and validator still enforce capacity and
  no-split.
- **"How accurate is the assistant?"** Offline, 30 test requests, including deliberately wrong model
  outputs, are all handled correctly by the guardrails. Live with GPT-4.1 mini (one run, 30 requests):
  request type right 24/30, approval offered exactly when expected 29/30, one approval offer counted
  unsafe by our strict scorer; even then nothing changes without the organiser clicking Approve.
- **"Is 1,875 really the minimum?"** Yes: 1,244 attendees need at least 25 buses (25 × 50 = 1,250
  seats), and 25 fifty-seaters cost 25 × 75 = 1,875. The solver proves it too.
- **"Why aren't the routing results optimal?"** Routing with mixed bus sizes is NP-hard. We report the
  best plan found within the time limit and never call it optimal.
- **"What data do you send out?"** Coordinates to the public OSRM routing server (time model, road
  paths); outcodes, head counts and plan figures to OpenAI when the assistant is used. No names.
  Both can be switched off.

---

## Demo checklist

Before the pitch (day before, and again 30 minutes before):

- [ ] Close other heavy programs. The machine once stopped the server when memory ran low.
- [ ] Check `.streamlit\secrets.toml` holds `OPENAI_API_KEY` and the account has credit (never show the file on screen).
- [ ] Start the app: `.\.venv\Scripts\python.exe -m streamlit run app.py`, open http://localhost:8501.
- [ ] Check `data/cache/` holds the OSRM files, so the time model works offline.
- [ ] Run the fixed model once (20 s) to warm up. Ask the assistant one question to check the key.
- [ ] Zoom the browser to 110–125% so the room can read it.
- [ ] Have `demo_output/benchmark.md` open in a second tab.

Live (about 2:15 in total):

1. Sidebar: **Fixed price per bus** (20 s limit by default), **Run optimiser**. Point at "Proven optimal".
2. **Routes**: pick one bus or school; its details appear beside the map.
3. **Adjust plan**: set CF15 to 25 and SA18 to 20. The fixed model re-optimises automatically
   (about 20 s). The result, 1,950, is labelled *Feasible* at 20 s (lower bound 1,935): say so if asked.
4. **Compare**: point at the cost change and the schools that moved.
5. **Assistant** button (bottom-right): type "NP11 now brings 15 people." → show the proposal table → **Approve and run**.
   Then "NP11 brings 60 people" → show the refusal.
6. **Scenarios**: change every school's attendance by +10% → **Run scenario** → comparison table; tick
   the box → **Use this scenario as the plan** (or **Discard scenario**).
7. **Overview**: point at cost per attendee, buses removed and bus-miles saved; **Responsible AI** for
   the six principles, beneficiaries and the pilot.

Do not run a 300 s routing search live. If asked, show the benchmark table or clip B.
If the assistant is slow or offline, say so plainly and play clip D; the rest of the app does not need it.

## Backup plan

Record these before the day (Windows: Win + Alt + R with Xbox Game Bar, or OBS):

| Clip | Length | Content |
|---|---|---|
| A | 60 s | Demo steps 1–4 |
| B | 20 s | Distance model result after a 300 s run; map with colour-coded buses |
| C | 15 s | Time model with road paths drawn |
| D | 40 s | Assistant: proposal → approve → new plan; then the capacity refusal |

Screenshots: Overview (fixed), map, Compare after the edit, assistant proposal table, Scenarios
comparison, Responsible AI principles table. Keep them in a **separate backup file**, not as hidden
slides in the submitted deck: the brief allows at most 10 slides.

If the live demo fails: say "Let me show the recording", play clip A (and D), carry on with slide 6.
If the projector fails: speak from slides 6–9 using the printed results table.

## AI-use disclosure (required by the brief)

Put this on the last slide's notes or footer, and in the submission:

> Generative AI use: Claude Code (Anthropic) helped write and test the code and draft the
> documentation and this pitch script; the team reviewed, ran and tested all output.
> TEAM TO CONFIRM: any other tools (EOI, slides, images) and how they were used.
> The app's optional assistant calls OpenAI or Anthropic at run time; that is a product feature.
