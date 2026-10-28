# Campus Cruise: final presentation checklist

Pitch: Wednesday 28 October 2026. Limit: 5–7 minutes, at most 10 slides (competition brief, section 4).
Deck: `presentation/Campus_Cruise_Hackathon_Pitch.pptx` (10 slides, speaker notes on every slide).
Script: `presentation/Campus_Cruise_Pitch_Speaker_Script.txt` (about 5 min 50 s spoken).

## 1. Competition requirements (from `hcch-student-competition-details.pdf`)

- [x] 5–7 minute presentation: script timed at about 5:50, including an 80-second live demo.
- [x] At most 10 slides: exactly 10.
- [x] Includes a demo of the solution: slide 6 (live), with real screenshots as a fallback.
- [x] Generative AI disclosure: slide 10 names Claude (via Claude Code) and ChatGPT.
- [x] Judging criteria covered:
  - Communication: slides 2–3, written for non-experts.
  - Innovation & AI approach: slides 4 and 7.
  - Sector impact & outcome: slides 5 and 8. Beneficiaries are named, metrics are measured, and a pilot is proposed.
  - Strategic alignment & responsible AI: slide 9.
- [ ] **Co-production evidence** (named in the "Sector Impact" criterion): none yet. If you get any real feedback before the pitch (a teacher, trip organiser or HCCH mentor), add one line to slide 8. Never invent feedback.

## 2. Figures used and where they come from

All results figures come from `python benchmark.py --offline`, run started 2026-10-26 23:36. The deck reads `demo_output/benchmark.json` directly, so rebuilding the deck after a new benchmark updates every number.

| Figure | Value | Source |
|---|---|---|
| Schools, attendees | 60, 1,244 (3–42 per school) | `attendants_dataset.csv` |
| Fixed model: plan vs baseline | 1,875 vs 3,330 cost units, saving 1,455 (43.7%), proven optimal | benchmark, fixed model |
| Buses | 60 → 25 | benchmark |
| Seats filled | 68.0% → 99.5% | benchmark |
| Cost per attendee | 2.68 → 1.51 | benchmark |
| Distance model | 2,787.25 vs 3,903.87 (28.6%), best found in 300 s | benchmark |
| Time model | 6,762.89 vs 8,220.26 (17.7%), best found in 300 s | benchmark |
| Bus-miles, fixed plan | 1,289.4 → 789.4 (−38.8%), straight-line | benchmark |
| Bus 04 demo | 4 schools, 50/50 passengers, 96 min estimate | real app screenshot, 26 Oct 2026 |

- Costs are in **cost units**. The data gives no currency, so never say pounds.
- Never compare costs across models: each model uses its own prices.
- Routing results can change slightly between runs, because the search is heuristic. Example: the time model saved 18.9% on 2026-10-25 and 17.7% on 2026-10-26. `docs/pitch.md` and the older `docs/Campus_Cruise_pitch.pptx` still show the 2026-10-25 figures.
- The fixed plan's bus-miles vary slightly between runs (785–792 seen), because its distance tie-break is time-limited. Its cost (1,875) never changes.

## 3. Claims checked against the code

| Claim | Evidence |
|---|---|
| CP-SAT exact model for fixed pricing | `src/solvers.py` (`solve_fixed`, `cp_model`) |
| Routing: cheapest-arc start + guided local search | `src/solvers.py` (`PATH_CHEAPEST_ARC`, `GUIDED_LOCAL_SEARCH`) |
| Independent validator | `src/validation.py` (`validate_solution`) |
| Road times and paths from OSRM, cached | `src/distances.py` |
| Assistant: JSON-only proposal, code checks, approval before any change | `src/assistant.py` (`PROPOSAL_SCHEMA`, `validate_proposal`, `CHANGE_TYPES`) |
| "Nearby schools" is a tie-break, not a hard rule | `src/assistant.py` (`GROUPING_LIMITATION`); the real assistant reply in `presentation/screenshots/crop-assistant.png` |
| Experimental ML model approximating OSRM | `src/ml_travel_time.py`, `models/travel_time_ml.json` (grouped-CV MAE 7.25 min vs 10.34 min for the fixed-speed formula) |
| Exports: plan CSV, report, audit report, metrics CSV | `app.py`, Reports tab download buttons |
| Driving-time estimate note | `app.py` (`DRIVING_TIME_NOTE`), visible in the slide 6 screenshot |
| National Events Strategy quotes | gov.wales/national-events-strategy-wales-2022-2030-html, checked 26 Oct 2026 |

## 4. Please confirm

- [ ] **ChatGPT's role.** Slide 10 says it was "a writing and planning aid". If you used it for something more specific (the EOI, prompts, or the slide text), change the wording.
- [ ] **Name and university** on slide 1: "Dhanush Kalapati, Cardiff University".

## 5. Day before and 30 minutes before

- [ ] Close heavy programs; the server once stopped when memory ran low.
- [ ] `.streamlit\secrets.toml` holds a working `OPENAI_API_KEY` with credit. Never show this file on screen.
- [ ] Start the app: `.\.venv\Scripts\python.exe -m streamlit run app.py`, then open http://localhost:8501.
- [ ] `data\cache\` holds the OSRM files, so the time model and road paths work offline.
- [ ] Run the fixed model once to warm up (20 s). Ask the assistant one question to check the key.
- [ ] Zoom the browser to 110–125%.
- [ ] Rehearse the demo (slide 6 notes) at least twice: target 80 seconds.
- [ ] Rehearse the full talk with a timer: it must finish under 7:00.

## 6. Demo plan (80 s, slide 6)

1. (15 s) Choose **Fixed price per bus**, then **Run optimiser**. Point at "Proven optimal", 25 buses, 1,875 vs 3,330.
2. (25 s) In **Routes**, set **Find a school** to CF15, then turn on **Road paths**. Show Bus 04: 50/50 passengers, the road route, and the driving-time note.
3. (25 s) In **Adjust plan**, set CF15 to 25 and SA18 to 20. The plan re-optimises automatically (about 20 s). In **Compare**, show the cost change and which schools moved. If asked: the result, 1,950, is labelled *Feasible* at 20 s.
4. (15 s) In **Scenarios**, set +10% attendance, then **Run scenario**. The plan changes only after you tick the box and adopt it.

If the demo fails, stay on slide 6 and talk through the screenshot. Do not run a 300 s routing search live.

## 7. Regenerate

```powershell
# 1. Refresh results (about 11 minutes; writes demo_output/benchmark.json and .md)
.\.venv\Scripts\python.exe benchmark.py --offline

# 2. Rebuild the deck and speaker script (needs Node.js)
cd presentation\build
npm install
node build_hackathon.js

# 3. Optional: retake screenshots (the app must be running at http://localhost:8501)
node shots.js ..\screenshots CF15
node shots2.js ..\screenshots
python crop.py ..\screenshots
node build_hackathon.js
```

`shots2.js` sends one real request to OpenAI through the app's assistant.

Open the deck with `start presentation\Campus_Cruise_Hackathon_Pitch.pptx`.
