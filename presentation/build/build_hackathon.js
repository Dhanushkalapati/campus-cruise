// Builds presentation/Campus_Cruise_Hackathon_Pitch.pptx (10 slides) and the speaker script.
// Figures are read from demo_output/benchmark.json, school data from attendants_dataset.csv,
// screenshots from the running app (captured with shots.js / shots2.js).
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const { applyTheme } = require("./apply_theme.js");

const PROJECT = path.resolve(__dirname, "..", "..");
const OUTDIR = path.join(PROJECT, "presentation");
const SHOTS = path.join(PROJECT, "presentation", "screenshots");
const OUT = path.join(OUTDIR, "Campus_Cruise_Hackathon_Pitch.pptx");
const SCRIPT = path.join(OUTDIR, "Campus_Cruise_Pitch_Speaker_Script.txt");

// ---------- data ----------
const bench = JSON.parse(fs.readFileSync(path.join(PROJECT, "demo_output", "benchmark.json"), "utf8"));
const run = Object.fromEntries(bench.runs.map((r) => [r.model, r]));
const F = run.fixed, D = run.distance, TT = run.time;
if (F.status !== "OPTIMAL") throw new Error("Fixed model not proven optimal in benchmark: " + F.status);
const n0 = (v) => v.toLocaleString("en-GB", { maximumFractionDigits: 0 });
const n1 = (v) => v.toLocaleString("en-GB", { minimumFractionDigits: 1, maximumFractionDigits: 1 });
const n2 = (v) => v.toLocaleString("en-GB", { minimumFractionDigits: 2, maximumFractionDigits: 2 });

const schools = fs.readFileSync(path.join(PROJECT, "attendants_dataset.csv"), "utf8").trim().split(/\r?\n/).slice(1)
  .map((l) => { const [id, lat, lon, n] = l.split(","); return { id, lat: +lat, lon: +lon, n: +n }; });
if (schools.length !== F.schools_served || schools.reduce((s, r) => s + r.n, 0) !== F.attendees) throw new Error("data mismatch");
const STADIUM = { lat: 51.478283, lon: -3.182652 };

// ---------- theme ----------
const THEME = {
  name: "Campus Cruise Pitch", headFontFace: "Calibri", bodyFontFace: "Calibri",
  colors: {
    dk1: "1B2A41", lt1: "FFFFFF", dk2: "0F1B2D", lt2: "F3F5F8",
    accent1: "D6453D", accent2: "2E7D6B", accent3: "E9A23B", accent4: "3C6E9F",
    accent5: "7A5BA6", accent6: "667085", hlink: "D6453D", folHlink: "7A5BA6",
  },
};
const H = THEME.colors;
const RED_TINT = "FBEAE8", GREEN_TINT = "E6F2EE", BLUE_TINT = "E8EFF7";

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "Campus Cruise - AI-Powered School Transport Optimisation";
pres.author = "Dhanush Kalapati, Samyaka, Sathwika";
const C = pres.SchemeColor;

pres.defineSlideMaster({ title: "CC_TITLE", background: { color: H.lt1 }, objects: [] });
pres.defineSlideMaster({
  title: "CC_CONTENT",
  background: { color: H.lt1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.28, w: 9, h: 0.62, fontSize: 28, bold: true, color: C.text1, valign: "middle", align: "left", margin: 0 }, text: "" } },
    { text: { text: "Campus Cruise · Dhanush Kalapati, Samyaka, Sathwika · AI Pollination Student Competition 2026", options: { x: 0.5, y: 5.28, w: 7.5, h: 0.22, fontSize: 9, color: C.accent6, margin: 0 } } },
  ],
  slideNumber: { x: 9.0, y: 5.28, w: 0.5, h: 0.22, fontSize: 9, color: C.accent6, align: "right" },
});

// ---------- helpers ----------
const T = (s, text, o) => s.addText(text, { isTextBox: true, margin: 0, fontSize: 14, color: C.text1, valign: "top", ...o });
const card = (s, x, y, w, h, fill, name) => s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: fill }, line: { color: fill }, objectName: name });
const dot = (s, x, y, label, fill, d = 0.4, fs = 13) => {
  s.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: fill }, line: { color: fill } });
  T(s, label, { x, y, w: d, h: d, fontSize: fs, bold: true, color: C.background1, align: "center", valign: "middle" });
};
const shot = (s, file, x, y, w, name, caption) => {
  const p = path.join(SHOTS, file);
  const [iw, ih] = pngSize(p);
  const h = w * ih / iw;
  s.addShape(pres.shapes.RECTANGLE, { x: x - 0.02, y: y - 0.02, w: w + 0.04, h: h + 0.04, fill: { color: "D0D5DD" }, line: { color: "D0D5DD" } });
  s.addImage({ path: p, x, y, w, h, altText: name });
  if (caption) T(s, caption, { x, y: y + h + 0.06, w, h: 0.22, fontSize: 10, color: C.accent6, italic: true });
  return h;
};
function pngSize(p) { const b = fs.readFileSync(p); return [b.readUInt32BE(16), b.readUInt32BE(20)]; }

function drawMap(s, box, color) {
  const k = Math.cos((51.6 * Math.PI) / 180);
  const all = schools.concat([STADIUM]);
  const xs = all.map((p) => p.lon * k), ys = all.map((p) => p.lat);
  const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
  const sc = Math.min((box.w - 0.3) / (maxX - minX), (box.h - 0.3) / (maxY - minY));
  const ox = box.x + (box.w - (maxX - minX) * sc) / 2, oy = box.y + (box.h - (maxY - minY) * sc) / 2;
  const pr = (p) => ({ x: ox + (p.lon * k - minX) * sc, y: oy + (maxY - p.lat) * sc });
  schools.forEach((p) => {
    const q = pr(p), d = 0.07 + Math.sqrt(p.n) * 0.02;
    s.addShape(pres.shapes.OVAL, { x: q.x - d / 2, y: q.y - d / 2, w: d, h: d, fill: { color, transparency: 20 }, line: { color: "F4B4AE", width: 0.5 }, objectName: `Pickup ${p.id}` });
  });
  const q = pr(STADIUM), d = 0.28;
  s.addShape(pres.shapes.STAR_5_POINT || "star5", { x: q.x - d / 2, y: q.y - d / 2, w: d, h: d, fill: { color: H.accent3 }, line: { color: H.lt1, width: 0.75 }, objectName: "Principality Stadium" });
}

const script = []; // companion speaker script
function notes(s, n, title, time, say, key, next) {
  s.addNotes(`TIME: ${time}\n\nSAY:\n${say}\n\nKEY MESSAGE: ${key}\n\nTRANSITION: ${next}`);
  script.push(`SLIDE ${n}: ${title}  [${time}]\n\nSay:\n${say}\n\nKey message for judges: ${key}\n\nTransition: ${next}\n`);
}

// ---------- 1. Title ----------
pres.addSection({ title: "Opening" });
{
  const s = pres.addSlide({ masterName: "CC_TITLE", sectionTitle: "Opening" });
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.12, fill: { color: H.accent1 }, line: { color: H.accent1 }, objectName: "Top bar" });
  const ctr = { x: 0.5, w: 9, align: "center" };
  T(s, "AI POLLINATION STUDENT COMPETITION 2026", { ...ctr, y: 0.7, h: 0.3, fontSize: 13, bold: true, color: C.accent1, charSpacing: 1 });
  T(s, "CAMPUS CRUISE", { ...ctr, y: 1.2, h: 0.9, fontSize: 48, bold: true });
  T(s, "AI-Powered School Transport Optimisation", { ...ctr, y: 2.1, h: 0.5, fontSize: 22, color: C.accent4 });
  T(s, "Technical Challenge 2 — Travel Organizer", { ...ctr, y: 2.65, h: 0.35, fontSize: 15, color: C.accent6 });
  s.addShape(pres.shapes.LINE, { x: 4.25, y: 3.3, w: 1.5, h: 0, line: { color: H.accent1, width: 2 } });
  T(s, [
    { text: "Dhanush Kalapati", options: { breakLine: true } },
    { text: "Samyaka", options: { breakLine: true } },
    { text: "Sathwika" },
  ], { ...ctr, y: 3.5, h: 1.05, fontSize: 18, bold: true, paraSpaceAfter: 2 });
  T(s, "Cardiff University", { ...ctr, y: 4.65, h: 0.35, fontSize: 14, color: C.accent6 });
  notes(s, 1, "Title", "0:00-0:20 (20 s)",
    "We're Dhanush, Samyaka and Sathwika from Cardiff University. This is Campus Cruise: it decides which schools should share a bus to an event, so getting pupils there costs less.",
    "Campus Cruise plans shared school buses to an event.",
    "So why is this a problem in the first place?");
}

// ---------- 2. Problem ----------
pres.addSection({ title: "Problem and solution" });
{
  const s = pres.addSlide({ masterName: "CC_CONTENT", sectionTitle: "Problem and solution" });
  s.addText("The problem: every school books its own bus", { placeholder: "title" });
  const stats = [
    [String(F.schools_served), "schools going to one event"],
    [n0(F.attendees), "attendees, 3 to 42 per school"],
    [String(F.baseline_buses), "buses if every school books its own"],
    [`${n1(F.baseline_utilisation_percent)}%`, "of those seats filled"],
  ];
  stats.forEach(([big, small], i) => {
    const x = 0.5 + i * 2.3;
    card(s, x, 1.15, 2.1, 1.35, i === 3 ? RED_TINT : H.lt2, `Stat ${i + 1}`);
    T(s, big, { x: x + 0.18, y: 1.25, w: 1.8, h: 0.65, fontSize: 34, bold: true, color: i === 3 ? C.accent1 : C.text1, valign: "middle" });
    T(s, small, { x: x + 0.18, y: 1.92, w: 1.8, h: 0.5, fontSize: 12, color: C.accent6 });
  });
  T(s, "Why planning by hand is hard", { x: 0.5, y: 2.75, w: 4.3, h: 0.35, fontSize: 17, bold: true });
  T(s, [
    { text: "Schools differ in size and location", options: { bullet: true, breakLine: true } },
    { text: "Buses come in 15, 35 and 50 seats, each priced differently", options: { bullet: true, breakLine: true } },
    { text: "Too many ways to group 60 schools to check by hand", options: { bullet: true } },
  ], { x: 0.5, y: 3.15, w: 4.4, h: 1.6, fontSize: 14, paraSpaceAfter: 6 });
  card(s, 5.2, 2.75, 4.3, 2.2, BLUE_TINT, "Rules");
  T(s, "Rules every plan must keep", { x: 5.4, y: 2.88, w: 3.9, h: 0.35, fontSize: 17, bold: true, color: C.accent4 });
  T(s, [
    { text: "A school's group is never split: one adult supervises it", options: { bullet: true, breakLine: true } },
    { text: "No bus carries more people than it has seats", options: { bullet: true, breakLine: true } },
    { text: "Every attendee gets a seat", options: { bullet: true } },
  ], { x: 5.4, y: 3.3, w: 3.95, h: 1.55, fontSize: 14, paraSpaceAfter: 6 });
  notes(s, 2, "Problem statement", "0:20-0:55 (35 s)",
    `Sixty schools are sending ${n0(F.attendees)} people to the same stadium. If every school books its own bus, that's ${F.baseline_buses} buses, and only ${n1(F.baseline_utilisation_percent)}% of the seats are filled. Sharing is obviously cheaper, but schools differ in size and location, buses come in three sizes, and there are far too many groupings to check by hand. And each school's group must stay together on one bus, because one adult supervises it.`,
    "Booking separately wastes seats and money; sharing is too complex to plan by hand.",
    "So here's how Campus Cruise plans it automatically.");
}

// ---------- 3. Workflow ----------
{
  const s = pres.addSlide({ masterName: "CC_CONTENT", sectionTitle: "Problem and solution" });
  s.addText("Campus Cruise: from data to a checked plan", { placeholder: "title" });
  const steps = [
    ["Load", "Schools, attendance, locations, bus prices; checked on load", H.accent6],
    ["Choose", "Cost model: fixed price, per mile or per driving hour", H.accent4],
    ["Optimise", "OR-Tools groups schools onto buses and orders pickups", H.accent1],
    ["Validate", "Separate code re-checks every rule and recomputes costs", H.accent2],
    ["Show", "Dashboard, route map, compare and scenario views", H.accent5],
    ["Export", "Plan CSV, report, audit report, metrics CSV", H.accent3],
  ];
  const w = 1.4, gap = 0.12;
  steps.forEach(([head, text, tone], i) => {
    const x = 0.5 + i * (w + gap);
    card(s, x, 1.2, w, 2.55, H.lt2, `Step ${head}`);
    dot(s, x + w / 2 - 0.22, 1.38, String(i + 1), tone, 0.44, 14);
    T(s, head, { x: x + 0.08, y: 1.95, w: w - 0.16, h: 0.35, fontSize: 16, bold: true, align: "center" });
    T(s, text, { x: x + 0.1, y: 2.35, w: w - 0.2, h: 1.3, fontSize: 12, color: C.accent6, align: "center" });
    if (i < steps.length - 1) T(s, "›", { x: x + w - 0.01, y: 2.2, w: gap + 0.02, h: 0.4, fontSize: 18, bold: true, color: C.accent6, align: "center", valign: "middle" });
  });
  card(s, 0.5, 4.0, 9, 0.95, RED_TINT, "Key point");
  T(s, [
    { text: "A plan that fails validation is never shown or exported. ", options: { bold: true, color: C.accent1 } },
    { text: "When attendance changes, the fixed-price model re-plans automatically, starting from the previous plan." },
  ], { x: 0.7, y: 4.0, w: 8.6, h: 0.95, fontSize: 14, valign: "middle" });
  notes(s, 3, "Proposed solution", "0:55-1:25 (30 s)",
    "Campus Cruise works in six steps. Load and check the data. Pick how buses are priced. OR-Tools decides which schools share each bus and the pickup order. Then separate code re-checks every rule and recalculates every cost, and a plan that fails is never shown. Finally, the results go to a dashboard and map, and can be downloaded.",
    "Every plan is optimised, then independently checked, before anyone sees it.",
    "Let me show what's inside the optimise step.");
}

// ---------- 4. Algorithms ----------
pres.addSection({ title: "Approach and results" });
{
  const s = pres.addSlide({ masterName: "CC_CONTENT", sectionTitle: "Approach and results" });
  s.addText("Algorithms: three cost models, one set of hard rules", { placeholder: "title" });
  const hdr = (t, al) => ({ text: t, options: { bold: true, color: H.lt1, fill: { color: H.dk1 }, fontSize: 12, align: al || "left", valign: "middle" } });
  const cell = (t, o = {}) => ({ text: t, options: { fontSize: 12, color: H.dk1, valign: "middle", ...o } });
  s.addTable([
    [hdr("Cost model"), hdr("Method (OR-Tools)"), hdr("Result")],
    [cell("Fixed price per bus", { bold: true }), cell("CP-SAT constraint solver: exact bin packing"), cell("Proven optimal", { bold: true, color: H.accent2 })],
    [cell("Price × straight-line miles", { bold: true }), cell("Vehicle routing: cheapest-arc start + guided local search"), cell("Best found in time limit", { color: H.accent4, bold: true })],
    [cell("Price × road driving hours", { bold: true }), cell("Same routing search on OSRM road times"), cell("Best found in time limit", { color: H.accent4, bold: true })],
  ], { x: 0.5, y: 1.1, w: 5.6, colW: [1.75, 2.45, 1.4], rowH: 0.48, border: { type: "solid", pt: 0.5, color: "E3E6EA" }, margin: 0.06 });
  T(s, "Hard rules in every model", { x: 0.5, y: 3.25, w: 5.6, h: 0.3, fontSize: 15, bold: true });
  T(s, [
    { text: "No school split between buses", options: { bullet: true, breakLine: true } },
    { text: "No bus over capacity; only the supplied 15/35/50 sizes and prices", options: { bullet: true, breakLine: true } },
    { text: "Every attendee allocated; independent validator re-checks all of it", options: { bullet: true } },
  ], { x: 0.5, y: 3.6, w: 5.6, h: 1.4, fontSize: 13, paraSpaceAfter: 4 });
  // What kind of AI is where
  card(s, 6.4, 1.1, 3.1, 3.9, H.lt2, "AI types");
  T(s, "What kind of AI, where", { x: 6.6, y: 1.2, w: 2.8, h: 0.3, fontSize: 14, bold: true });
  const kinds = [
    ["Constraint optimisation", "CP-SAT: decides the plan", H.accent2],
    ["Heuristic search", "Routing: best plan found, never called optimal", H.accent4],
    ["Generative AI", "GPT-4.1 mini assistant: proposes only, you approve", H.accent1],
    ["Machine learning (experimental)", "Ridge model approximating OSRM times when offline", H.accent5],
  ];
  kinds.forEach(([k, d], i) => {
    const y = 1.6 + i * 0.83;
    s.addShape(pres.shapes.OVAL, { x: 6.6, y: y + 0.05, w: 0.18, h: 0.18, fill: { color: kinds[i][2] }, line: { color: kinds[i][2] } });
    T(s, k, { x: 6.88, y, w: 2.5, h: 0.28, fontSize: 12, bold: true });
    T(s, d, { x: 6.88, y: y + 0.28, w: 2.5, h: 0.5, fontSize: 11, color: C.accent6 });
  });
  notes(s, 4, "Algorithms and AI approach", "1:25-2:10 (45 s)",
    "The challenge prices buses three ways. With a fixed price per bus, location doesn't matter, so it's a packing problem: OR-Tools' CP-SAT solver finds the cheapest plan and proves nothing cheaper exists. When price depends on miles or driving time, pickup order matters, so we use vehicle-routing search. It finds very good plans, but we never call them optimal. The rules are hard in every model: no school split, no bus over capacity. To be precise: this is optimisation, not a trained model. Generative AI is a separate assistant, and the machine-learning model is experimental.",
    "Exact optimisation where it can be proven; honest 'best found' labels where it cannot.",
    "So what does that achieve on the real data?");
}

// ---------- 5. Results ----------
{
  const s = pres.addSlide({ masterName: "CC_CONTENT", sectionTitle: "Approach and results" });
  s.addText(`Results: ${n1(F.saving_percent)}% lower cost, proven optimal`, { placeholder: "title" });
  card(s, 0.5, 1.1, 2.9, 1.95, H.dk1, "Headline");
  T(s, n0(F.saving), { x: 0.7, y: 1.2, w: 2.5, h: 0.75, fontSize: 40, bold: true, color: C.background1 });
  T(s, "cost units saved (fixed price per bus)", { x: 0.7, y: 1.95, w: 2.5, h: 0.3, fontSize: 12, color: "C9D1DC" });
  T(s, `${n0(F.baseline_cost)} → ${n0(F.optimised_cost)}`, { x: 0.7, y: 2.3, w: 2.5, h: 0.35, fontSize: 18, bold: true, color: C.accent3 });
  T(s, "Proven cheapest possible", { x: 0.7, y: 2.68, w: 2.5, h: 0.28, fontSize: 12, color: "C9D1DC" });
  card(s, 0.5, 3.2, 2.9, 1.8, H.lt2, "Buses");
  T(s, `${F.baseline_buses} → ${F.optimised_buses}`, { x: 0.7, y: 3.3, w: 2.5, h: 0.6, fontSize: 30, bold: true });
  T(s, `buses · seats filled ${n1(F.baseline_utilisation_percent)}% → ${n1(F.optimised_utilisation_percent)}%`, { x: 0.7, y: 3.9, w: 2.5, h: 0.5, fontSize: 12, color: C.accent6 });
  T(s, `Cost per attendee ${n2(F.cost_per_student_before)} → ${n2(F.cost_per_student_after)}`, { x: 0.7, y: 4.45, w: 2.5, h: 0.3, fontSize: 12, color: C.accent6 });

  s.addChart(pres.charts.BAR, [
    { name: "Every school books its own bus", labels: ["Fixed price", "Per mile", "Per driving hour"], values: [F.baseline_cost, D.baseline_cost, TT.baseline_cost] },
    { name: "Campus Cruise plan", labels: ["Fixed price", "Per mile", "Per driving hour"], values: [F.optimised_cost, D.optimised_cost, TT.optimised_cost] },
  ], {
    x: 3.65, y: 1.05, w: 5.85, h: 3.0, barDir: "col", barGrouping: "clustered", barGapWidthPct: 60,
    chartColors: ["B8C1CC", H.accent1],
    showTitle: true, title: "Total cost (cost units) — each model has its own prices, so compare within a model only", titleFontSize: 11, titleColor: H.dk1, titleFontFace: "+mn-lt",
    showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 10, dataLabelColor: H.dk1, dataLabelFormatCode: "#,##0", dataLabelFontFace: "+mn-lt",
    catAxisLabelColor: H.accent6, catAxisLabelFontSize: 11, catAxisLabelFontFace: "+mn-lt",
    valAxisHidden: true, valGridLine: { style: "none" }, catGridLine: { style: "none" },
    showLegend: true, legendPos: "b", legendFontSize: 10, legendColor: H.dk1, legendFontFace: "+mn-lt",
  });
  T(s, [
    { text: `Saving: fixed ${n1(F.saving_percent)}% · per mile ${n1(D.saving_percent)}% · per hour ${n1(TT.saving_percent)}%. `, options: { bold: true } },
    { text: `Routing results are the best found in ${n0(D.time_limit_s)} s, not proven minimums. Results on the challenge data only: not yet measured at a real event.` },
  ], { x: 3.65, y: 4.15, w: 5.85, h: 0.85, fontSize: 12, color: C.text1 });
  notes(s, 5, "Results and performance", "2:10-2:45 (35 s)",
    `With a fixed price per bus, the plan costs ${n0(F.optimised_cost)} instead of ${n0(F.baseline_cost)}: ${n1(F.saving_percent)}% less, with ${F.optimised_buses} buses instead of ${F.baseline_buses}. That is provably the cheapest. The other two models save ${n1(D.saving_percent)}% and ${n1(TT.saving_percent)}%; those are the best plans found in five minutes. Each model has its own prices, so I only compare within a model, and these are cost units, not pounds.`,
    `${n1(F.saving_percent)}% cheaper and ${F.baseline_buses - F.optimised_buses} fewer buses, proven optimal for fixed pricing.`,
    "Let me show you the app producing this.");
}

// ---------- 6. Demo ----------
pres.addSection({ title: "Demo" });
{
  const s = pres.addSlide({ masterName: "CC_CONTENT", sectionTitle: "Demo" });
  s.addText("Live demo: one bus, its schools and its road route", { placeholder: "title" });
  shot(s, "crop-routes.png", 0.5, 1.05, 6.2, "Campus Cruise Routes tab: Bus 04 road path from four schools to Principality Stadium, with passengers, seats left, distance and driving time",
    "Real screenshot: Campus Cruise Routes tab, fixed-price plan, 26 Oct 2026");
  const pts = [
    ["Bus 04, 50 seats", "Collects 4 schools in order"],
    ["50 / 50 passengers", "0 seats left: a full bus"],
    ["Road path from OSRM", "Drawn for display; costs use the model's own measure"],
    ["96 min estimated", "Distance card shows straight-line miles"],
  ];
  pts.forEach(([a, b], i) => {
    const y = 1.05 + i * 0.72;
    dot(s, 6.95, y + 0.03, String(i + 1), H.accent1, 0.32, 11);
    T(s, a, { x: 7.37, y, w: 2.13, h: 0.28, fontSize: 13, bold: true });
    T(s, b, { x: 7.37, y: y + 0.28, w: 2.13, h: 0.42, fontSize: 11, color: C.accent6 });
  });
  card(s, 6.95, 4.0, 2.55, 1.0, RED_TINT, "Estimate note");
  T(s, "Estimated times exclude live traffic, school boarding time and delays. Real journeys may take longer.", { x: 7.07, y: 4.06, w: 2.33, h: 0.9, fontSize: 11, color: C.text1, valign: "middle" });
  notes(s, 6, "Live demonstration", "2:45-4:05 (80 s, live)",
    "[Switch to the app at http://localhost:8501]\n1. (15 s) Sidebar: Fixed price per bus, Run optimiser. Point at 'Proven optimal', 25 buses, 1,875 vs 3,330.\n2. (25 s) Routes tab: Find a school, CF15. Turn on Road paths. 'This is Bus 04: it picks up four schools and is completely full, 50 of 50. The green line is the real road route from OSRM.' Point at the estimated driving time and read the note under it: it excludes live traffic and boarding time, so real journeys may take longer.\n3. (25 s) Adjust plan: set CF15 to 25 and SA18 to 20. 'The fixed model re-plans by itself in about 20 seconds.' Open Compare: the cost change and which schools moved.\n4. (15 s) Scenarios tab: +10% attendance, Run scenario. 'Scenarios are tested separately; the plan only changes if I tick the box and adopt it.'\nIf the demo fails: say 'Let me show the screenshot', stay on this slide, and talk through points 1-4.",
    "It's a working app: real map, real routes, honest estimates.",
    "Organisers can also just type what they want. That's the assistant.");
}

// ---------- 7. Assistant ----------
{
  const s = pres.addSlide({ masterName: "CC_CONTENT", sectionTitle: "Demo" });
  s.addText("AI planning assistant: it proposes, you approve", { placeholder: "title" });
  const steps = [
    ["You type a request", "\"Minimise bus costs while keeping nearby schools together.\""],
    ["GPT-4.1 mini proposes", "Structured JSON only: no tools, cannot change anything"],
    ["Code checks the proposal", "Every school, number and rule against the data"],
    ["You approve or reject", "Proposed changes: nothing has changed yet"],
    ["Optimiser + validator run", "The same OR-Tools models and checks as every plan"],
  ];
  steps.forEach(([a, b], i) => {
    const y = 1.05 + i * 0.72;
    dot(s, 0.5, y + 0.03, String(i + 1), i === 3 ? H.accent1 : H.dk1, 0.36, 12);
    T(s, a, { x: 1.0, y, w: 4.6, h: 0.3, fontSize: 14, bold: true });
    T(s, b, { x: 1.0, y: y + 0.3, w: 4.6, h: 0.38, fontSize: 12, color: C.accent6 });
  });
  card(s, 0.5, 4.7, 5.2, 0.42, BLUE_TINT, "Honest note");
  T(s, "\"Nearby\" is a cost tie-break in the fixed model, not a hard rule.", { x: 0.65, y: 4.7, w: 5.0, h: 0.42, fontSize: 12, color: C.accent4, bold: true, valign: "middle" });
  const h = 3.95, w = h * 660 / 1050;
  s.addShape(pres.shapes.RECTANGLE, { x: 9.5 - w - 0.02, y: 1.03, w: w + 0.04, h: h + 0.04, fill: { color: "D0D5DD" }, line: { color: "D0D5DD" } });
  s.addImage({ path: path.join(SHOTS, "crop-assistant.png"), x: 9.5 - w, y: 1.05, w, h, altText: "Campus Cruise assistant panel answering 'Minimise bus costs while keeping nearby schools together' with a proposal awaiting approval" });
  T(s, "Real screenshot, 26 Oct 2026", { x: 9.5 - w, y: 5.04, w, h: 0.18, fontSize: 9, italic: true, color: C.accent6, align: "center" });
  notes(s, 7, "AI Transport Planning Assistant", "4:05-4:40 (35 s)",
    "Organisers can also just type: 'Minimise bus costs while keeping nearby schools together.' GPT-4.1 mini turns that into a proposal, but it can't change anything. Code checks it against the data and shows it: nothing has changed yet. Only when I approve does the optimiser run. And it's honest: 'nearby' isn't a hard rule, it's how the fixed model breaks ties, and it says so.",
    "Generative AI makes the tool easy to use, but people and the optimiser stay in control.",
    "So who actually benefits from this?");
}

// ---------- 8. Impact ----------
pres.addSection({ title: "Impact and responsibility" });
{
  const s = pres.addSlide({ masterName: "CC_CONTENT", sectionTitle: "Impact and responsibility" });
  s.addText("Sector impact: who benefits and what we can measure", { placeholder: "title" });
  const who = [
    ["Event organisers & SMEs", "a checked plan in seconds"],
    ["Schools & teachers", "lower, fair cost shares"],
    ["Pupils", "lower cost to attend"],
    ["Welsh coach operators", "a clear booking list"],
    ["Event venues", "fewer coaches to park"],
    ["Audiences", "less coach traffic on event day"],
  ];
  who.forEach(([a, b], i) => {
    const x = 0.5 + (i % 2) * 2.3, y = 1.1 + Math.floor(i / 2) * 0.82;
    card(s, x, y, 2.15, 0.7, H.lt2, a);
    T(s, a, { x: x + 0.12, y: y + 0.08, w: 1.95, h: 0.28, fontSize: 12, bold: true });
    T(s, b, { x: x + 0.12, y: y + 0.36, w: 1.95, h: 0.28, fontSize: 11, color: C.accent6 });
  });
  const mets = [
    [`−${F.baseline_buses - F.optimised_buses}`, "buses to book and coordinate"],
    [`${n1(F.optimised_utilisation_percent)}%`, `seats filled (was ${n1(F.baseline_utilisation_percent)}%)`],
    [`−${n1(F.bus_miles_saved_percent)}%`, `bus-miles: ${n1(F.baseline_bus_miles)} → ${n1(F.plan_bus_miles)}`],
  ];
  mets.forEach(([big, small], i) => {
    const y = 1.1 + i * 0.82;
    card(s, 5.25, y, 4.25, 0.7, i === 2 ? GREEN_TINT : H.lt2, `Metric ${i + 1}`);
    T(s, big, { x: 5.4, y, w: 1.35, h: 0.7, fontSize: 24, bold: true, color: i === 2 ? C.accent2 : C.accent1, valign: "middle" });
    T(s, small, { x: 6.8, y, w: 2.6, h: 0.7, fontSize: 12, valign: "middle" });
  });
  T(s, "Fixed-price plan on challenge data. Bus-miles are straight-line, first pickup to stadium, summed over buses; baseline = one bus per school. No fuel or emissions claim.", { x: 5.25, y: 3.55, w: 4.25, h: 0.55, fontSize: 10, color: C.accent6 });
  card(s, 0.5, 4.2, 9, 0.85, BLUE_TINT, "Pilot");
  T(s, [
    { text: "Next: a pilot. ", options: { bold: true, color: C.accent4 } },
    { text: "One Welsh event organiser or coach operator, same inputs, current plan vs Campus Cruise. Measure cost, buses, seat use, mileage, planning time and user feedback, with school-level data only." },
  ], { x: 0.7, y: 4.2, w: 8.6, h: 0.85, fontSize: 13, valign: "middle" });
  notes(s, 8, "Sector impact and beneficiaries", "4:40-5:15 (35 s)",
    `Who gains? Organisers get a checked plan in seconds. Schools and pupils pay less to attend. Coach operators get a clear booking list, and venues have fewer coaches to park. On the challenge data that's ${F.baseline_buses - F.optimised_buses} fewer buses, ${n1(F.optimised_utilisation_percent)}% of seats filled, and about ${n0(F.bus_miles_saved_percent)}% fewer bus-miles, straight-line, with no emissions claim. It hasn't run at a real event yet, so the next step is a pilot with one Welsh organiser, measuring cost, buses, mileage and planning time.`,
    "Measurable savings on the data, and a clear plan to prove them in the real world.",
    "Just as important is how it's built responsibly.");
}

// ---------- 9. Responsible AI & Wales ----------
{
  const s = pres.addSlide({ masterName: "CC_CONTENT", sectionTitle: "Impact and responsibility" });
  s.addText("Responsible AI, aligned with Welsh strategy", { placeholder: "title" });
  const pr = [
    ["Transparency", "Every plan labelled proven optimal or best found; estimates marked"],
    ["Privacy", "School-level head counts only, no pupil names; assistant never sees coordinates"],
    ["Security", "API keys never shown or logged; LLM has no tools, returns checked JSON"],
    ["Accountability", "Organiser approves every AI change; audit report of every run"],
    ["Safety & fairness", "Hard rules re-checked by independent code; cost shared per attendee"],
    ["Honest limits", "No traffic or boarding time; not yet tested at a real event"],
  ];
  pr.forEach(([a, b], i) => {
    const x = 0.5 + (i % 2) * 2.75, y = 1.05 + Math.floor(i / 2) * 1.0;
    card(s, x, y, 2.6, 0.88, H.lt2, a);
    T(s, a, { x: x + 0.12, y: y + 0.07, w: 2.4, h: 0.26, fontSize: 13, bold: true, color: C.accent2 });
    T(s, b, { x: x + 0.12, y: y + 0.34, w: 2.4, h: 0.52, fontSize: 10.5, color: C.text1 });
  });
  T(s, "Human oversight: decision support, not an autonomous system.", { x: 0.5, y: 4.15, w: 5.35, h: 0.3, fontSize: 13, bold: true, color: C.accent1 });
  card(s, 6.1, 1.05, 3.4, 3.95, BLUE_TINT, "Wales");
  T(s, "National Events Strategy for Wales 2022–2030", { x: 6.28, y: 1.15, w: 3.05, h: 0.5, fontSize: 13, bold: true, color: C.accent4 });
  T(s, [
    { text: "\"A balanced portfolio will consider age, region, ethnicity, religion, cost to attend…\"", options: { italic: true, breakLine: true } },
    { text: "→ lower cost to attend for school groups", options: { bold: true, breakLine: true } },
    { text: " ", options: { breakLine: true, fontSize: 5 } },
    { text: "Success measure: \"Percentage of events encouraging or offering active and sustainable transport options\"", options: { italic: true, breakLine: true } },
    { text: "→ shared coaches, fewer bus-miles", options: { bold: true } },
  ], { x: 6.28, y: 1.7, w: 3.05, h: 2.4, fontSize: 11 });
  T(s, "The strategy builds on the Well-being of Future Generations (Wales) Act 2015. No endorsement implied.", { x: 6.28, y: 3.95, w: 3.05, h: 0.7, fontSize: 10, color: C.accent6 });
  T(s, "Source: gov.wales/national-events-strategy-wales-2022-2030-html (checked 26 Oct 2026)", { x: 0.5, y: 4.55, w: 5.35, h: 0.4, fontSize: 9, color: C.accent6 });
  notes(s, 9, "Responsible AI and Welsh strategic alignment", "5:15-5:50 (35 s)",
    "Campus Cruise is decision support, not an autonomous system. Every plan says whether it's proven or just the best found. It uses school-level head counts only, no names. The organiser approves every AI change, and independent code re-checks every rule. It fits the National Events Strategy for Wales, which asks events to consider cost to attend and measures sustainable transport options. Shared coaches help with both. We don't claim any endorsement.",
    "Responsible by design, and pointed at goals Wales has already set.",
    "To wrap up.");
}

// ---------- 10. Conclusion ----------
pres.addSection({ title: "Close" });
{
  const s = pres.addSlide({ masterName: "CC_CONTENT", sectionTitle: "Close" });
  s.addText("Fewer buses, lower cost, plans people can trust", { placeholder: "title" });
  const pts = [
    ["Problem", "Schools booking alone waste seats and money", H.accent6],
    ["Result", `${n1(F.saving_percent)}% cheaper, ${F.baseline_buses} → ${F.optimised_buses} buses, proven optimal`, H.accent1],
    ["Contribution", "Exact + routing optimisation, independent validation, a guarded AI assistant", H.accent4],
    ["Next step", "Pilot with a Welsh organiser and collect stakeholder feedback", H.accent2],
  ];
  pts.forEach(([a, b, tone], i) => {
    const x = 0.5 + i * 2.3;
    card(s, x, 1.1, 2.1, 1.95, H.lt2, a);
    s.addShape(pres.shapes.OVAL, { x: x + 0.18, y: 1.25, w: 0.28, h: 0.28, fill: { color: tone }, line: { color: tone } });
    T(s, a, { x: x + 0.55, y: 1.22, w: 1.45, h: 0.34, fontSize: 15, bold: true, valign: "middle" });
    T(s, b, { x: x + 0.18, y: 1.7, w: 1.8, h: 1.3, fontSize: 13 });
  });
  T(s, "Thank you. Questions?", { x: 0.5, y: 3.25, w: 9, h: 0.5, fontSize: 24, bold: true, color: C.accent1 });
  card(s, 0.5, 3.95, 9, 1.1, H.lt2, "AI disclosure");
  T(s, [
    { text: "Generative AI disclosure. ", options: { bold: true } },
    { text: "Claude (Anthropic, via Claude Code) helped write and test the code and draft the documentation, speaker script and this presentation. ChatGPT (OpenAI) was also used as a writing and planning aid. Dhanush reviewed, ran and tested all output. Separately, the app's optional assistant calls OpenAI GPT-4.1 mini at run time as a product feature." },
  ], { x: 0.7, y: 3.95, w: 8.6, h: 1.1, fontSize: 11, valign: "middle" });
  notes(s, 10, "Conclusion, next steps and AI disclosure", "5:50-6:10 (20 s)",
    `Campus Cruise finds which schools should share a bus: ${n1(F.saving_percent)}% cheaper on the challenge data, ${F.optimised_buses} buses instead of ${F.baseline_buses}, proven optimal and independently checked. Next, I'd like to pilot it with a Welsh event organiser. My AI-use disclosure is on the slide. Thank you, I'm happy to take questions.`,
    "Campus Cruise: fewer buses, lower cost, and plans organisers can trust.",
    "(Questions.)");
}

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  const header = [
    "CAMPUS CRUISE — FINAL PITCH SPEAKER SCRIPT",
    "AI Pollination Student Competition 2026 · Technical Challenge 2: Travel Organizer",
    "Team: Dhanush Kalapati, Samyaka, Sathwika, Cardiff University",
    `Target length: about 6 minutes spoken (limit 5-7 min). Live demo: 80 s on slide 6.`,
    `Figures: python benchmark.py --offline, run started ${bench.started}. Speak at about 130 words a minute.`,
    "", "=".repeat(78), "",
  ].join("\n");
  fs.writeFileSync(SCRIPT, header + script.join("\n" + "-".repeat(78) + "\n\n"), "utf8");
  const words = script.join(" ").split(/\s+/).length;
  console.log("wrote", OUT, "\nwrote", SCRIPT, "\nscript words (incl. labels):", words);
})();
