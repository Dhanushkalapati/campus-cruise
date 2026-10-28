// Builds "presentation/ai project.pptx": the Campus Cruise pitch in the style of the slides in "ppt ref/".
// Figures come from demo_output/benchmark.json; screenshots are real captures of the app
// (shots.js, shots3.js); stadium photos are crops of the reference slides (presentation/assets).
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa");
const { applyTheme } = require("./apply_theme.js");

const PROJECT = path.resolve(__dirname, "..", "..");
const SHOTS = path.join(PROJECT, "presentation", "screenshots");
const ASSETS = path.join(PROJECT, "presentation", "assets");
const OUT = path.join(PROJECT, "presentation", "ai project.pptx");

// ---------- data ----------
const bench = JSON.parse(fs.readFileSync(path.join(PROJECT, "demo_output", "benchmark.json"), "utf8"));
const run = Object.fromEntries(bench.runs.map((r) => [r.model, r]));
const F = run.fixed, D = run.distance, TT = run.time;
if (F.status !== "OPTIMAL") throw new Error("Fixed model not proven optimal in benchmark: " + F.status);
const n0 = (v) => v.toLocaleString("en-GB", { maximumFractionDigits: 0 });
const n1 = (v) => v.toLocaleString("en-GB", { minimumFractionDigits: 1, maximumFractionDigits: 1 });
const n2 = (v) => v.toLocaleString("en-GB", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const busCut = ((F.baseline_buses - F.optimised_buses) / F.baseline_buses) * 100;

// ---------- theme (navy + red, as in the reference slides) ----------
const THEME = {
  name: "Campus Cruise", headFontFace: "Arial", bodyFontFace: "Arial",
  colors: {
    dk1: "0B1F45", lt1: "FFFFFF", dk2: "334155", lt2: "F5F7FA",
    accent1: "D7192F", accent2: "1F5FBF", accent3: "2E9E4F", accent4: "F29A1F",
    accent5: "6B3FB5", accent6: "64748B", hlink: "1F5FBF", folHlink: "6B3FB5",
  },
};
const H = THEME.colors;
const TINT = { red: "FDECEE", blue: "EAF1FB", green: "EAF6EE", orange: "FEF4E6", purple: "F2EDFB", grey: "F5F7FA" };
const HEX = { red: H.accent1, blue: H.accent2, green: H.accent3, orange: H.accent4, purple: H.accent5, navy: H.dk1, grey: H.accent6 };

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 in
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "Campus Cruise - AI-Powered School Transport Optimisation";
pres.author = "Dhanush Kalapati, Samyaka Nadavala, Sathwika Orugonda";
const C = pres.SchemeColor;
const W = 13.333;

pres.defineSlideMaster({ title: "CC_COVER", background: { color: H.lt1 }, objects: [] });
pres.defineSlideMaster({ title: "CC_DARK", background: { color: H.dk1 }, objects: [] });
pres.defineSlideMaster({
  title: "CC_CONTENT",
  background: { color: H.lt1 },
  objects: [
    { placeholder: { options: { name: "kicker", type: "body", x: 0.55, y: 0.32, w: 8, h: 0.3, fontSize: 13, bold: true, color: C.accent6, charSpacing: 1, margin: 0, valign: "top" }, text: "" } },
    { rect: { x: 0.55, y: 0.68, w: 0.55, h: 0.07, fill: { color: C.accent1 }, line: { color: C.accent1 } } },
    { placeholder: { options: { name: "title", type: "title", x: 0.55, y: 0.85, w: 8.3, h: 1.15, fontSize: 34, bold: true, color: C.text1, valign: "top", align: "left", margin: 0 }, text: "" } },
    { text: { text: "Campus Cruise · AI Pollination Student Competition 2026 · Cardiff University", options: { x: 0.55, y: 7.12, w: 8, h: 0.24, fontSize: 9, color: C.accent6, margin: 0 } } },
  ],
  slideNumber: { x: 12.28, y: 7.12, w: 0.5, h: 0.24, fontSize: 9, color: C.accent6, align: "right" },
});

// ---------- helpers ----------
const T = (s, text, o) => s.addText(text, { isTextBox: true, margin: 0, fontSize: 14, color: C.text1, valign: "top", ...o });
const card = (s, x, y, w, h, fill, name, line) => s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
  x, y, w, h, rectRadius: 0.12, fill: { color: fill }, line: { color: line || fill, width: 0.75 }, objectName: name,
});
const softCard = (s, x, y, w, h, fill, name) => s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
  x, y, w, h, rectRadius: 0.12, fill: { color: fill }, line: { color: "E6E9EF", width: 0.75 }, objectName: name,
  shadow: { type: "outer", color: "0B1F45", opacity: 0.08, blur: 6, offset: 1.5, angle: 90 },
});
const circle = (s, x, y, d, fill, name) => s.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: fill }, line: { color: fill }, objectName: name });
const num = (s, x, y, label, fill, d = 0.5, fs = 18) => {
  circle(s, x, y, d, fill, `Step ${label}`);
  T(s, String(label), { x, y, w: d, h: d, fontSize: fs, bold: true, color: C.background1, align: "center", valign: "middle" });
};
function pngSize(p) { const b = fs.readFileSync(p); return [b.readUInt32BE(16), b.readUInt32BE(20)]; }
const img = (s, file, x, y, w, h, alt, extra = {}) => s.addImage({ path: file, x, y, w, h, altText: alt, ...extra });
const frame = (s, x, y, w, h) => s.addShape(pres.shapes.RECTANGLE, { x: x - 0.02, y: y - 0.02, w: w + 0.04, h: h + 0.04, fill: { color: "D7DCE3" }, line: { color: "D7DCE3" } });

const iconCache = {};
async function icon(name, color) {
  const key = name + color;
  if (!iconCache[key]) {
    const Comp = fa[name];
    if (!Comp) throw new Error("No icon " + name);
    const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(Comp, { color: "#" + color, size: 256 }));
    iconCache[key] = "image/png;base64," + (await sharp(Buffer.from(svg)).png().toBuffer()).toString("base64");
  }
  return iconCache[key];
}
async function ico(s, name, color, x, y, d, alt) { s.addImage({ data: await icon(name, color), x, y, w: d, h: d, altText: alt || name }); }
// Icon centred in a tinted circle (the deck's motif).
async function badge(s, name, tone, x, y, d = 0.75, fill) {
  circle(s, x, y, d, fill || TINT[tone], `Icon ${name}`);
  const k = d * 0.5;
  await ico(s, name, HEX[tone], x + (d - k) / 2, y + (d - k) / 2, k, name);
}
// Header photo top-right; height fixed, width from the image.
function headerPhoto(s, file, h = 2.15) {
  const p = path.join(ASSETS, file);
  const [iw, ih] = pngSize(p);
  const w = h * iw / ih;
  img(s, p, W - w, 0, w, h, "Coaches outside Principality Stadium, Cardiff (illustration from the team's reference slides)");
}
const title = (s, navy, red, kicker) => {
  s.addText(kicker, { placeholder: "kicker" });
  const runs = [{ text: navy, options: { color: H.dk1 } }];
  if (red) runs.push({ text: red, options: { color: H.accent1 } });
  s.addText(runs, { placeholder: "title" });
};
const sub = (s, text, w = 8.3, y = 2.05) => T(s, text, { x: 0.55, y, w, h: 0.62, fontSize: 15, color: C.text2 });
const notes = (s, time, say, key) => s.addNotes(`TIME: ${time}\n\nSAY:\n${say}\n\nKEY MESSAGE: ${key}`);

(async () => {
  // ======================= 1. Cover =======================
  pres.addSection({ title: "Opening" });
  {
    const s = pres.addSlide({ masterName: "CC_COVER", sectionTitle: "Opening" });
    const hero = path.join(ASSETS, "hero.png");
    const [iw, ih] = pngSize(hero);
    const hh = 5.75, hw = hh * iw / ih;
    img(s, hero, W - hw, 0, hw, hh, "School pupils boarding coaches outside Principality Stadium, Cardiff (illustration from the team's reference slides)");
    T(s, "AI POLLINATION STUDENT COMPETITION 2026", { x: 0.6, y: 1.55, w: 6.2, h: 0.3, fontSize: 13, bold: true, color: C.accent6, charSpacing: 1 });
    s.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: 1.92, w: 0.55, h: 0.07, fill: { color: H.accent1 }, line: { color: H.accent1 } });
    T(s, [{ text: "Campus ", options: { color: H.dk1 } }, { text: "Cruise", options: { color: H.accent1 } }],
      { x: 0.6, y: 2.15, w: 6.2, h: 1.0, fontSize: 52, bold: true });
    T(s, "AI-powered school transport optimisation for events in Wales", { x: 0.6, y: 3.3, w: 6.0, h: 0.85, fontSize: 21, bold: true, color: C.text1 });
    softCard(s, 0.6, 5.95, 12.13, 1.05, H.lt1, "Team");
    await badge(s, "FaUsers", "navy", 0.82, 6.12, 0.72, TINT.blue);
    T(s, "Dhanush Kalapati · Samyaka Nadavala · Sathwika Orugonda", { x: 1.75, y: 6.12, w: 10.8, h: 0.38, fontSize: 18, bold: true });
    T(s, [
      { text: "Cardiff University", options: { color: H.accent6 } },
      { text: "   ·   Technical Challenge 2: Travel Organizer", options: { color: H.accent1, bold: true } },
    ], { x: 1.75, y: 6.52, w: 10.8, h: 0.3, fontSize: 13 });
    notes(s, "0:00-0:20 (20 s)",
      "We're Dhanush Kalapati, Samyaka Nadavala and Sathwika Orugonda from Cardiff University. This is Campus Cruise: it decides which schools should share a bus to an event, so getting pupils there costs less, and it proves every plan keeps the rules.",
      "Campus Cruise plans shared school buses to an event.");
  }

  // ======================= 2. Problem =======================
  pres.addSection({ title: "Problem and solution" });
  {
    const s = pres.addSlide({ masterName: "CC_CONTENT", sectionTitle: "Problem and solution" });
    const hero = path.join(ASSETS, "hero.png");
    const [iw, ih] = pngSize(hero);
    const hh = 4.4, hw = hh * iw / ih;
    img(s, hero, W - hw, 0, hw, hh, "Pupils boarding coaches outside Principality Stadium (illustration from the team's reference slides)");
    title(s, "The Challenge of\n", "School Transport Planning", "PROBLEM STATEMENT");
    sub(s, `${F.schools_served} schools are sending ${n0(F.attendees)} people to one event. Getting them there efficiently is a hard planning problem.`, 7.2, 2.05);
    const cards = [
      ["FaCoins", "red", "High transport costs", `Every school booking its own bus means ${F.baseline_buses} buses and ${n0(F.baseline_cost)} cost units.`],
      ["FaBus", "blue", "Empty seats", `Booked separately, only ${n1(F.baseline_utilisation_percent)}% of seats are filled; the rest are paid for and empty.`],
      ["FaUsers", "blue", "Groups stay together", "Each school's group is supervised by its own adult, so it can never be split across buses."],
      ["FaRoute", "red", "Too many choices", "Sizes, locations, 15/35/50-seat buses and prices: far too many groupings to check by hand."],
    ];
    cards.forEach(([icn, tone, head, body], i) => {
      const x = 0.55 + (i % 2) * 3.95, y = 2.8 + Math.floor(i / 2) * 1.62;
      softCard(s, x, y, 3.75, 1.48, H.lt1, head);
      // badges are async; drawn below
      cards[i].push(x, y);
    });
    for (const [icn, tone, head, body, x, y] of cards) {
      await badge(s, icn, tone, x + 0.18, y + 0.22, 0.78);
      T(s, head, { x: x + 1.12, y: y + 0.17, w: 2.5, h: 0.32, fontSize: 15, bold: true });
      s.addShape(pres.shapes.RECTANGLE, { x: x + 1.12, y: y + 0.53, w: 0.35, h: 0.04, fill: { color: H.accent1 }, line: { color: H.accent1 } });
      T(s, body, { x: x + 1.12, y: y + 0.65, w: 2.5, h: 0.78, fontSize: 11.5, color: C.text2 });
    }
    // Flow card under the photo: schools -> shared bus -> venue
    softCard(s, 8.75, 4.62, 4.03, 1.65, H.lt1, "Flow");
    const tones = ["blue", "red", "green"];
    for (let i = 0; i < 3; i++) {
      await ico(s, "FaSchool", H.dk1, 8.95, 4.8 + i * 0.47, 0.36, "School");
      await ico(s, "FaUsers", HEX[tones[i]], 9.4, 4.8 + i * 0.47, 0.36, "Pupils");
    }
    T(s, "›", { x: 9.85, y: 5.05, w: 0.4, h: 0.6, fontSize: 30, bold: true, color: C.accent6, align: "center", valign: "middle" });
    await ico(s, "FaBus", H.dk1, 10.3, 5.0, 0.7, "Shared bus");
    T(s, "›", { x: 11.05, y: 5.05, w: 0.4, h: 0.6, fontSize: 30, bold: true, color: C.accent1, align: "center", valign: "middle" });
    await ico(s, "FaMapMarkerAlt", H.accent1, 11.65, 4.86, 0.6, "Event venue");
    T(s, "Event venue", { x: 11.35, y: 5.55, w: 1.25, h: 0.3, fontSize: 11, bold: true, align: "center" });
    // Bottom banner
    card(s, 0.55, 6.2, 8.0, 0.78, TINT.red, "Problem question");
    circle(s, 0.75, 6.29, 0.6, H.accent1, "Question mark");
    await ico(s, "FaQuestion", H.lt1, 0.9, 6.44, 0.3, "Question");
    T(s, [
      { text: "The problem we are solving: ", options: { bold: true, color: H.dk1 } },
      { text: "how can schools share buses, never split a group or overfill a bus, and pay as little as possible?" },
    ], { x: 1.55, y: 6.2, w: 6.85, h: 0.78, fontSize: 13.5, valign: "middle" });
    notes(s, "0:20-0:55 (35 s)",
      `Sixty schools are sending ${n0(F.attendees)} people to the same stadium. If every school books its own bus, that's ${F.baseline_buses} buses, and only ${n1(F.baseline_utilisation_percent)}% of the seats are filled. Sharing is obviously cheaper, but schools differ in size and location, buses come in three sizes, and there are far too many groupings to check by hand. And each school's group must stay together on one bus, because one adult supervises it.`,
      "Booking separately wastes seats and money; sharing is too complex to plan by hand.");
  }

  // ======================= 3. Proposed solution =======================
  {
    const s = pres.addSlide({ masterName: "CC_CONTENT", sectionTitle: "Problem and solution" });
    headerPhoto(s, "photo_features.png", 2.0);
    title(s, "Campus ", "Cruise", "PROPOSED SOLUTION");
    sub(s, "Plans shared school buses to any event: picks which schools share each bus and the pickup order, then checks every rule before anyone sees the plan.", 8.3, 1.62);
    const cols = [
      { n: 1, tone: "navy", head: "Input data", tag: "Schools and venue", fill: TINT.blue },
      { n: 2, tone: "red", head: "Optimise", tag: "Choose the cost model", fill: TINT.red },
      { n: 3, tone: "navy", head: "Validate", tag: "Independent re-check", fill: TINT.green },
      { n: 4, tone: "red", head: "Visualise", tag: "Dashboard and map", fill: TINT.red },
      { n: 5, tone: "navy", head: "Export", tag: "Share and book", fill: TINT.blue },
    ];
    const cw = 2.3, gap = 0.155, y0 = 2.5, ch = 3.45;
    for (let i = 0; i < cols.length; i++) {
      const c = cols[i], x = 0.55 + i * (cw + gap);
      softCard(s, x, y0, cw, ch, H.lt1, `Step ${c.head}`);
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: y0, w: cw, h: 0.82, rectRadius: 0.12, fill: { color: c.fill }, line: { color: c.fill } });
      num(s, x + 0.14, y0 + 0.16, c.n, c.tone === "red" ? H.accent1 : H.dk1, 0.5, 17);
      T(s, c.head, { x: x + 0.74, y: y0 + 0.12, w: cw - 0.8, h: 0.32, fontSize: 15, bold: true, color: c.tone === "red" ? C.accent1 : C.text1 });
      T(s, c.tag, { x: x + 0.74, y: y0 + 0.44, w: cw - 0.8, h: 0.3, fontSize: 10.5, color: C.text2 });
      if (i < cols.length - 1) T(s, "›", { x: x + cw - 0.02, y: y0 + 1.5, w: gap + 0.04, h: 0.5, fontSize: 20, bold: true, color: C.text1, align: "center", valign: "middle" });
    }
    const colX = (i) => 0.55 + i * (cw + gap);
    const rows = async (i, items, tone) => {
      for (let k = 0; k < items.length; k++) {
        const [icn, text] = items[k], y = y0 + 1.0 + k * 0.6;
        await ico(s, icn, HEX[tone], colX(i) + 0.18, y + 0.04, 0.32, text);
        T(s, text, { x: colX(i) + 0.62, y, w: cw - 0.72, h: 0.5, fontSize: 10.5, color: C.text1, valign: "middle" });
      }
    };
    await rows(0, [["FaUpload", "School CSV upload, checked row by row"], ["FaUsers", "Attendees per school"], ["FaBus", "Bus sizes and prices"], ["FaSearchLocation", "Venue: search any address"]], "navy");
    const models = [["FaCoins", "Fixed price per bus", "red"], ["FaRoute", "Price per mile", "blue"], ["FaClock", "Price per driving hour", "red"]];
    for (let k = 0; k < models.length; k++) {
      const [icn, text, tone] = models[k], y = y0 + 1.0 + k * 0.78, x = colX(1) + 0.14;
      card(s, x, y, cw - 0.28, 0.66, TINT[tone], text);
      await ico(s, icn, HEX[tone], x + 0.12, y + 0.17, 0.32, text);
      T(s, text, { x: x + 0.54, y, w: cw - 0.9, h: 0.66, fontSize: 11, bold: true, color: HEX[tone], valign: "middle" });
    }
    const checks = ["Every school allocated", "No school split", "No bus over capacity", "Every attendee seated", "Costs recalculated"];
    for (let k = 0; k < checks.length; k++) {
      const y = y0 + 0.98 + k * 0.48;
      await ico(s, "FaCheckCircle", H.accent3, colX(2) + 0.18, y + 0.06, 0.28, "Check");
      T(s, checks[k], { x: colX(2) + 0.56, y, w: cw - 0.66, h: 0.4, fontSize: 10.5, valign: "middle" });
    }
    const map = path.join(SHOTS, "map-all.png");
    const [mw, mh] = pngSize(map);
    const iw3 = cw - 0.28, ih3 = iw3 * mh / mw;
    frame(s, colX(3) + 0.14, y0 + 0.98, iw3, ih3);
    img(s, map, colX(3) + 0.14, y0 + 0.98, iw3, ih3, "Campus Cruise route map: 25 buses from 60 schools to Principality Stadium");
    T(s, `${F.optimised_buses} buses, one colour each; star = venue`, { x: colX(3) + 0.14, y: y0 + 1.04 + ih3, w: iw3, h: 0.5, fontSize: 10, color: C.text2 });
    await rows(4, [["FaFileCsv", "Plan CSV: bus, pickup order, cost share"], ["FaFileAlt", "Report and audit report"], ["FaMapMarkerAlt", "Clear pickup order per bus"], ["FaHandshake", "Ready for booking"]], "navy");
    // Goal + benefits banner
    card(s, 0.55, 6.15, 12.23, 0.85, TINT.red, "Goal and benefits");
    await ico(s, "FaBullseye", H.accent1, 0.8, 6.3, 0.55, "Goal");
    T(s, [{ text: "Our goal: ", options: { bold: true, color: H.accent1 } }, { text: "the cheapest plan that keeps every rule, for any schools and any venue." }],
      { x: 1.5, y: 6.15, w: 4.8, h: 0.85, fontSize: 13, valign: "middle" });
    const ben = [["FaArrowDown", "Lower cost", "red"], ["FaUsers", "Fuller buses", "navy"], ["FaLeaf", "Fewer miles", "green"], ["FaClock", "Plan in seconds", "navy"]];
    for (let k = 0; k < ben.length; k++) {
      const x = 6.55 + k * 1.55;
      await ico(s, ben[k][0], HEX[ben[k][2]], x, 6.38, 0.38, ben[k][1]);
      T(s, ben[k][1], { x: x + 0.46, y: 6.15, w: 1.05, h: 0.85, fontSize: 11.5, bold: true, valign: "middle" });
    }
    notes(s, "0:55-1:25 (30 s)",
      "Campus Cruise works in five steps. Load the schools: the bundled data, or upload any school CSV, and search any venue by address. Pick how buses are priced. OR-Tools decides which schools share each bus and the pickup order. Then separate code re-checks every rule and recalculates every cost; a plan that fails is never shown. Finally the plan goes to a dashboard and map, and can be downloaded for booking.",
      "Every plan is optimised, then independently checked, before anyone sees it, for any schools and any venue.");
  }

  // ======================= 4. Models =======================
  pres.addSection({ title: "Approach and results" });
  {
    const s = pres.addSlide({ masterName: "CC_CONTENT", sectionTitle: "Approach and results" });
    headerPhoto(s, "photo_models.png", 2.05);
    title(s, "Optimisation Models\n", "and AI Approach", "ALGORITHMS AND AI APPROACH");
    sub(s, "One general engine, three pricing rules: no retraining for new schools or venues.", 8.0, 2.05);
    const models = [
      { tone: "red", n: 1, head: "Fixed Cost Model", obj: "Minimise total bus rental price", icn: "FaCoins",
        pts: ["OR-Tools CP-SAT constraint solver", "Exact bin packing: chooses bus sizes and groups", "Location does not change the price", "Can prove no cheaper plan exists"],
        big: `${n0(F.optimised_cost)}`, base: `vs ${n0(F.baseline_cost)} booked separately`, tag: "Proven optimal" },
      { tone: "blue", n: 2, head: "Distance Model", obj: "Minimise rate × miles", icn: "FaRoute",
        pts: ["OR-Tools vehicle routing", "Cheapest-arc start + guided local search", "Orders the pickups on each bus", "Miles are straight-line (haversine)"],
        big: n0(D.optimised_cost), base: `vs ${n0(D.baseline_cost)} booked separately`, tag: `Best found in ${n0(D.time_limit_s)} s` },
      { tone: "green", n: 3, head: "Travel-time Model", obj: "Minimise rate × driving hours", icn: "FaClock",
        pts: ["Same routing search", "OSRM road driving times, cached", "If OSRM is unavailable: labelled estimate", "No live traffic or boarding time"],
        big: n0(TT.optimised_cost), base: `vs ${n0(TT.baseline_cost)} booked separately`, tag: `Best found in ${n0(TT.time_limit_s)} s` },
    ];
    const cw = 3.95, y0 = 2.72, ch = 4.15;
    for (let i = 0; i < 3; i++) {
      const m = models[i], x = 0.55 + i * (cw + 0.19), col = HEX[m.tone];
      card(s, x, y0, cw, ch, TINT[m.tone], m.head);
      num(s, x + 0.2, y0 + 0.2, m.n, col, 0.58, 20);
      T(s, m.head, { x: x + 0.92, y: y0 + 0.17, w: cw - 1.65, h: 0.36, fontSize: 16, bold: true });
      T(s, m.obj, { x: x + 0.92, y: y0 + 0.53, w: cw - 1.65, h: 0.3, fontSize: 10.5, color: col, bold: true });
      await ico(s, m.icn, col, x + cw - 0.68, y0 + 0.22, 0.48, m.icn);
      s.addShape(pres.shapes.LINE, { x: x + 0.2, y: y0 + 0.98, w: cw - 0.4, h: 0, line: { color: "D9DEE6", width: 0.75 } });
      T(s, m.pts.map((p, k) => ({ text: p, options: { bullet: { indent: 14 }, breakLine: k < m.pts.length - 1 } })),
        { x: x + 0.2, y: y0 + 1.12, w: cw - 0.35, h: 1.8, fontSize: 13, paraSpaceAfter: 6, color: C.text1 });
      card(s, x + 0.2, y0 + 3.05, cw - 0.4, 0.9, H.lt1, `${m.head} result`, "E6E9EF");
      T(s, m.big, { x: x + 0.38, y: y0 + 3.09, w: 1.55, h: 0.55, fontSize: 26, bold: true, color: col, valign: "middle" });
      T(s, m.tag, { x: x + 1.95, y: y0 + 3.11, w: cw - 2.25, h: 0.5, fontSize: 11, bold: true, color: C.text1, valign: "middle" });
      T(s, m.base, { x: x + 0.38, y: y0 + 3.61, w: cw - 0.7, h: 0.28, fontSize: 10, color: C.text2 });
    }
    notes(s, "1:25-2:10 (45 s)",
      "The challenge prices buses three ways. With a fixed price per bus, location doesn't matter, so it's a packing problem: OR-Tools' CP-SAT solver finds the cheapest plan and proves nothing cheaper exists. When price depends on miles or driving time, pickup order matters, so we use vehicle-routing search. It finds very good plans, but we never call them optimal. The rules are hard in every model. This is optimisation, not a trained model, so it works for any schools and any venue without retraining. Generative AI is a separate assistant, and the machine-learning model is an experimental offline fallback.",
      "Exact optimisation where it can be proven; honest 'best found' labels where it cannot.");
  }

  // ======================= 5. Results =======================
  {
    const s = pres.addSlide({ masterName: "CC_CONTENT", sectionTitle: "Approach and results" });
    headerPhoto(s, "photo_results.png", 2.05);
    title(s, "Significant Improvements in\n", "Transport Efficiency", "RESULTS AND PERFORMANCE");
    sub(s, "Fixed-price plan on the challenge data: fewer, fuller buses, every school kept together.", 8.0, 2.05);
    const kpi = [
      ["FaCoins", "red", "Total transport cost", `↓ ${n1(F.saving_percent)}%`, `${n0(F.baseline_cost)} → ${n0(F.optimised_cost)} cost units`],
      ["FaBus", "blue", "Number of buses", `${F.baseline_buses} → ${F.optimised_buses}`, `${n1(busCut)}% fewer buses`],
      ["FaUsers", "green", "Attendees", n0(F.attendees), `all seated, from ${F.schools_served} schools`],
      ["FaChair", "orange", "Seats filled", `${n1(F.optimised_utilisation_percent)}%`, `was ${n1(F.baseline_utilisation_percent)}% booked separately`],
    ];
    for (let i = 0; i < kpi.length; i++) {
      const [icn, tone, lab, big, small] = kpi[i], x = 0.55 + i * 3.1;
      card(s, x, 2.72, 2.95, 1.32, TINT[tone], lab);
      await ico(s, icn, HEX[tone], x + 0.2, 3.05, 0.62, lab);
      T(s, lab, { x: x + 1.0, y: 2.82, w: 1.9, h: 0.28, fontSize: 12, bold: true });
      T(s, big, { x: x + 1.0, y: 3.1, w: 1.9, h: 0.52, fontSize: 25, bold: true, color: HEX[tone] });
      T(s, small, { x: x + 1.0, y: 3.62, w: 1.9, h: 0.36, fontSize: 10, color: C.text2 });
    }
    const chartBox = (x, w, ttl) => { softCard(s, x, 4.2, w, 2.8, H.lt1, ttl); };
    const chartOpts = (x, w, ttl, fmt, max) => ({
      x: x + 0.1, y: 4.27, w: w - 0.2, h: 2.66, barDir: "col", barGapWidthPct: 70,
      chartColors: [H.accent1, H.dk1], showTitle: true, title: ttl, titleFontSize: 12, titleColor: H.dk1, titleFontFace: "+mn-lt", titleBold: true,
      showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 12, dataLabelColor: H.dk1, dataLabelFormatCode: fmt, dataLabelFontFace: "+mn-lt", dataLabelFontBold: true,
      catAxisLabelColor: H.accent6, catAxisLabelFontSize: 10, catAxisLabelFontFace: "+mn-lt",
      valAxisHidden: true, valGridLine: { style: "none" }, catGridLine: { style: "none" }, showLegend: false, valAxisMinVal: 0, valAxisMaxVal: max,
    });
    chartBox(0.55, 3.0, "Cost chart");
    s.addChart(pres.charts.BAR, [{ name: "Total cost", labels: ["Booked separately", "Campus Cruise"], values: [F.baseline_cost, F.optimised_cost] }],
      chartOpts(0.55, 3.0, "Total cost (cost units)", "#,##0", 4200));
    chartBox(3.7, 3.0, "Bus chart");
    s.addChart(pres.charts.BAR, [{ name: "Buses", labels: ["Booked separately", "Campus Cruise"], values: [F.baseline_buses, F.optimised_buses] }],
      chartOpts(3.7, 3.0, "Number of buses", "0", 76));
    // Key metrics table
    softCard(s, 6.85, 4.2, 5.93, 2.8, H.lt1, "Metrics table");
    const hd = (t, al) => ({ text: t, options: { bold: true, color: H.lt1, fill: { color: H.dk1 }, fontSize: 11, align: al || "center", valign: "middle" } });
    const c = (t, o = {}) => ({ text: t, options: { fontSize: 11, color: H.dk1, align: "center", valign: "middle", ...o } });
    const good = (t) => c(t, { bold: true, color: H.accent3 });
    s.addTable([
      [hd("Metric", "left"), hd("Separate"), hd("Campus Cruise"), hd("Change")],
      [c("Total cost", { align: "left" }), c(n0(F.baseline_cost)), c(n0(F.optimised_cost)), good(`−${n1(F.saving_percent)}%`)],
      [c("Buses", { align: "left" }), c(String(F.baseline_buses)), c(String(F.optimised_buses)), good(`−${F.baseline_buses - F.optimised_buses}`)],
      [c("Seats filled", { align: "left" }), c(`${n1(F.baseline_utilisation_percent)}%`), c(`${n1(F.optimised_utilisation_percent)}%`), good(`+${n1(F.optimised_utilisation_percent - F.baseline_utilisation_percent)} pp`)],
      [c("Cost per attendee", { align: "left" }), c(n2(F.cost_per_student_before)), c(n2(F.cost_per_student_after)), good(`−${n0((1 - F.cost_per_student_after / F.cost_per_student_before) * 100)}%`)],
      [c("Bus-miles (straight line)", { align: "left" }), c(n1(F.baseline_bus_miles)), c(n1(F.plan_bus_miles)), good(`−${n1(F.bus_miles_saved_percent)}%`)],
    ], { x: 7.0, y: 4.33, w: 5.63, colW: [1.85, 1.33, 1.25, 1.2], rowH: 0.33, border: { type: "solid", pt: 0.5, color: "E3E6EA" }, margin: 0.05, fill: { color: H.lt1 } });
    T(s, `Other models (best found in ${n0(D.time_limit_s)} s): per mile −${n1(D.saving_percent)}%, per driving hour −${n1(TT.saving_percent)}%. Challenge data only.`,
      { x: 7.0, y: 6.6, w: 5.63, h: 0.34, fontSize: 9.5, color: C.text2 });
    notes(s, "2:10-2:45 (35 s)",
      `With a fixed price per bus, the plan costs ${n0(F.optimised_cost)} instead of ${n0(F.baseline_cost)}: ${n1(F.saving_percent)}% less, with ${F.optimised_buses} buses instead of ${F.baseline_buses}, and ${n1(F.optimised_utilisation_percent)}% of seats filled. That is provably the cheapest. Bus-miles drop ${n1(F.bus_miles_saved_percent)}%. The other two models save ${n1(D.saving_percent)}% and ${n1(TT.saving_percent)}%; those are the best plans found in five minutes. These are cost units, not pounds.`,
      `${n1(F.saving_percent)}% cheaper and ${F.baseline_buses - F.optimised_buses} fewer buses, proven optimal for fixed pricing.`);
  }

  // ======================= 6. In action =======================
  pres.addSection({ title: "Demo" });
  {
    const s = pres.addSlide({ masterName: "CC_CONTENT", sectionTitle: "Demo" });
    headerPhoto(s, "photo_features.png", 2.0);
    title(s, "Campus Cruise ", "in Action", "APPLICATION AND KEY FEATURES");
    sub(s, "A working web application to plan, optimise and check shared school transport for any event.", 8.3, 1.62);
    const feats = [
      ["FaUpload", "red", "Upload school data", "Any CSV; bad rows rejected by row number"],
      ["FaSearchLocation", "blue", "Search any venue", "Type an address; coordinates fill in"],
      ["FaCogs", "green", "Choose the objective", "Fixed price, per mile or per hour"],
      ["FaMapMarkedAlt", "orange", "Interactive route map", "Every bus, its schools and road path"],
      ["FaDownload", "purple", "Results and export", "Plan CSV, report and audit report"],
    ];
    for (let i = 0; i < feats.length; i++) {
      const [icn, tone, head, body] = feats[i], y = 2.42 + i * 0.92;
      card(s, 0.55, y, 4.15, 0.8, TINT[tone], head);
      circle(s, 0.7, y + 0.1, 0.6, HEX[tone], head);
      await ico(s, icn, H.lt1, 0.85, y + 0.25, 0.3, head);
      T(s, head, { x: 1.45, y: y + 0.08, w: 3.15, h: 0.32, fontSize: 14, bold: true });
      T(s, body, { x: 1.45, y: y + 0.4, w: 3.15, h: 0.32, fontSize: 11, color: C.text2 });
    }
    const panel = (x, w, label) => {
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 2.42, w, h: 4.58, rectRadius: 0.1, fill: { color: H.lt1 }, line: { color: H.dk1, width: 1 }, objectName: label });
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 2.42, w, h: 0.5, rectRadius: 0.1, fill: { color: H.dk1 }, line: { color: H.dk1 } });
      s.addShape(pres.shapes.RECTANGLE, { x, y: 2.7, w, h: 0.22, fill: { color: H.dk1 }, line: { color: H.dk1 } });
      T(s, label, { x: x + 0.18, y: 2.42, w: w - 0.3, h: 0.5, fontSize: 14, bold: true, color: C.background1, valign: "middle" });
    };
    // Sidebar inputs screenshot
    panel(4.95, 2.35, "Inputs");
    const side = path.join(ASSETS, "side-crop.png");
    const [sw, sh] = pngSize(side);
    const sih = 3.9, siw = sih * sw / sh;
    img(s, side, 4.95 + (2.35 - siw) / 2, 3.0, siw, sih, "Campus Cruise sidebar: school data upload and destination search with a map of the chosen venue");
    // Route map screenshot
    panel(7.55, 5.23, "Optimised route map");
    const map = path.join(SHOTS, "map-all.png");
    const [mw, mh] = pngSize(map);
    const mih = 3.95, miw = Math.min(5.0, mih * mw / mh);
    img(s, map, 7.55 + (5.23 - miw) / 2, 3.0, miw, miw * mh / mw, "Campus Cruise route map: 25 buses from 60 schools to Principality Stadium");
    T(s, "Real screenshots, 28 Oct 2026", { x: 10.2, y: 2.42, w: 2.42, h: 0.5, fontSize: 9.5, italic: true, color: "B8C2D1", align: "right", valign: "middle" });
    notes(s, "2:45-4:05 (80 s, live)",
      "[Switch to the app at http://localhost:8501]\n1. (15 s) Sidebar: Fixed price per bus, Run optimiser. Point at 'Proven optimal', 25 buses, 1,875 vs 3,330.\n2. (20 s) Routes tab: Find a school, CF15, turn on Road paths. 'Bus 04 picks up four schools and is completely full, 50 of 50. The line is the real road route from OSRM.' Read the note: estimates exclude live traffic and boarding time.\n3. (25 s) Destination: type 'Wembley Stadium', Find and use. 'Any venue: the coordinates fill in, the map moves, and every cost and route now goes there.' Run again, show routes ending at Wembley. Then Reset to Principality Stadium.\n4. (20 s) School data: 'Organisers can upload their own schools as a CSV; a bad row is rejected with its row number, never silently dropped.'\nIf the demo fails: stay on this slide and talk through the screenshots.",
      "It's a working app for any schools and any venue, with honest estimates.");
  }

  // ======================= 7. Assistant =======================
  {
    const s = pres.addSlide({ masterName: "CC_CONTENT", sectionTitle: "Demo" });
    headerPhoto(s, "photo_models.png", 2.05);
    title(s, "AI Assistant: ", "You Approve", "GENERATIVE AI WITH HUMAN CONTROL");
    sub(s, "Organisers type what they want in plain English. The assistant cannot change anything on its own.", 8.3, 1.62);
    const steps = [
      ["FaKeyboard", "navy", "You type a request", "\"Minimise bus costs while keeping nearby schools together.\""],
      ["FaRobot", "blue", "GPT-4.1 mini proposes", "Structured JSON only: no tools, no direct changes"],
      ["FaShieldAlt", "green", "Code checks the proposal", "Every school, number and rule against the data"],
      ["FaUserCheck", "red", "You approve or reject", "Nothing changes until you approve"],
      ["FaCogs", "navy", "Optimiser and validator run", "The same OR-Tools models and checks as every plan"],
    ];
    for (let i = 0; i < steps.length; i++) {
      const [icn, tone, head, body] = steps[i], y = 2.45 + i * 0.85;
      softCard(s, 0.55, y, 7.0, 0.74, i === 3 ? TINT.red : H.lt1, head);
      num(s, 0.72, y + 0.13, i + 1, i === 3 ? H.accent1 : H.dk1, 0.48, 15);
      await ico(s, icn, HEX[tone], 1.38, y + 0.19, 0.36, head);
      T(s, head, { x: 1.95, y: y + 0.06, w: 5.4, h: 0.32, fontSize: 14, bold: true, color: i === 3 ? C.accent1 : C.text1 });
      T(s, body, { x: 1.95, y: y + 0.38, w: 5.4, h: 0.3, fontSize: 11.5, color: C.text2 });
    }
    const shotP = path.join(SHOTS, "crop-assistant.png");
    const [aw, ah] = pngSize(shotP);
    const sh = 4.5, swd = sh * aw / ah;
    const sx = 8.0 + (4.78 - swd) / 2;
    frame(s, sx, 2.45, swd, sh);
    img(s, shotP, sx, 2.45, swd, sh, "Campus Cruise assistant panel answering 'Minimise bus costs while keeping nearby schools together' with a proposal awaiting approval");
    T(s, "Real screenshot", { x: sx, y: 6.98, w: swd, h: 0.18, fontSize: 9, italic: true, color: C.accent6, align: "center" });
    notes(s, "4:05-4:40 (35 s)",
      "Organisers can also just type: 'Minimise bus costs while keeping nearby schools together.' GPT-4.1 mini turns that into a proposal, but it can't change anything. Code checks it against the data and shows it: nothing has changed yet. Only when I approve does the optimiser run. And it's honest: 'nearby' isn't a hard rule, it's how the fixed model breaks ties, and it says so.",
      "Generative AI makes the tool easy to use, but people and the optimiser stay in control.");
  }

  // ======================= 8. Impact =======================
  pres.addSection({ title: "Impact and responsibility" });
  {
    const s = pres.addSlide({ masterName: "CC_CONTENT", sectionTitle: "Impact and responsibility" });
    headerPhoto(s, "photo_results.png", 2.05);
    title(s, "Sector Impact: ", "Who Benefits", "IMPACT AND BENEFICIARIES");
    sub(s, "Measured on the challenge data; the next step is to measure it at a real event.", 8.3, 1.62);
    const who = [
      ["FaClipboardList", "red", "Event organisers", "A checked plan in seconds"],
      ["FaSchool", "blue", "Schools and teachers", "Lower, fair cost shares"],
      ["FaChild", "green", "Pupils", "Lower cost to attend"],
      ["FaBus", "orange", "Coach operators", "A clear booking list"],
      ["FaLandmark", "purple", "Event venues", "Fewer coaches to park"],
      ["FaUsers", "blue", "Local communities", "Less coach traffic"],
    ];
    for (let i = 0; i < who.length; i++) {
      const [icn, tone, head, body] = who[i], x = 0.55 + (i % 3) * 2.42, y = 2.45 + Math.floor(i / 3) * 1.85;
      softCard(s, x, y, 2.27, 1.7, H.lt1, head);
      await badge(s, icn, tone, x + 0.18, y + 0.2, 0.7);
      T(s, head, { x: x + 0.18, y: y + 1.05, w: 2.0, h: 0.28, fontSize: 13, bold: true });
      T(s, body, { x: x + 0.18, y: y + 1.33, w: 2.0, h: 0.28, fontSize: 12, color: C.text2 });
    }
    const mets = [
      [`−${F.baseline_buses - F.optimised_buses}`, "buses to book and coordinate", "red"],
      [`${n1(F.optimised_utilisation_percent)}%`, `seats filled (was ${n1(F.baseline_utilisation_percent)}%)`, "blue"],
      [`−${n1(F.bus_miles_saved_percent)}%`, `bus-miles, ${n0(F.baseline_bus_miles)} → ${n0(F.plan_bus_miles)} (straight line)`, "green"],
    ];
    mets.forEach(([big, small, tone], i) => {
      const y = 2.45 + i * 1.2;
      card(s, 8.0, y, 4.78, 1.08, TINT[tone], `Metric ${i + 1}`);
      T(s, big, { x: 8.2, y, w: 1.55, h: 1.08, fontSize: 30, bold: true, color: HEX[tone], valign: "middle" });
      T(s, small, { x: 9.8, y, w: 2.85, h: 1.08, fontSize: 13, valign: "middle" });
    });
    T(s, "Fixed-price plan; baseline = one bus per school. No fuel or emissions claim.", { x: 8.0, y: 6.1, w: 4.78, h: 0.35, fontSize: 10, color: C.text2 });
    notes(s, "4:40-5:15 (35 s)",
      `Who gains? Organisers get a checked plan in seconds. Schools and pupils pay less to attend. Coach operators get a clear booking list, and venues have fewer coaches to park. On the challenge data that's ${F.baseline_buses - F.optimised_buses} fewer buses, ${n1(F.optimised_utilisation_percent)}% of seats filled, and about ${n0(F.bus_miles_saved_percent)}% fewer bus-miles, straight-line, with no emissions claim. And because it takes any schools and any venue, it isn't tied to one stadium.`,
      "Measurable savings on the data, and a clear plan to prove them in the real world.");
  }

  // ======================= 9. Responsible AI =======================
  {
    const s = pres.addSlide({ masterName: "CC_CONTENT", sectionTitle: "Impact and responsibility" });
    headerPhoto(s, "photo_models.png", 2.05);
    title(s, "Responsible AI ", "by Design", "RESPONSIBLE AI");
    sub(s, "Decision support, not an autonomous system: people approve, code verifies.", 8.3, 1.62);
    const pr = [
      ["FaEye", "blue", "Transparency", "Every plan labelled proven optimal or best found; estimates marked"],
      ["FaLock", "green", "Privacy", "School-level head counts only, no pupil names; address search sends only the typed text"],
      ["FaShieldAlt", "red", "Security", "API keys never shown or logged; the LLM has no tools"],
      ["FaUserCheck", "orange", "Accountability", "Organiser approves every AI change; audit report of every run"],
      ["FaBalanceScale", "purple", "Safety and fairness", "Rules re-checked by independent code; cost shared per attendee"],
      ["FaExclamationTriangle", "red", "Honest limits", "No live traffic or boarding time; not yet tested at a real event"],
    ];
    for (let i = 0; i < pr.length; i++) {
      const [icn, tone, head, body] = pr[i], x = 0.55 + (i % 3) * 4.14, y = 2.45 + Math.floor(i / 3) * 2.2;
      softCard(s, x, y, 3.95, 2.0, H.lt1, head);
      await badge(s, icn, tone, x + 0.25, y + 0.28, 0.8);
      T(s, head, { x: x + 1.25, y: y + 0.28, w: 2.55, h: 0.36, fontSize: 17, bold: true, color: HEX[tone] });
      T(s, body, { x: x + 1.25, y: y + 0.72, w: 2.55, h: 1.15, fontSize: 13, color: C.text1 });
    }
    notes(s, "5:15-5:50 (35 s)",
      "Campus Cruise is decision support, not an autonomous system. Every plan says whether it's proven or just the best found. It uses school-level head counts only, no names, and the address search sends only the typed text. API keys are never shown, and the language model has no tools. The organiser approves every AI change, every run is in an audit report, and independent code re-checks every rule. And we are honest about limits: no live traffic or boarding time, and it has not yet run at a real event.",
      "Responsible by design: people approve, code verifies.");
  }

  // ======================= 10. Conclusion (dark) =======================
  pres.addSection({ title: "Close" });
  {
    const s = pres.addSlide({ masterName: "CC_DARK", sectionTitle: "Close" });
    const hero = path.join(ASSETS, "photo_dark.png");
    const [iw, ih] = pngSize(hero);
    T(s, "CONCLUSION", { x: 0.6, y: 0.45, w: 6, h: 0.3, fontSize: 13, bold: true, color: "B8C2D1", charSpacing: 1 });
    s.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: 0.82, w: 0.55, h: 0.07, fill: { color: H.accent1 }, line: { color: H.accent1 } });
    T(s, [{ text: "Fewer buses, lower cost,\n", options: { color: "FFFFFF" } }, { text: "plans people can trust", options: { color: "FF5A6A" } }],
      { x: 0.6, y: 1.0, w: 7.2, h: 1.35, fontSize: 36, bold: true });
    const ph = 2.6, pw = ph * iw / ih;
    img(s, hero, W - pw - 0.55, 0.45, pw, ph, "Coaches outside Principality Stadium (illustration from the team's reference slides)", { rounding: false });
    const pts = [
      ["FaQuestion", "red", "Problem", "Schools booking alone waste seats and money"],
      ["FaTrophy", "orange", "Result", `${n1(F.saving_percent)}% cheaper, ${F.baseline_buses} → ${F.optimised_buses} buses, proven optimal`],
      ["FaCogs", "blue", "Contribution", "Exact + routing optimisation, independent checks, any schools and venue"],
      ["FaFlag", "green", "Next step", "Pilot with a Welsh organiser; collect feedback"],
    ];
    for (let i = 0; i < pts.length; i++) {
      const [icn, tone, head, body] = pts[i], x = 0.6 + i * 3.06;
      card(s, x, 3.35, 2.88, 1.85, "15305F", head);
      await ico(s, icn, tone === "red" ? "FF5A6A" : HEX[tone], x + 0.2, 3.52, 0.4, head);
      T(s, head, { x: x + 0.72, y: 3.5, w: 2.0, h: 0.44, fontSize: 16, bold: true, color: C.background1, valign: "middle" });
      T(s, body, { x: x + 0.2, y: 4.1, w: 2.5, h: 1.0, fontSize: 13, color: "DDE3EC" });
    }
    T(s, "Thank you. Questions?", { x: 0.6, y: 5.75, w: 12.1, h: 0.7, fontSize: 32, bold: true, color: "FF5A6A" });
    notes(s, "5:50-6:10 (20 s)",
      `Campus Cruise finds which schools should share a bus: ${n1(F.saving_percent)}% cheaper on the challenge data, ${F.optimised_buses} buses instead of ${F.baseline_buses}, proven optimal and independently checked, and it now works for any schools and any venue. Next, we'd like to pilot it with a Welsh event organiser.. Thank you, we're happy to take questions.`,
      "Campus Cruise: fewer buses, lower cost, and plans organisers can trust.");
  }

  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})().catch((e) => { console.error(e); process.exit(1); });
