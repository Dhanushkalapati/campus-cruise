// Screenshots for the "ai project" deck: full-plan route map and the sidebar inputs.
// Needs the app running at http://localhost:8501. Usage: node shots3.js <outdir>
const { chromium } = require("playwright-core");
const OUT = process.argv[2];
const step = (m) => console.log(new Date().toISOString().slice(11, 19), m);

(async () => {
  const browser = await chromium.launch({ executablePath: "C:/Program Files/Google/Chrome/Application/chrome.exe", headless: true });
  const page = await browser.newPage({ viewport: { width: 1600, height: 1000 }, deviceScaleFactor: 2 });
  await page.goto("http://localhost:8501", { waitUntil: "load", timeout: 60000 });
  await page.getByRole("button", { name: "Run optimiser" }).waitFor({ timeout: 60000 });
  step("loaded");

  // Sidebar: open the Destination section so search, coordinates and map preview show.
  await page.getByText("Destination · Principality Stadium").click();
  await page.waitForTimeout(4000);
  const side = page.locator('[data-testid="stSidebarContent"]');
  await side.screenshot({ path: `${OUT}/side-inputs.png` });
  step("sidebar");
  await page.getByText("Destination · Principality Stadium").click();

  await page.getByRole("button", { name: "Run optimiser" }).click();
  await page.getByText("Proven optimal").first().waitFor({ timeout: 120000 });
  await page.waitForTimeout(3000);
  step("solved");
  await page.getByRole("tab", { name: "Routes" }).click();
  await page.waitForTimeout(6000);
  const map = page.locator("iframe").first();
  await map.screenshot({ path: `${OUT}/map-all.png` });
  step("map");
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
