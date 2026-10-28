// Captures real screenshots of the running Campus Cruise app (http://localhost:8501).
const { chromium } = require("playwright-core");
const OUT = process.argv[2];
const step = (m) => console.log(new Date().toISOString().slice(11, 19), m);

(async () => {
  const browser = await chromium.launch({ executablePath: "C:/Program Files/Google/Chrome/Application/chrome.exe", headless: true });
  const page = await browser.newPage({ viewport: { width: 1600, height: 1000 }, deviceScaleFactor: 1.5 });
  await page.goto("http://localhost:8501", { waitUntil: "networkidle" });
  await page.getByRole("button", { name: "Run optimiser" }).waitFor({ timeout: 60000 });
  step("loaded");
  await page.getByRole("button", { name: "Run optimiser" }).click();
  await page.getByText("Proven optimal").first().waitFor({ timeout: 120000 });
  await page.waitForTimeout(3000);
  step("solved");
  await page.screenshot({ path: `${OUT}/app-overview.png` });

  await page.getByRole("tab", { name: "Routes" }).click();
  await page.waitForTimeout(2000);
  // Pick one school: the map then shows only its bus, with the bus panel beside it.
  const schoolBox = page.getByText("Find a school", { exact: true }).locator("xpath=ancestor::div[@data-testid='stSelectbox']");
  await schoolBox.click();
  await page.keyboard.type(process.argv[3] || "CF15");
  await page.keyboard.press("Enter");
  await page.waitForTimeout(3000);
  await page.getByText("Road paths (OSRM)").click();
  await page.waitForTimeout(12000);
  step("routes");
  await page.screenshot({ path: `${OUT}/app-routes.png` });
  await page.screenshot({ path: `${OUT}/app-routes-full.png`, fullPage: true });
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
