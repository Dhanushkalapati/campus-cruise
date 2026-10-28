// Second pass: clean overview (no tooltip) and the assistant answering one request.
const { chromium } = require("playwright-core");
const OUT = process.argv[2];
const REQUEST = "Minimise bus costs while keeping nearby schools together.";
const step = (m) => console.log(new Date().toISOString().slice(11, 19), m);

(async () => {
  const browser = await chromium.launch({ executablePath: "C:/Program Files/Google/Chrome/Application/chrome.exe", headless: true });
  const page = await browser.newPage({ viewport: { width: 1600, height: 1000 }, deviceScaleFactor: 1.5 });
  await page.goto("http://localhost:8501", { waitUntil: "networkidle" });
  await page.getByRole("button", { name: "Run optimiser" }).click();
  await page.getByText("Proven optimal").first().waitFor({ timeout: 120000 });
  await page.mouse.move(1200, 950);
  await page.waitForTimeout(2500);
  await page.screenshot({ path: `${OUT}/app-overview.png` });
  step("overview");

  await page.getByRole("button", { name: /Assistant/ }).click();
  await page.waitForTimeout(1500);
  const box = page.getByPlaceholder("Ask about the plan...");
  await box.fill(REQUEST);
  await box.press("Enter");
  // Wait for a proposal (Approve button) or any answer text to appear.
  await Promise.race([
    page.getByRole("button", { name: "Approve and run" }).first().waitFor({ timeout: 90000 }),
    page.getByText(/not supported|preference|fixed-cost model/i).first().waitFor({ timeout: 90000 }),
  ]).catch(() => step("no proposal marker seen"));
  await page.waitForTimeout(3000);
  await page.screenshot({ path: `${OUT}/app-assistant.png` });
  step("assistant");
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
