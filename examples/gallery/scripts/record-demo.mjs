// stanwasm gallery demo — Playwright headless capture.
//
//   npx vite preview --port 4173          # from examples/gallery, after `make gallery-build`
//   node scripts/record-demo.mjs ./out    # writes a .webm plus sections.json
//
// The webm is the raw material; `sections.json` records when each beat actually
// started and ended, which is what the OpenScreen zoom ranges are derived from.
// Do not screen-record this: synthetic input never moves the OS cursor, so
// --auto-zoom finds nothing and the cursor overlay freezes mid-screen.
// 16:9 exactly, so the OpenScreen export (aspectRatio "16:9") does not stretch it.
// Budget: ~3s per feature; the sandbox gets four beats because it is a flow
// (whole view -> graphical model -> compile -> results), not a still.
import { chromium } from "playwright";
import { writeFileSync } from "node:fs";

const OUT = process.argv[2];
const URL = process.argv[3] ?? "http://localhost:4173/";
// Record at the export resolution. Recording 1600x900 and exporting 1920x1080
// upscales every frame; deviceScaleFactor 2 then renders at 3840x2160 and
// downsamples into it, which is what keeps the small type legible.
const W = 1920, H = 1080;

const browser = await chromium.launch();
const context = await browser.newContext({
  viewport: { width: W, height: H },
  recordVideo: { dir: OUT, size: { width: W, height: H } },
  deviceScaleFactor: 2,
});
const page = await context.newPage();

const t0 = Date.now();
const at = () => (Date.now() - t0) / 1000;
const sections = [];
const section = (name) => {
  const s = sections.at(-1);
  if (s) s.end = at();
  sections.push({ name, start: at(), end: null });
  console.log(`${at().toFixed(2)}s  ── ${name}`);
};
const beat = (ms) => page.waitForTimeout(ms);
// spend the remainder of a section's budget, so each one lands on ~10s
const fill = (budget) => beat(Math.max(300, budget * 1000 - (at() - sections.at(-1).start) * 1000));

try {
  await page.goto(URL, { waitUntil: "networkidle" });
  await page.waitForSelector(".tab-btn", { timeout: 30000 });

  section("intro");
  await beat(1600);

  section("mcmc");
  await page.getByRole("button", { name: "Restart" }).click();
  await beat(1300);
  const plus = page.locator(".controls button.secondary").nth(1);
  await plus.click(); await beat(400); await plus.click();
  await fill(3);

  section("live-regression");
  await page.getByRole("button", { name: "Live Regression" }).click();
  await page.waitForSelector("svg.plot-wrap circle", { timeout: 20000 });
  await beat(900);
  // Two outliers at opposite corners, both left in place. Top-left (low x,
  // high y) and bottom-right (high x, low y) each pull against the true
  // positive slope, so they reinforce: the conjugate-normal fit collapses
  // while the Student-t fit stays on the data. Dragging one out and back, as
  // this used to, undid the effect before the viewer could read it.
  const pts = page.locator("svg.plot-wrap circle");
  const n = await pts.count();
  const plot = await page.locator("svg.plot-wrap").boundingBox();
  const corners = [
    { idx: Math.floor(n / 3), fx: 0.10, fy: 0.08 },      // top-left
    { idx: Math.floor((2 * n) / 3), fx: 0.90, fy: 0.92 }, // bottom-right
  ];
  for (const { idx, fx, fy } of corners) {
    const box = await pts.nth(idx).boundingBox();
    if (!box || !plot) continue;
    const from = { x: box.x + box.width / 2, y: box.y + box.height / 2 };
    const to = { x: plot.x + plot.width * fx, y: plot.y + plot.height * fy };
    const STEPS = 22;
    await page.mouse.move(from.x, from.y);
    await page.mouse.down();
    for (let i = 1; i <= STEPS; i++) {
      const t = i / STEPS;
      await page.mouse.move(from.x + (to.x - from.x) * t, from.y + (to.y - from.y) * t);
      await beat(30);
    }
    await page.mouse.up();
    await beat(500);
  }
  await fill(6.5);

  // The sandbox is the one section that earns more than 3s: it is a flow, not
  // a still. Whole view -> open the graphical model -> compile -> results.
  section("sandbox-overview");
  await page.getByRole("button", { name: "Wasm Sandbox" }).click();
  await page.waitForSelector(".compile-status", { timeout: 20000 });
  await fill(3.5);

  section("sandbox-diagram");
  await page.locator(".diagram-overlay").click();
  await page.waitForSelector(".diagram-overlay.expanded", { timeout: 10000 });
  await fill(4);
  await page.locator(".diagram-overlay.expanded").click();
  await beat(600);

  section("sandbox-compile");
  await page.getByRole("button", { name: /^Compile$/ }).click();
  await page.locator('button:has-text("Run NUTS"):not([disabled])')
    .waitFor({ timeout: 30000 });
  await fill(3);

  section("sandbox-result");
  await page.getByRole("button", { name: "Run NUTS" }).click();
  await page.waitForFunction(() => !/Sampling/.test(document.body.innerText), null, { timeout: 60000 }).catch(() => {});
  await beat(500);
  await page.locator("text=/Posterior summary/").first().scrollIntoViewIfNeeded().catch(() => {});
  await fill(4.5);

  section("outro");
  await beat(1800);
  sections.at(-1).end = at();
} finally {
  await context.close();
  await browser.close();
}

writeFileSync(`${OUT}/sections.json`, JSON.stringify(sections, null, 2));
console.log("");
for (const s of sections) console.log(`  ${s.name.padEnd(16)} ${s.start.toFixed(2)} → ${s.end.toFixed(2)}  (${(s.end - s.start).toFixed(1)}s)`);
console.log(`\ntotal ${at().toFixed(1)}s`);
