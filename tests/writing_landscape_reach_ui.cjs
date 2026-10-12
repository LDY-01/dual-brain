"use strict";

const { chromium } = require("playwright");
const assert = require("node:assert/strict");
const fs = require("node:fs/promises");
const path = require("node:path");
const base = process.env.WRITING_TEST_URL || "http://127.0.0.1:8765";
const output = path.resolve("outputs/writing/2026-10-12-landscape-sequence-ui");

async function submit(page, text, expected) {
  await page.locator("#text").fill(text);
  await page.locator("#character-mm").fill("28");
  const response = page.waitForResponse(r => r.url() === `${base}/api/jobs` && r.request().method() === "POST");
  await page.locator("#start").click();
  const accepted = await response;
  assert.equal(accepted.status(), 202, await accepted.text());
  const job = await accepted.json();
  await page.waitForFunction(({ id, state }) => {
    const status = document.getElementById("state");
    return status.dataset.jobId === id && status.dataset.state === state;
  }, { id: job.job_id, state: expected }, { timeout: 90000 });
  return job;
}

(async () => {
  await fs.mkdir(output, { recursive: true });
  const browser = await chromium.launch({ channel: "msedge", headless: true });
  const errors = [], cases = [];
  try {
    for (const viewport of [{ width: 1280, height: 1000 }, { width: 390, height: 844 }]) {
      const context = await browser.newContext({ viewport });
      const page = await context.newPage();
      page.on("pageerror", error => errors.push(error.message));
      await page.goto(`${base}/jobs`);
      await page.waitForFunction(() => !document.getElementById("new-page").disabled);
      page.once("dialog", dialog => dialog.accept());
      await page.locator("#new-page").click();
      await page.waitForFunction(() => !document.getElementById("start").disabled);
      const call = await submit(page, "네", "completed");
      const greeting = await submit(page, "안녕하세요", "completed");
      assert.equal(call.placement.page_id, greeting.placement.page_id);
      assert(call.placement.reserved_rect_mm[3] < greeting.placement.reserved_rect_mm[1]);
      assert.equal(greeting.placement.layout_rect_mm[1], 53.25);
      await page.waitForFunction(() => {
        const img = document.getElementById("trace"); return img.complete && img.naturalWidth === 1188;
      });
      const ink = await page.locator("#trace").evaluate(img => {
        const canvas = document.createElement("canvas"); canvas.width = img.naturalWidth; canvas.height = img.naturalHeight;
        const ctx = canvas.getContext("2d"); ctx.drawImage(img, 0, 0);
        const data = ctx.getImageData(0, 0, canvas.width, canvas.height).data; let count = 0;
        for (let i = 0; i < data.length; i += 4) if (data[i] < 80 && data[i + 1] < 80 && data[i + 2] < 80) count++;
        return count;
      });
      assert(ink > 500, "Greeting actual TCP trace is blank");
      assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
      const clipped = await page.locator("button,select,h1,h2,.metrics dd,.sheet-info dd").evaluateAll(elements => elements.filter(e => e.scrollWidth > e.clientWidth + 1).map(e => e.id || e.tagName));
      assert.deepEqual(clipped, []);
      const results = [];
      for (const job of [call, greeting]) {
        const final = await (await context.request.get(`${base}/api/jobs/${job.job_id}`)).json();
        const report = await (await context.request.get(`${base}/api/jobs/${job.job_id}/artifacts/report.json`)).json();
        assert.equal(final.state, "completed"); assert.equal(final.page_reservation_state, "occupied");
        assert(report.dynamics_validated && report.ready_return_validated && report.trajectory_complete);
        assert.equal(report.motion_authorized, false); assert.equal(report.pressure_validated, false);
        assert.deepEqual(report.paper_frame.origin_m, [.160, -.070, .005]);
        assert(report.max_tracking_error_mm <= 1 && report.max_tip_speed_mm_s <= 30);
        results.push({ job: final, metrics: { simulation_time_s: report.actual_simulated_duration_s,
          tracking_mm: report.max_tracking_error_mm, path_mm: report.max_path_error_mm, tip_speed_mm_s: report.max_tip_speed_mm_s } });
      }
      await page.screenshot({ path: path.join(output, `${viewport.width}-sequence.png`), fullPage: true });
      const third = await submit(page, "네", "failed");
      assert.equal(third.placement.page_id, call.placement.page_id, "Another test changed the sheet during the sequence");
      const failed = await (await context.request.get(`${base}/api/jobs/${third.job_id}`)).json();
      assert.equal(failed.failure_category, "robot_reach");
      assert.equal(failed.page_reservation_state, "released_no_motion");
      const sheet = await (await context.request.get(`${base}/api/page`)).json();
      assert.equal(sheet.page_id, call.placement.page_id);
      assert.equal(sheet.next_row_y_mm, 91);
      assert.equal(sheet.state, "ready");
      assert((await page.locator("#failure").textContent()).includes("로봇 도달 불가"));
      cases.push({ viewport, results, actual_ink_pixels: ink, third_row_rejected: failed, page: sheet });
      await context.close();
    }
    assert.deepEqual(errors, []);
    await fs.writeFile(path.join(output, "summary.json"), JSON.stringify({ date: "2026-10-12", cases, browser_errors: errors }, null, 2));
    console.log("PASS desktop/mobile: actual 28mm call then greeting, separate occupied rows, trace pixels, unchanged guards, unreachable third row retained as failure");
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
