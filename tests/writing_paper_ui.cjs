"use strict";
const { chromium } = require("playwright");
const assert = require("node:assert/strict");
const fs = require("node:fs/promises");
const path = require("node:path");
const base = process.env.WRITING_TEST_URL || "http://127.0.0.1:8765";
const output = path.resolve("outputs/writing/2026-10-12-landscape-ui");
const waitState = (page, expected, id) => page.waitForFunction(({ state, jobId }) => {
  const status = document.getElementById("state"); return status.dataset.state === state && (!jobId || status.dataset.jobId === jobId);
}, { state: expected, jobId: id }, { timeout: 30000 });

async function readPage(context) { return (await context.request.get(`${base}/api/page`)).json(); }
async function newSheet(page) {
  await page.waitForFunction(() => !document.getElementById("new-page").disabled);
  page.once("dialog", dialog => dialog.accept());
  const response = page.waitForResponse(r => r.url().endsWith("/api/page/reset") && r.request().method() === "POST");
  await page.locator("#new-page").click();
  assert.equal((await response).status(), 200);
  await page.waitForFunction(() => !document.getElementById("start").disabled);
}
async function submit(page, text, mm) {
  await page.locator("#text").fill(text); await page.locator("#character-mm").fill(String(mm));
  const accepted = page.waitForResponse(r => r.url() === `${base}/api/jobs` && r.request().method() === "POST");
  await page.locator("#start").click();
  const response = await accepted;
  assert.equal(response.status(), 202, await response.text());
  return response.json();
}
async function layout(page) {
  assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), "Horizontal overflow");
  const clipped = await page.locator("button,select,h1,h2,.metrics dd,.sheet-info dd").evaluateAll(elements => elements.filter(e => e.scrollWidth > e.clientWidth + 1).map(e => e.id || e.tagName));
  assert.deepEqual(clipped, [], "Clipped text");
}

(async () => {
  await fs.mkdir(output, { recursive: true });
  const browser = await chromium.launch({ channel: "msedge", headless: true });
  const errors = [], results = [];
  try {
    for (const viewport of [{ width: 1280, height: 1000 }, { width: 390, height: 844 }]) {
      const context = await browser.newContext({ viewport, acceptDownloads: true });
      const page = await context.newPage();
      page.on("pageerror", error => errors.push(error.message));
      await page.goto(`${base}/jobs`); await newSheet(page);
      const fresh = await readPage(context);
      assert.equal(fresh.paper.width_mm, 297); assert.equal(fresh.paper.height_mm, 210);
      assert.deepEqual(fresh.working_rect_mm, [8, 8, 289, 202]);
      const ratio = await page.locator("#page-map").evaluate(canvas => canvas.getBoundingClientRect().width / canvas.getBoundingClientRect().height);
      assert(Math.abs(ratio - 297 / 210) < .02, "Landscape map is distorted");
      await page.locator("#text").fill("네"); await page.locator("#character-mm").fill("18");
      await page.waitForFunction(() => document.getElementById("placement-status").textContent.includes("y 11.8mm"));
      assert.equal((await readPage(context)).reservations.length, 0, "Preview reserved real space");
      const first = await submit(page, "네", 18); await waitState(page, "completed", first.job_id);
      const second = await submit(page, "네", 18); await waitState(page, "completed", second.job_id);
      const a = first.placement.reserved_rect_mm, b = second.placement.reserved_rect_mm;
      assert(a[3] < b[1], "Consecutive rows overlap");
      const occupied = await readPage(context);
      assert.equal(occupied.reservations.filter(r => r.state === "occupied").length, 2);
      const greenPixels = await page.locator("#page-map").evaluate(canvas => {
        const data = canvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height).data; let count = 0;
        for (let i = 0; i < data.length; i += 4) if (data[i] === 181 && data[i + 1] === 217 && data[i + 2] === 193) count++;
        return count;
      });
      assert(greenPixels > 10000, "Occupancy map is blank");
      const unreachable = await submit(page, "가".repeat(24), 28); await waitState(page, "failed", unreachable.job_id);
      const unreachableResult = await (await context.request.get(`${base}/api/jobs/${unreachable.job_id}`)).json();
      assert.equal(unreachableResult.failure_category, "robot_reach");
      assert.equal(unreachableResult.page_reservation_state, "released_no_motion");
      assert((await page.locator("#failure").textContent()).includes("로봇 도달 불가"));
      assert.equal((await readPage(context)).state, "ready", "Reach failure incorrectly marked paper full");
      const denied = await context.request.post(`${base}/api/jobs`, { data: { text: "가".repeat(120), character_mm: 40, page_id: fresh.page_id } });
      assert.equal(denied.status(), 409);
      const paperFailure = await denied.json(); assert.equal(paperFailure.category, "paper_space");
      assert.equal(paperFailure.reason, "text_does_not_fit_blank_sheet");
      await page.locator("#text").fill("가".repeat(120)); await page.locator("#character-mm").fill("40");
      await page.waitForFunction(() => document.getElementById("placement-status").textContent === "종이 공간 부족");
      await page.locator("#start").click();
      await page.waitForFunction(() => document.getElementById("error").textContent.includes("종이 공간 부족"));
      await page.locator("#text").fill("네"); await page.locator("#character-mm").fill("18");
      await page.reload();
      await page.waitForFunction(() => document.getElementById("sheet-state").textContent === "준비");
      const restored = await readPage(context);
      assert.equal(restored.page_id, fresh.page_id); assert.equal(restored.next_row_y_mm, occupied.next_row_y_mm);
      await layout(page);
      await page.screenshot({ path: path.join(output, `${viewport.width}-full-sheet.png`), fullPage: true });
      page.once("dialog", dialog => dialog.dismiss()); await page.locator("#new-page").click();
      assert.equal((await readPage(context)).page_id, fresh.page_id, "Cancelled confirmation reset the sheet");
      await newSheet(page); const newPage = await readPage(context);
      assert.notEqual(newPage.page_id, fresh.page_id); assert.equal(newPage.reservations.length, 0);
      const stale = await context.request.post(`${base}/api/page/reset`, { data: { page_id: fresh.page_id, confirmed: true } });
      assert.equal(stale.status(), 409);
      const staleJob = await context.request.post(`${base}/api/jobs`, { data: { text: "네", character_mm: 18, page_id: fresh.page_id } });
      assert.equal(staleJob.status(), 409);
      const third = await submit(page, "네", 18); await waitState(page, "completed", third.job_id);
      assert.deepEqual(third.placement.reserved_rect_mm, first.placement.reserved_rect_mm, "New sheet did not restart at the first row");
      const report = await (await context.request.get(`${base}/api/jobs/${second.job_id}/artifacts/report.json`)).json();
      assert.equal(report.trajectory_complete, true); assert.equal(report.motion_authorized, false); assert.equal(report.pressure_validated, false);
      await page.screenshot({ path: path.join(output, `${viewport.width}-new-sheet.png`), fullPage: true });
      const finalJobs = await Promise.all([first, second, third].map(async job => (await context.request.get(`${base}/api/jobs/${job.job_id}`)).json()));
      assert(finalJobs.every(job => job.state === "completed" && job.page_reservation_state === "occupied"));
      results.push({ viewport, first: finalJobs[0], second: finalJobs[1], third: finalJobs[2], greenPixels, full_page: restored, reset_page: newPage, rejected_space_status: 409, stale_reset_status: 409, stale_job_status: 409,
        unreachable: unreachableResult, paper_failure: paperFailure,
        second_report_metrics: { actual_simulated_duration_s: report.actual_simulated_duration_s, max_tracking_error_mm: report.max_tracking_error_mm, max_path_error_mm: report.max_path_error_mm, dynamics_validated: report.dynamics_validated, pressure_validated: report.pressure_validated } });
      await context.close();
    }
    const context = await browser.newContext({ viewport: { width: 1280, height: 1000 } });
    const page = await context.newPage(); page.on("pageerror", e => errors.push(e.message));
    await page.goto(`${base}/jobs`); await newSheet(page);
    const stoppedJob = await submit(page, "안녕하세요.", 18);
    await waitState(page, "executing", stoppedJob.job_id);
    await page.waitForFunction(() => document.getElementById("phase").textContent === "필기");
    const blockedReset = await context.request.post(`${base}/api/page/reset`, { data: { page_id: stoppedJob.placement.page_id, confirmed: true } });
    assert.equal(blockedReset.status(), 409);
    await page.locator("#stop").click(); await waitState(page, "stopped", stoppedJob.job_id);
    const partial = await readPage(context); assert.equal(partial.reservations[0].state, "partial");
    const stopReport = await (await context.request.get(`${base}/api/jobs/${stoppedJob.job_id}/artifacts/report.json`)).json();
    assert(stopReport.stop_settled && stopReport.stop_test_passed); assert.equal(stopReport.trajectory_complete, false);
    await page.screenshot({ path: path.join(output, "1280-partial.png"), fullPage: true });
    const next = await submit(page, "네", 18); await waitState(page, "completed", next.job_id);
    assert(partial.reservations[0].reserved_rect_mm[3] < next.placement.reserved_rect_mm[1], "Next job reused partial space");
    assert.deepEqual(errors, []);
    const finalStopped = await (await context.request.get(`${base}/api/jobs/${stoppedJob.job_id}`)).json();
    const finalNext = await (await context.request.get(`${base}/api/jobs/${next.job_id}`)).json();
    await context.close();
    await fs.writeFile(path.join(output, "summary.json"), JSON.stringify({ date: "2026-10-12", cases: results,
      partial: { job: finalStopped, page: partial, next: finalNext, stop_settling_time_s: stopReport.stop_settling_time_s, stop_max_displacement_mm: stopReport.stop_max_displacement_mm }, browser_errors: errors }, null, 2));
    console.log("PASS desktop/mobile: A4 landscape, full-paper placement, robot-reach vs paper-space errors, two actual rows, reload, manual reset, stale guards, pixels, partial stop and next-row execution");
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
