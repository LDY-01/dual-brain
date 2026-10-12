"use strict";

const { chromium } = require("playwright");
const assert = require("node:assert/strict");
const fs = require("node:fs/promises");
const path = require("node:path");
const base = process.env.WRITING_TEST_URL || "http://127.0.0.1:8765";
const output = path.resolve("outputs/writing/2026-10-12-landscape-jobs-ui");
const waitState = (page, state, jobId) => page.waitForFunction(expected => {
  const status = document.getElementById("state");
  return status.dataset.state === expected.state && (!expected.jobId || status.dataset.jobId === expected.jobId);
}, { state, jobId }, { timeout: 30000 });

async function newSheet(page) {
  await page.waitForFunction(() => !document.getElementById("new-page").disabled);
  page.once("dialog", dialog => dialog.accept());
  const reset = page.waitForResponse(response => response.url().endsWith("/api/page/reset") && response.request().method() === "POST");
  await page.locator("#new-page").click();
  assert.equal((await reset).status(), 200);
  await page.waitForFunction(() => !document.getElementById("start").disabled);
}

async function layout(page) {
  assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), "Horizontal overflow");
  const overflow = await page.locator("button,select,h1,h2,.metrics dd").evaluateAll(elements => elements.filter(element => element.scrollWidth > element.clientWidth + 1).map(element => element.id || element.tagName));
  assert.deepEqual(overflow, [], "Text is clipped");
  assert(await page.locator(".icon img,.brand img").evaluateAll(images => images.every(image => image.complete && image.naturalWidth > 0)), "Missing icons");
}

(async () => {
  await fs.mkdir(output, { recursive: true });
  const browser = await chromium.launch({ channel: "msedge", headless: true });
  const errors = [], results = [];
  try {
    for (const viewport of [{ width: 1280, height: 900 }, { width: 390, height: 844 }]) {
      const context = await browser.newContext({ viewport, acceptDownloads: true });
      const page = await context.newPage();
      page.on("pageerror", error => errors.push(error.message));
      await page.goto(`${base}/jobs`);
      await newSheet(page);
      await page.locator("#text").fill("네");
      await page.locator("#character-mm").fill("28");
      const accepted = page.waitForResponse(response => response.url() === `${base}/api/jobs` && response.request().method() === "POST");
      await page.locator("#start").click();
      const job = await (await accepted).json();
      await waitState(page, "completed", job.job_id);
      assert.equal(await page.locator("#verification").textContent(), "가상 경로 통과");
      assert((await page.locator("#jobs").textContent()).includes("2026-10-12"), "Missing absolute record date");
      await page.locator("#trace").waitFor({ state: "visible" });
      await page.waitForFunction(() => document.getElementById("trace").complete && document.getElementById("trace").naturalWidth === 1188 && document.getElementById("trace").naturalHeight === 840);
      const black = await page.locator("#trace").evaluate(image => {
        const canvas = document.createElement("canvas"); canvas.width = image.naturalWidth; canvas.height = image.naturalHeight;
        const ctx = canvas.getContext("2d"); ctx.drawImage(image, 0, 0);
        const data = ctx.getImageData(0, 0, canvas.width, canvas.height).data; let pixels = 0;
        for (let i = 0; i < data.length; i += 4) if (data[i] < 80 && data[i + 1] < 80 && data[i + 2] < 80) pixels++;
        return pixels;
      });
      assert(black > 100, "Actual TCP trace is blank");
      const downloadEvent = page.waitForEvent("download"); await page.locator("#download").click();
      const report = JSON.parse(await fs.readFile(await (await downloadEvent).path(), "utf8"));
      assert.equal(report.status, "passed"); assert.equal(report.motion_authorized, false); assert.equal(report.pressure_validated, false);
      await page.reload(); await waitState(page, "completed");
      assert((await page.locator("#jobs").textContent()).includes("네"), "Reload lost stored history");
      assert.equal(await page.locator("#review-status").inputValue(), "", "Human review was automatically marked passed");
      await layout(page);
      await page.screenshot({ path: path.join(output, `${viewport.width}-completed.png`), fullPage: true });
      results.push({ viewport, normal_job_id: job.job_id, black_trace_pixels: black, passed: true });
      await context.close();
    }

    const context = await browser.newContext({ viewport: { width: 1280, height: 900 } });
    const page = await context.newPage(), second = await context.newPage();
    for (const tab of [page, second]) { tab.on("pageerror", error => errors.push(error.message)); await tab.goto(`${base}/jobs`); }
    await newSheet(page);
    await page.locator("#text").fill("안녕하세요."); await page.locator("#character-mm").fill("18");
    const accepted = page.waitForResponse(response => response.url() === `${base}/api/jobs` && response.request().method() === "POST");
    await page.locator("#start").click(); const job = await (await accepted).json();
    await waitState(page, "executing", job.job_id);
    await page.waitForFunction(() => document.getElementById("phase").textContent === "필기");
    await second.waitForFunction(() => document.getElementById("start").disabled);
    const duplicate = await context.request.post(`${base}/api/jobs`, { data: { text: "네" } });
    assert.equal(duplicate.status(), 409, "Cross-tab duplicate was accepted");
    await page.locator("#stop").click(); await waitState(page, "stopped", job.job_id);
    const result = await (await context.request.get(`${base}/api/jobs/${job.job_id}`)).json();
    assert(result.stop_requested && result.stop_settled && result.stop_test_passed, "Stop was not settled");
    assert(result.completed_strokes < result.total_strokes, "Partial stop claimed every stroke completed");
    await page.screenshot({ path: path.join(output, "1280-stopped.png"), fullPage: true });
    await second.waitForFunction(() => !document.getElementById("start").disabled);
    const before = (await (await context.request.get(`${base}/api/jobs`)).json()).jobs.length;
    await page.locator("#text").fill("abc"); await page.locator("#start").click();
    await page.waitForFunction(() => document.getElementById("error").textContent.length > 0);
    assert.equal((await (await context.request.get(`${base}/api/jobs`)).json()).jobs.length, before, "Invalid text created a job");
    await context.close();
    assert.deepEqual(errors, [], "Browser errors");
    await fs.writeFile(path.join(output, "summary.json"), JSON.stringify({ date: "2026-10-12", normal: results, stop: result, cross_tab_conflict_status: duplicate.status(), browser_errors: errors }, null, 2));
    console.log("PASS desktop/mobile: actual trace pixels, history/reload, report download, absolute dates, icons/layout; two tabs: exclusive execution, settled stop, invalid text rejection");
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
