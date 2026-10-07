"use strict";

const { chromium } = require("playwright");
const assert = require("node:assert/strict");
const fs = require("node:fs/promises");
const path = require("node:path");

const url = process.env.WRITING_TEST_URL || "http://127.0.0.1:8765";
const output = path.resolve("outputs/writing/chat-ui-tests");
const wait = ms => new Promise(resolve => setTimeout(resolve, ms));
const state = (page, expected) => page.waitForFunction(value => document.getElementById("status").dataset.state === value, expected);
const inkPixels = page => page.locator("#paper").evaluate(canvas => {
  const data = canvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height).data;
  let count = 0;
  for (let i = 0; i < data.length; i += 4) if (data[i] < 80 && data[i + 1] < 80 && data[i + 2] < 80) count++;
  return count;
});

(async () => {
  await fs.mkdir(output, { recursive: true });
  const browser = await chromium.launch({ channel: "msedge", headless: true });
  const failures = [];
  try {
    for (const viewport of [{ width: 1280, height: 800 }, { width: 390, height: 844 }]) {
      const context = await browser.newContext({ viewport, acceptDownloads: true });
      const page = await context.newPage();
      page.on("pageerror", error => failures.push(error.message));
      let posts = 0;
      page.on("request", request => { if (request.method() === "POST") posts++; });
      await page.goto(url);
      await state(page, "idle");
      const ratio = await page.locator("#paper").evaluate(canvas => {
        const rect = canvas.getBoundingClientRect();
        return rect.width / rect.height;
      });
      assert(Math.abs(ratio - 210 / 297) < .001, "A4 portrait aspect ratio is incorrect");
      assert((await page.locator("#paper-size").textContent()).includes("210 × 297"));
      assert.equal(await inkPixels(page), 0);
      await page.locator("#send").click();
      await state(page, "writing");
      assert(await page.locator("#send").isDisabled());
      assert(await page.locator("#call-input").isDisabled());
      await page.locator("#call-form").evaluate(form => form.requestSubmit());
      await state(page, "completed");
      assert.equal(posts, 1, "Duplicate input was not blocked");
      assert.equal(await page.locator(".entry-response").last().textContent(), "네");
      assert(await inkPixels(page) > 200, "Writing canvas is blank");
      assert(await page.locator("#paper").evaluate(canvas => canvas.getAttribute("aria-busy") === "false"));
      assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
      assert(await page.locator("img").evaluateAll(images => images.every(image => image.complete && image.naturalWidth > 0)));
      await page.screenshot({ path: path.join(output, `${viewport.width}-completed.png`), fullPage: true });

      const downloaded = page.waitForEvent("download");
      await page.locator("#download").click();
      const job = JSON.parse(await fs.readFile(await (await downloaded).path(), "utf8"));
      assert.equal(job.response, "네");
      assert.equal(job.preview_result.status, "completed");
      assert.equal(job.plan.motion_authorized, false);
      assert.equal(job.plan.physical_calibration_required, true);
      assert.equal(job.plan.paper.width_mm, 210);
      assert.equal(job.plan.paper.height_mm, 297);

      await page.locator("#replay").click();
      await state(page, "writing");
      await page.locator("#stop").click();
      await state(page, "stopped");
      const stoppedPixels = await inkPixels(page);
      await wait(400);
      assert.equal(await inkPixels(page), stoppedPixels, "Stopped animation resumed");
      await page.locator("#replay").click();
      await state(page, "completed");

      await page.locator("#call-input").fill("사과가 무슨 색이야");
      await page.locator("#send").click();
      await state(page, "error");
      assert((await page.locator("#error").textContent()).includes("야"));
      assert.equal(await inkPixels(page), 0, "Unsupported input drew an answer");
      await page.locator("#call-input").fill("야");
      await page.route("**/api/call", route => route.abort());
      await page.locator("#send").click();
      await state(page, "error");
      assert((await page.locator("#error").textContent()).includes("연결"));
      await page.unroute("**/api/call");

      let attempts = 0;
      await page.route("**/api/call", async route => {
        if (++attempts === 1) { await wait(300); await route.abort().catch(() => {}); }
        else await route.continue();
      });
      await page.locator("#send").click();
      await state(page, "planning");
      await page.locator("#stop").click();
      await state(page, "stopped");
      await page.locator("#send").click();
      await state(page, "completed");
      assert.equal(await page.locator(".entry-response").last().textContent(), "네");
      assert(await inkPixels(page) > 200);
      await context.close();
      console.log(`PASS ${viewport.width}x${viewport.height}: completion, duplicate blocking, stop/retry, API failures, JSON download, canvas pixels, icons, layout`);
    }
    assert.deepEqual(failures, []);
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
