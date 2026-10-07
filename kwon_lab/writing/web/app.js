"use strict";

const ui = Object.fromEntries(["call-form", "call-input", "send", "history", "error", "status", "paper", "paper-size", "stroke-progress", "elapsed", "progress", "stop", "replay", "download", "response-text"].map(id => [id, document.getElementById(id)]));
const ctx = ui.paper.getContext("2d");
const pixelsPerMm = 4;
let paper = { width_mm: 210, height_mm: 297, margin_mm: 8 };
const labels = { idle: "대기", planning: "계획 생성 중", writing: "쓰는 중", completed: "완료", stopped: "중지됨", error: "실패" };
let state = "idle", currentJob = null, lastJob = null, frame = 0, generation = 0;
let controller = null, startedAt = 0, currentEntry = null;

function setState(next) {
  state = next;
  ui.status.textContent = labels[next];
  ui.status.dataset.state = next;
  const busy = next === "planning" || next === "writing";
  ui.send.disabled = busy;
  ui["call-input"].disabled = busy;
  ui.stop.disabled = !busy;
  ui.replay.disabled = busy || !lastJob;
  ui.download.disabled = busy || !lastJob;
  ui.paper.setAttribute("aria-busy", String(busy));
}

function clearPaper(settings = paper) {
  paper = settings;
  ui.paper.width = Math.round(paper.width_mm * pixelsPerMm);
  ui.paper.height = Math.round(paper.height_mm * pixelsPerMm);
  ui["paper-size"].textContent = `${paper.width_mm} × ${paper.height_mm} mm · A4`;
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, ui.paper.width, ui.paper.height);
  ctx.strokeStyle = "#e4e9e6";
  ctx.lineWidth = 1;
  const margin = paper.margin_mm * pixelsPerMm;
  ctx.strokeRect(margin, margin, ui.paper.width - margin * 2, ui.paper.height - margin * 2);
  ctx.lineWidth = .6 * pixelsPerMm;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.strokeStyle = "#25282a";
}

function addEntry(input) {
  document.getElementById("empty-history")?.remove();
  const entry = document.createElement("article");
  entry.className = "entry";
  const top = document.createElement("div");
  top.className = "entry-top";
  const label = document.createElement("span");
  label.textContent = "호출";
  const time = document.createElement("time");
  time.textContent = new Date().toLocaleTimeString("ko-KR", { timeZone: "Asia/Seoul", hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false });
  top.append(label, time);
  const question = document.createElement("p");
  question.className = "entry-input";
  question.textContent = input;
  const answer = document.createElement("p");
  answer.className = "entry-response";
  const result = document.createElement("div");
  result.className = "entry-result";
  result.textContent = "계획 생성 중";
  entry.append(top, question, answer, result);
  ui.history.append(entry);
  while (ui.history.children.length > 30) ui.history.firstElementChild.remove();
  ui.history.scrollTop = ui.history.scrollHeight;
  return { answer, result };
}

function finish(next, message) {
  cancelAnimationFrame(frame);
  if (currentJob) {
    currentJob.preview_result = { status: next, elapsed_seconds: Math.round((performance.now() - startedAt) / 100) / 10, finished_at: new Date().toISOString() };
    lastJob = currentJob;
  }
  currentEntry.result.textContent = message || labels[next];
  currentJob = null;
  setState(next);
  ui["call-input"].focus();
}

function animate(job, token) {
  currentJob = job;
  currentEntry.answer.textContent = job.response;
  currentEntry.result.textContent = labels.writing;
  ui["response-text"].textContent = job.response;
  clearPaper(job.plan.paper);
  setState("writing");
  const strokes = job.plan.strokes;
  let index = 0, segment = 1, offset = 0, pauseUntil = 0;
  let previous = performance.now();
  const distances = strokes.map(s => s.points_mm.slice(1).reduce((total, point, i) => total + Math.hypot(point[0] - s.points_mm[i][0], point[1] - s.points_mm[i][1]), 0));
  const total = distances.reduce((a, b) => a + b, 0);
  let drawn = 0;
  function tick(now) {
    if (token !== generation || state !== "writing") return;
    let budget = Math.min(now - previous, 80) / 1000 * 36;
    previous = now;
    ui.elapsed.textContent = `${((now - startedAt) / 1000).toFixed(1)}초`;
    if (now < pauseUntil) { frame = requestAnimationFrame(tick); return; }
    while (budget > 0 && index < strokes.length) {
      const points = strokes[index].points_mm;
      if (segment >= points.length) {
        index++; segment = 1; offset = 0;
        ui["stroke-progress"].textContent = `${index} / ${strokes.length} 획`;
        pauseUntil = now + 140;
        break;
      }
      const a = points[segment - 1], b = points[segment];
      const length = Math.hypot(b[0] - a[0], b[1] - a[1]);
      if (length < .00001) { segment++; offset = 0; continue; }
      const step = Math.min(budget, length - offset);
      const start = offset / length, end = (offset + step) / length;
      ctx.beginPath();
      ctx.moveTo((a[0] + (b[0] - a[0]) * start) * pixelsPerMm, (a[1] + (b[1] - a[1]) * start) * pixelsPerMm);
      ctx.lineTo((a[0] + (b[0] - a[0]) * end) * pixelsPerMm, (a[1] + (b[1] - a[1]) * end) * pixelsPerMm);
      ctx.stroke();
      offset += step; drawn += step; budget -= step;
      if (offset >= length - .00001) { segment++; offset = 0; }
    }
    ui.progress.value = total ? Math.min(drawn / total, 1) : 1;
    if (index >= strokes.length) {
      ui.progress.value = 1;
      finish("completed", `완료 · ${((now - startedAt) / 1000).toFixed(1)}초`);
      return;
    }
    frame = requestAnimationFrame(tick);
  }
  ui["stroke-progress"].textContent = `0 / ${strokes.length} 획`;
  frame = requestAnimationFrame(tick);
}

async function submit(input) {
  if (state === "planning" || state === "writing") return;
  const token = ++generation;
  ui.error.textContent = "";
  ui.progress.value = 0;
  ui.elapsed.textContent = "0.0초";
  ui["stroke-progress"].textContent = "0 / 0 획";
  ui["response-text"].textContent = "";
  clearPaper();
  currentJob = null;
  currentEntry = addEntry(input);
  startedAt = performance.now();
  setState("planning");
  const requestController = new AbortController();
  controller = requestController;
  const timeout = setTimeout(() => requestController.abort("timeout"), 6000);
  try {
    const response = await fetch("/api/call", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ input }), signal: requestController.signal });
    const job = await response.json();
    if (token !== generation) return;
    if (!response.ok) throw new Error(job.error || "계획을 생성하지 못했습니다.");
    if (job.plan.motion_authorized !== false || job.plan.physical_calibration_required !== true) throw new Error("미리보기 안전 조건이 올바르지 않습니다.");
    animate(job, token);
  } catch (error) {
    if (token !== generation) return;
    ui.error.textContent = error.name === "AbortError" || error === "timeout" ? "요청 시간이 초과됐습니다." : (error instanceof TypeError ? "로컬 서버에 연결하지 못했습니다." : String(error.message || error));
    finish("error", ui.error.textContent);
  } finally { clearTimeout(timeout); if (token === generation) controller = null; }
}

ui["call-form"].addEventListener("submit", event => { event.preventDefault(); submit(ui["call-input"].value); });
ui.stop.addEventListener("click", () => {
  if (state !== "planning" && state !== "writing") return;
  generation++;
  controller?.abort(); controller = null;
  finish("stopped");
});
ui.replay.addEventListener("click", () => { if (lastJob) submit(lastJob.input); });
ui.download.addEventListener("click", () => {
  if (!lastJob) return;
  const url = URL.createObjectURL(new Blob([JSON.stringify(lastJob, null, 2)], { type: "application/json" }));
  const link = document.createElement("a");
  link.href = url; link.download = `writing-${lastJob.job_id}.json`; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
clearPaper();
setState("idle");
