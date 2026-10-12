"use strict";

const ids = ["job-form", "text", "character-mm", "start", "jobs", "empty", "count", "error", "state", "job-text", "stop", "download", "strokes", "simulation-time", "wall-time", "verification", "trace", "trace-empty", "phase", "failure", "artifacts", "review-form", "review-status", "readback", "save-review", "review-at", "sheet-state", "sheet-number", "work-area", "remaining", "placement-status", "page-map", "new-page", "position", "reservation-state"];
const ui = Object.fromEntries(ids.map(id => [id, document.getElementById(id)]));
const states = { planning: "계획", checking: "사전검사", executing: "가상 구동", stopping: "중지 처리", verifying: "결과 검사", completed: "가상 완료", stopped: "중지", failed: "실패", uncertain: "불확실", interrupted: "실행 중단" };
const phases = { settling: "초기 유지", approach: "접근", writing: "필기", departure: "이탈", return: "복귀", final_hold: "마지막 유지", stopping: "중지 유지" };
const activeStates = new Set(["planning", "checking", "executing", "stopping", "verifying"]);
const sheetStates = { ready: "준비", awaiting_new_sheet: "새 종이 대기", inspection_required: "확인·교체 대기", confirmation_required: "새 종이 확인 필요" };
const reservationStates = { reserved: "예약", occupied: "사용 완료", partial: "부분 사용 보존", uncertain: "확인 필요", released_no_motion: "미구동·예약 해제" };
let selected = null, jobs = [], active = null, polling = false, sending = false, renderedReview = null, renderedRows = "";
let sheet = null, storageFault = false, placement = null, previewToken = 0, previewTimer = 0, pageKey = "";
let mutation = 0;

function date(value) {
  if (!value) return "-";
  return new Intl.DateTimeFormat("sv-SE", { timeZone: "Asia/Seoul", year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false }).format(new Date(value));
}

async function request(path, payload) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 15000);
  try {
    const response = await fetch(path, payload === undefined ? { signal: controller.signal } : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload), signal: controller.signal });
    const result = await response.json();
    if (!response.ok) { const error = new Error(result.error || "요청 실패"); error.reason = result.reason; throw error; }
    return result;
  } finally { clearTimeout(timeout); }
}

function artifact(job, name) { return `/api/jobs/${job.job_id}/artifacts/${name}`; }
function seconds(value) { return value === null || value === undefined ? "-" : `${value.toFixed(3)}초`; }

function drawPage() {
  if (!sheet) return;
  const canvas = ui["page-map"], scale = 4;
  canvas.width = Math.round(sheet.paper.width_mm * scale);
  canvas.height = Math.round(sheet.paper.height_mm * scale);
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "white"; ctx.fillRect(0, 0, canvas.width, canvas.height);
  const rect = (box, fill, stroke) => {
    const [x0, y0, x1, y1] = box.map(value => value * scale);
    if (fill) { ctx.fillStyle = fill; ctx.fillRect(x0, y0, x1 - x0, y1 - y0); }
    ctx.strokeStyle = stroke; ctx.lineWidth = 2; ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
  };
  rect(sheet.working_rect_mm, null, "#59936d");
  for (const reservation of sheet.reservations) {
    if (reservation.state === "released_no_motion") continue;
    const colors = { reserved: "#dce8dc", occupied: "#b5d9c1", partial: "#f2d196", uncertain: "#eeb5b4" };
    rect(reservation.reserved_rect_mm, colors[reservation.state], "#8faaa0");
  }
  if (placement?.page_id === sheet.page_id) {
    ctx.setLineDash([8, 6]); rect(placement.reserved_rect_mm, null, "#286899"); ctx.setLineDash([]);
  }
}

async function preview() {
  const token = ++previewToken; placement = null; drawPage();
  if (!sheet || sheet.state !== "ready") { ui["placement-status"].textContent = "새 종이 확인 필요"; return; }
  ui["placement-status"].textContent = "배치 계산";
  try {
    const result = await request("/api/page/preview", { text: ui.text.value, character_mm: Number(ui["character-mm"].value), page_id: sheet.page_id });
    if (token !== previewToken) return;
    placement = result.placement;
    ui["placement-status"].textContent = `y ${placement.layout_rect_mm[1].toFixed(1)}mm · ${placement.lines}줄 · 경로 미검사`;
    drawPage();
  } catch (error) {
    if (token !== previewToken) return;
    ui["placement-status"].textContent = ["insufficient_remaining_space", "text_does_not_fit_blank_sheet"].includes(error.reason) ? "종이 공간 부족" : "배치 불가";
  }
}

function render() {
  ui.start.disabled = sending || Boolean(active) || storageFault || !sheet || sheet.state !== "ready";
  ui.text.disabled = ui["character-mm"].disabled = sending || Boolean(active);
  ui.count.textContent = `${jobs.length}건`;
  ui.empty.hidden = jobs.length > 0;
  ui["new-page"].disabled = sending || Boolean(active) || storageFault || !sheet;
  if (sheet) {
    ui["sheet-state"].textContent = sheetStates[sheet.state] || sheet.state;
    ui["sheet-number"].textContent = `${sheet.number} · ${sheet.page_id.slice(0, 8)}`;
    const [x0, y0, x1, y1] = sheet.working_rect_mm;
    ui["work-area"].textContent = `A4 ${sheet.paper.width_mm} × ${sheet.paper.height_mm}mm · 배치 ${(x1 - x0).toFixed(1)} × ${(y1 - y0).toFixed(1)}mm`;
    ui.remaining.textContent = `${sheet.remaining_height_mm.toFixed(1)} mm`;
    drawPage();
  }
  const rowsKey = JSON.stringify([selected, jobs.map(job => [job.job_id, job.state])]);
  if (rowsKey !== renderedRows) {
    renderedRows = rowsKey;
    const rows = jobs.map(job => {
    const row = document.createElement("tr");
    row.classList.toggle("selected", job.job_id === selected);
    const time = document.createElement("td"); time.textContent = date(job.created_at);
    const text = document.createElement("td");
    const choose = document.createElement("button"); choose.className = "choose"; choose.textContent = job.text; choose.title = `작업 ${job.job_id} 선택`; choose.addEventListener("click", () => { selected = job.job_id; render(); }); text.append(choose);
    const status = document.createElement("td"); status.textContent = states[job.state] || job.state;
    row.append(time, text, status); return row;
    });
    ui.jobs.replaceChildren(...rows);
  }
  const job = jobs.find(item => item.job_id === selected);
  if (!job) return;
  ui.state.textContent = states[job.state] || job.state; ui.state.dataset.state = job.state;
  ui.state.dataset.jobId = job.job_id;
  ui["job-text"].textContent = job.text;
  ui.strokes.textContent = `${job.completed_strokes} / ${job.total_strokes}`;
  ui["simulation-time"].textContent = seconds(job.simulation_time_s ?? job.progress.simulation_time_s);
  ui["wall-time"].textContent = seconds(job.wall_elapsed_s);
  ui.position.textContent = job.placement ? `종이 ${job.placement.page_number} · ${job.paper.character_mm}mm · y ${job.placement.layout_rect_mm[1].toFixed(1)}mm` : "이전 위치 기록 없음";
  ui["reservation-state"].textContent = reservationStates[job.page_reservation_state] || "이전 위치 기록 없음";
  ui.verification.textContent = job.verification === "passed_virtual_path" ? "가상 경로 통과" : (states[job.verification] || { pending: "검사 대기" }[job.verification] || job.verification);
  ui.phase.textContent = phases[job.progress.phase] || "-";
  ui.stop.disabled = job.job_id !== active || !["planning", "checking", "executing"].includes(job.state);
  ui["save-review"].disabled = activeStates.has(job.state);
  const failureLabels = { robot_reach: "로봇 도달 불가: 종이에는 들어가지만 현재 로봇 설정으로는 해당 경로를 쓸 수 없습니다.", robot_validation: "로봇 검증 실패: 종이 공간 부족과 별개의 경로·안전 검사 문제입니다.", execution_record: "실행 또는 기록 오류" };
  ui.failure.textContent = job.failure ? `${failureLabels[job.failure_category] || "실행 실패"} ${JSON.stringify(job.failure)}` : "";
  const hasTrace = job.artifacts.includes("trace.png");
  ui.trace.hidden = !hasTrace; ui["trace-empty"].hidden = hasTrace;
  if (hasTrace && ui.trace.getAttribute("src") !== artifact(job, "trace.png")) ui.trace.src = artifact(job, "trace.png");
  const filename = job.artifacts.includes("report.json") ? "report.json" : "job.json";
  ui.download.href = artifact(job, filename); ui.download.download = `${job.job_id}-${filename}`; ui.download.classList.remove("disabled");
  ui.artifacts.replaceChildren(...job.artifacts.map(name => { const a = document.createElement("a"); a.href = artifact(job, name); a.download = `${job.job_id}-${name}`; a.textContent = name; return a; }));
  const reviewKey = `${job.job_id}:${job.human_review.recorded_at || ""}`;
  if (reviewKey !== renderedReview) {
    renderedReview = reviewKey; ui["review-status"].value = job.human_review.status === "unreviewed" ? "" : job.human_review.status; ui.readback.value = job.human_review.readback || "";
  }
  ui["review-at"].textContent = job.human_review.recorded_at ? `검수 기록 · ${date(job.human_review.recorded_at)} KST` : "미검수";
}

async function refresh() {
  if (polling) return;
  polling = true;
  const version = mutation;
  try {
    const result = await request("/api/jobs");
    if (version !== mutation) return;
    jobs = result.jobs; active = result.active_job_id; sheet = result.page; storageFault = result.storage_fault;
    if (!selected) selected = active || jobs[0]?.job_id;
    if (result.storage_fault) ui.error.textContent = "기록 저장 오류 · 추가 실행 차단";
    else if (sheet.reason === "paper_layout_or_simulation_profile_changed") ui.error.textContent = "종이 배치 또는 가상 로봇 설정이 변경됐습니다. 새 종이 준비를 확인하세요. 기존 기록은 보존됩니다.";
    render();
    const nextKey = JSON.stringify([sheet.page_id, sheet.next_row_y_mm, sheet.state]);
    if (pageKey !== nextKey) { pageKey = nextKey; preview(); }
  } catch (error) { ui.error.textContent = error.message || "서버 연결 실패"; ui.start.disabled = true; ui.stop.disabled = true; }
  finally { polling = false; }
}

ui["job-form"].addEventListener("submit", async event => {
  event.preventDefault(); if (sending || active || !sheet || sheet.state !== "ready") return;
  sending = true; mutation++; ui.error.textContent = ""; render();
  try {
    const job = await request("/api/jobs", { text: ui.text.value, character_mm: Number(ui["character-mm"].value), page_id: sheet.page_id });
    mutation++; selected = job.job_id; active = job.job_id;
    jobs = [job, ...jobs.filter(item => item.job_id !== job.job_id)]; render();
    if (matchMedia("(max-width:800px)").matches) ui.state.scrollIntoView({ block: "start", behavior: "smooth" });
  }
  catch (error) { ui.error.textContent = error.message; }
  finally { sending = false; await refresh(); }
});
ui.stop.addEventListener("click", async () => {
  if (!selected) return; ui.stop.disabled = true; ui.error.textContent = "";
  try { await request(`/api/jobs/${selected}/stop`, {}); mutation++; }
  catch (error) { ui.error.textContent = error.message; }
  await refresh();
});
ui["review-form"].addEventListener("submit", async event => {
  event.preventDefault(); if (!selected) return; ui.error.textContent = "";
  try { await request(`/api/jobs/${selected}/review`, { status: ui["review-status"].value, readback: ui.readback.value }); }
  catch (error) { ui.error.textContent = error.message; }
  await refresh();
});
ui.trace.addEventListener("error", () => { ui.trace.hidden = true; ui["trace-empty"].hidden = false; ui["trace-empty"].textContent = "결과 이미지 로드 실패"; });
for (const input of [ui.text, ui["character-mm"]]) input.addEventListener("input", () => { clearTimeout(previewTimer); previewToken++; previewTimer = setTimeout(preview, 250); });
ui["new-page"].addEventListener("click", async () => {
  if (!sheet || active || sending) return;
  const pageId = sheet.page_id;
  if (!confirm("새 종이를 같은 위치에 준비했나요? 현재 종이의 사용 기록은 보존하고 새 종이로 전환합니다.")) return;
  sending = true; ui.error.textContent = ""; render();
  try { await request("/api/page/reset", { page_id: pageId, confirmed: true }); mutation++; }
  catch (error) { ui.error.textContent = error.message; }
  finally { sending = false; await refresh(); }
});
refresh();
setInterval(refresh, 500);
