"""Gemma 숨바꼭질 평가 — 손목캠 단독으로 4단계 사다리를 오르는지 측정.

  1단계  보임/안보임 판정 정확도 (스캔 자세, CV 심판과 대조)
  2단계  pan 스윕 스캔으로 블록 발견
  3단계  발견 물체 인식 (빨간 블록 vs 파란 공 구분)
  4단계  시각 센터링(pan·wrist_flex, 3x3→5x5 2단 격자 + 적응 스텝)
         → 카메라 중심선 레이캐스트 → pick.  한 번이라도 잡으면 합격.

심판(정답)은 렌더 픽셀 색 분할(CV) — 채점 전용, 제어 판단은 전부 학생.
--dry: CV 심판이 학생 대역 (제어 루프 자체 검증, LLM 호출 없음)

실행:  .venv/bin/python kwon_lab/tools/gemma_seek.py [--dry] [모델=gemma4:e4b] [시드=7000]
출력:  콘솔 단계별 성적 + outputs/gemma_seek/ (전 프레임 + report.json)
"""

import base64
import json
import sys
import time
import urllib.request
from io import BytesIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import mujoco
import numpy as np
from PIL import Image

from envs.so101_pick_env import SO101PickEnv
from skills.primitives import pick

OUT = Path("outputs/gemma_seek")
SCAN_POSE = dict(lift=-0.6, elbow=0.3, wf=1.2)  # pan 스윕으로 스폰존 커버 (흰바닥 씬, 4/4시드)
PANS = np.linspace(-1.2, 1.2, 7)
BLOCK_Z = 0.03   # pick 목표 높이 (블록 중심)
AIM_Z = 0.055    # 레이캐스트 평면: 시각은 '보이는 면(윗면)'을 조준하므로 윗면 높이로 교차
                 # (중심 높이 3cm로 연장하면 광선이 블록 너머로 3~4cm 오버슈트)

PROMPTS = {
    "vis": ('로봇 손목 카메라 사진이다. 빨간 블록(상자)과 파란 공이 보이는지 판단하라. '
            '반드시 JSON만: {"red_block_visible": true/false, "blue_ball_visible": true/false}'),
    "loc3": ('로봇 손목 카메라 사진이다. 화면을 3x3 격자로 나눌 때 빨간 블록 중심이 속한 칸은? '
             '반드시 JSON만: {"col": "left"|"center"|"right", "row": "top"|"middle"|"bottom"}. '
             '블록이 안 보이면 col과 row에 "none".'),
    "loc5": ('로봇 손목 카메라 사진이다. 화면을 가로 5칸, 세로 5칸으로 나눌 때 빨간 블록 중심이 '
             '속한 칸은? 반드시 JSON만: '
             '{"col": "far_left"|"left"|"center"|"right"|"far_right", '
             '"row": "far_top"|"top"|"middle"|"bottom"|"far_bottom"}. 안 보이면 "none".'),
}
COL3 = {"left": +1, "center": 0, "right": -1}          # 값 = pan 보정 방향
ROW3 = {"top": -1, "middle": 0, "bottom": +1}          # 값 = wf 보정 방향
COL5 = {"far_left": +2, "left": +1, "center": 0, "right": -1, "far_right": -2}
ROW5 = {"far_top": -2, "top": -1, "middle": 0, "bottom": +1, "far_bottom": +2}


# ── 환경 조작 ──────────────────────────────────────────────
def move_joints(env, q, secs=0.6):
    start = env.data.ctrl[:6].copy()
    n = max(1, int(secs * 25))
    for i in range(n):
        env.step(start + (i + 1) / n * (np.asarray(q) - start))


def wrist_frame(env):
    env.renderer.update_scene(env.data, camera="wrist")
    return env.renderer.render()


def cam_pose(env):
    cid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist")
    return env.data.cam_xpos[cid].copy(), env.data.cam_xmat[cid].reshape(3, 3).copy()


# ── 심판 (CV — 채점·드라이런 전용) ─────────────────────────
def referee(frame):
    r, g, b = (frame[:, :, i].astype(int) for i in range(3))
    # 비율 기반: 블록 빨강(r/g≈6) 통과, 골대 림 주황(r/g≈3)·음영 차단.
    # 공 파랑(b/g≈2.3) 통과, 하늘(b/g≈1.6) 차단.
    red = (r > 100) & (r > 4 * g) & (r > 4 * b)
    blue = (b > 130) & (b > 2 * g) & (r < 80)
    out = {"red_block_visible": int(red.sum()) > 150,
           "blue_ball_visible": int(blue.sum()) > 150}
    if out["red_block_visible"]:
        ys, xs = np.nonzero(red)
        cx, cy = xs.mean() / 640, ys.mean() / 480  # 0~1
        out["col"] = ["left", "center", "right"][min(2, int(cx * 3))]
        out["row"] = ["top", "middle", "bottom"][min(2, int(cy * 3))]
        out["col5"] = ["far_left", "left", "center", "right", "far_right"][min(4, int(cx * 5))]
        out["row5"] = ["far_top", "top", "middle", "bottom", "far_bottom"][min(4, int(cy * 5))]
    return out


# ── 학생 ───────────────────────────────────────────────────
class Student:
    def __init__(self, model, dry):
        self.model, self.dry, self.calls, self.total_sec = model, dry, 0, 0.0

    def ask(self, frame, kind):
        self.calls += 1
        if self.dry:
            v = referee(frame)
            if kind == "vis":
                return v
            if not v["red_block_visible"]:
                return {"col": "none", "row": "none"}
            if kind == "loc3":
                return {"col": v["col"], "row": v["row"]}
            return {"col": v["col5"], "row": v["row5"]}
        buf = BytesIO()
        Image.fromarray(frame).save(buf, "JPEG", quality=90)
        req = urllib.request.Request(
            "http://localhost:11434/api/chat",
            data=json.dumps({
                "model": self.model, "stream": False, "format": "json", "think": False,
                "messages": [{"role": "user", "content": PROMPTS[kind],
                              "images": [base64.b64encode(buf.getvalue()).decode()]}],
                "options": {"temperature": 0},
            }).encode(),
            headers={"Content-Type": "application/json"})
        t0 = time.time()
        body = json.load(urllib.request.urlopen(req, timeout=180))
        self.total_sec += time.time() - t0
        return json.loads(body["message"]["content"])


def save(frame, name):
    OUT.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame).save(OUT / f"{name}.jpg")


# ── 단계 구현 ──────────────────────────────────────────────
def scan(env, student, tag, score=None):
    """pan 스윕 — 학생이 '보임'이라 한 pan 반환 (심판 대조 채점은 score dict에)."""
    for k, pan in enumerate(PANS):
        move_joints(env, [pan, SCAN_POSE["lift"], SCAN_POSE["elbow"], SCAN_POSE["wf"], 0, 1.0])
        frame = wrist_frame(env)
        save(frame, f"{tag}_scan{k}_pan{pan:+.1f}")
        ans = student.ask(frame, "vis")
        if score is not None:
            truth = referee(frame)
            score["total"] += 1
            score["ok"] += bool(ans.get("red_block_visible")) == truth["red_block_visible"]
            score["ball"].append(bool(ans.get("blue_ball_visible")) == truth["blue_ball_visible"])
            print(f"  pan {pan:+.1f}: 학생 {'보임' if ans.get('red_block_visible') else '안보임'} "
                  f"/ 심판 {'보임' if truth['red_block_visible'] else '안보임'}")
        if ans.get("red_block_visible"):
            return pan
    return None


def center(env, student, pan, wf, grid, max_iter, tag):
    """적응 스텝 센터링. 반환: (수렴여부, pan, wf, 시야상실여부)."""
    cmap, rmap = (COL3, ROW3) if grid == "loc3" else (COL5, ROW5)
    sp, sw = (0.14, 0.10) if grid == "loc3" else (0.05, 0.04)
    last_c = last_r = 0
    for it in range(max_iter):
        frame = wrist_frame(env)
        save(frame, f"{tag}_{grid}_{it}")
        ans = student.ask(frame, grid)
        c, r = cmap.get(ans.get("col"), None), rmap.get(ans.get("row"), None)
        print(f"    [{grid}] {it}: col={ans.get('col')} row={ans.get('row')} "
              f"(pan {pan:+.2f} wf {wf:+.2f})")
        if c is None or r is None:
            return False, pan, wf, True  # 시야 상실
        if c == 0 and r == 0:
            return True, pan, wf, False
        # 적응 스텝: 방향 뒤집힘 → 반감(이진 수렴), 같은 방향 지속 → 증폭
        if c * last_c < 0:
            sp = max(0.02, sp * 0.5)
        elif c == last_c and c != 0:
            sp = min(0.3, sp * 1.4)
        if r * last_r < 0:
            sw = max(0.02, sw * 0.5)
        elif r == last_r and r != 0:
            sw = min(0.25, sw * 1.4)
        pan = float(np.clip(pan + np.sign(c) * sp, -1.9, 1.9))
        wf = float(np.clip(wf + np.sign(r) * sw, -1.6, 1.6))
        last_c, last_r = c, r
        move_joints(env, [pan, SCAN_POSE["lift"], SCAN_POSE["elbow"], wf, 0, 1.0], secs=0.4)
    return False, pan, wf, False


def bisect_center(env, student, pan, wf, tag):
    """경계 이분법: 5x5 답이 뒤집히는 pan/wf 경계 2개의 중점 = 광축이 블록 중심.
    격자 칸 너비가 아니라 탐색 스텝(0.03rad)이 정밀도를 결정 (±0.5cm급)."""
    def probe(p, w):
        move_joints(env, [p, SCAN_POSE["lift"], SCAN_POSE["elbow"], w, 0, 1.0], secs=0.3)
        ans = student.ask(wrist_frame(env), "loc5")
        return COL5.get(ans.get("col")), ROW5.get(ans.get("row"))

    def edges(axis):  # axis 0=pan(col), 1=wf(row)
        lo = hi = None
        for sgn in (+1, -1):
            p, w = pan, wf
            for k in range(1, 7):
                if axis == 0:
                    p = pan + sgn * 0.03 * k
                else:
                    w = wf + sgn * 0.03 * k
                c, r = probe(p, w)
                v = c if axis == 0 else r
                if v is None:
                    break
                if v != 0:  # center 칸을 벗어남 → 경계 발견
                    if sgn > 0:
                        hi = (p if axis == 0 else w)
                    else:
                        lo = (p if axis == 0 else w)
                    break
        return lo, hi

    lo, hi = edges(0)
    if lo is not None and hi is not None:
        pan = (lo + hi) / 2
    lo, hi = edges(1)
    if lo is not None and hi is not None:
        wf = (lo + hi) / 2
    move_joints(env, [pan, SCAN_POSE["lift"], SCAN_POSE["elbow"], wf, 0, 1.0], secs=0.3)
    print(f"    [이분법] 수렴: pan {pan:+.3f} wf {wf:+.3f}")
    return pan, wf


def main():
    args = [a for a in sys.argv[1:] if a != "--dry"]
    dry = "--dry" in sys.argv
    model = (args[0] if args else "gemma4:e4b")
    seed = int(args[1]) if len(args) > 1 else 7000

    env = SO101PickEnv()
    obs, info = env.reset(seed=seed)
    student = Student(model, dry)
    report = {"model": "referee(dry)" if dry else model, "seed": seed}
    vis_score = {"ok": 0, "total": 0, "ball": []}
    t_start = time.time()
    print(f"[숨바꼭질] {'심판 대역(dry)' if dry else model} | 시드 {seed} "
          f"| 블록 정답 ({info['block_pos'][0]:.3f}, {info['block_pos'][1]:.3f})")

    grasped = False
    for attempt in range(4):
        pan = scan(env, student, f"a{attempt}", score=vis_score if attempt == 0 else None)
        if pan is None:
            print("  스캔 실패 — 블록 미발견")
            break
        if attempt == 0:
            report["stage2_found_at_pan"] = float(pan)
        wf = SCAN_POSE["wf"]
        ok3, pan, wf, lost = center(env, student, pan, wf, "loc3", 15, f"a{attempt}")
        if lost:
            continue  # 재스캔
        ok5, pan, wf, lost = center(env, student, pan, wf, "loc5", 10, f"a{attempt}")
        if lost:
            continue
        pan, wf = bisect_center(env, student, pan, wf, f"a{attempt}")
        c, R = cam_pose(env)
        d = -R[:, 2]
        if d[2] > -1e-3:
            continue
        t = (AIM_Z - c[2]) / d[2]
        target = c + t * d
        err = np.linalg.norm(target[:2] - env._get_info()["block_pos"][:2]) * 100
        print(f"  레이캐스트 → ({target[0]:.3f}, {target[1]:.3f}) [오차 {err:.1f}cm] "
              f"(3x3수렴 {ok3}, 5x5수렴 {ok5})")
        grasped, _ = pick(env, [target[0], target[1], BLOCK_Z])
        print(f"  pick 시도 {attempt + 1}: {'성공 ✅' if grasped else '실패 ❌'}")
        if grasped:
            break

    report["stage1_visibility"] = f"{vis_score['ok']}/{vis_score['total']}"
    report["stage3_ball_agree"] = (f"{sum(vis_score['ball'])}/{len(vis_score['ball'])}"
                                   if vis_score["ball"] else "-")
    report["stage4_grasp"] = bool(grasped)
    report["student_calls"] = student.calls
    if not dry and student.calls:
        report["sec_per_call"] = round(student.total_sec / student.calls, 2)
    report["wall_sec"] = round(time.time() - t_start, 1)
    print("\n" + "=" * 60)
    print(f"1단계 보임판정 {report['stage1_visibility']} | "
          f"2단계 발견 {'pan %.1f' % report['stage2_found_at_pan'] if 'stage2_found_at_pan' in report else '실패'} | "
          f"3단계 공 판정 {report['stage3_ball_agree']} | 4단계 그랩 {'✅' if grasped else '❌'}")
    print(f"학생 호출 {student.calls}회"
          + (f", 호출당 {report.get('sec_per_call', 0)}s" if not dry else "")
          + f" | 전체 {report['wall_sec']}s")
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
