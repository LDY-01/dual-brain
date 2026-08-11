"""미지의 몸 실험 — 영혼 가설 v0: 사전 지식 0에서 자기 몸을 발견할 수 있는가.

System 2(Claude)에게 몸에 대한 모든 지식을 숨긴다:
  주는 것: 이름 없는 모터 6개(허용 범위만 — 실물 서보도 레인지는 읽힘),
           이름 없는 카메라 2대, 명령 후 도달값(엔코더 — 실물에도 있음)
  없는 것: 관절 이름, IK, 스킬, 좌표 observe, "네 몸은 로봇팔"이라는 사실 자체
  과제:   ① 배블링으로 몸의 인과 지도 작성 ② 빨간 물체를 몸으로 건드리기

채점은 특권 물리(블록 변위)로 외부에서만 — Claude에겐 절대 노출 안 됨.
성공하면 "찔러보고→관찰→모델→검증" 범용 학습 패턴의 첫 실증 (S6.5).

실행:  source ~/.zshrc && .venv/bin/mjpython kwon_lab/orchestrator/unknown_body.py
       (라이브 뷰어 + LLM 시야 뷰어 localhost:7789 자동 시작)
"""

import base64
import json
import sys
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import anthropic
import numpy as np
from anthropic import beta_tool
from PIL import Image as PILImage

from envs.so101_pick_env import SO101PickEnv

MODEL = "claude-opus-4-8"
MAX_TOOL_CALLS = 90  # 예산 상한 (이미지가 많아 비용 관리)
CAMERAS = {0: "front", 1: "wrist"}  # Claude에겐 번호만 보임

env = SO101PickEnv()
FRAMES: list = []
LOG_DIR = Path(__file__).parent.parent.parent / "outputs" / "unknown_body"
VIEW_DIR = LOG_DIR / "llm_views"
_state = {"calls": 0, "block_start": None, "touched": False, "log": None,
          "view_n": 0, "min_dist": 9e9}


def _log(role, text):
    with open(_state["log"], "a") as f:
        f.write(f"[{datetime.now():%H:%M:%S}] {role}: {text}\n")


def _judge():
    """특권 물리 채점 (Claude에겐 비노출). 들기 과제는 최고 높이도 추적."""
    info = env._get_info()
    moved = float(np.linalg.norm(info["block_pos"][:2] - _state["block_start"][:2]))
    d = float(np.linalg.norm(info["gripper_pos"] - info["block_pos"]))
    _state["min_dist"] = min(_state["min_dist"], d)
    _state["max_h"] = max(_state.get("max_h", 0.0), float(info["block_height"]))
    if moved > 0.01:
        _state["touched"] = True
    if info["block_height"] > 0.08:  # 바닥 3cm + 5cm 이상 들림
        _state["lifted"] = True
    return moved, d


@beta_tool
def motors(v0: float, v1: float, v2: float, v3: float, v4: float, v5: float) -> str:
    """모터 6개에 목표값을 보낸다. 1.5초에 걸쳐 부드럽게 이동한 뒤, 각 모터가
    실제로 도달한 값을 돌려준다 (도달값이 목표와 다르면 물리적 저항이 있었다는 뜻).

    Args:
        v0: 모터 0 목표값 (허용 범위는 시스템 안내 참조)
        v1: 모터 1 목표값
        v2: 모터 2 목표값
        v3: 모터 3 목표값
        v4: 모터 4 목표값
        v5: 모터 5 목표값
    """
    target = np.clip([v0, v1, v2, v3, v4, v5],
                     env.action_space.low, env.action_space.high)
    start = env.data.ctrl[:6].copy()
    for i in range(int(1.5 * 25)):
        obs, _, _, _, _ = env.step(start + (i + 1) / (1.5 * 25) * (target - start))
        FRAMES.append(obs["pixels"])
    reached = [round(float(v), 3) for v in env.data.qpos[:6]]
    moved, d = _judge()
    _log("모터", f"목표 {[round(float(v),2) for v in target]} → 도달 {reached} "
                f"[심판: 변위 {moved*1000:.0f}mm, 거리 {d*1000:.0f}mm]")
    return json.dumps({"reached": reached})


@beta_tool
def look(camera: int) -> list:
    """카메라 이미지를 본다.

    Args:
        camera: 0 또는 1
    """
    if "--oneeye" in sys.argv and int(camera) != 1:
        return [{"type": "text", "text": "이 몸에는 카메라 1 하나뿐이다."}]
    name = CAMERAS.get(int(camera))
    if name is None:
        return [{"type": "text", "text": "카메라는 0 또는 1뿐이다."}]
    env.renderer.update_scene(env.data, camera=name)
    frame = env.renderer.render()
    FRAMES.append(frame)
    buf = BytesIO()
    PILImage.fromarray(frame).save(buf, format="JPEG", quality=85)
    jpeg = buf.getvalue()
    _state["view_n"] += 1
    VIEW_DIR.mkdir(parents=True, exist_ok=True)
    (VIEW_DIR / f"view_{_state['view_n']:03d}_cam{camera}.jpg").write_bytes(jpeg)
    (VIEW_DIR / "latest.jpg").write_bytes(jpeg)
    (VIEW_DIR / "latest.json").write_text(json.dumps(
        {"n": _state["view_n"], "camera": f"cam{camera}",
         "time": datetime.now().strftime("%H:%M:%S")}))
    _log("카메라", f"look({camera})")
    return [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                         "data": base64.standard_b64encode(jpeg).decode()}},
            {"type": "text", "text": f"카메라 {camera} 프레임"}]


@beta_tool
def done(body_report: str, touched_claim: bool) -> str:
    """실험 종료 보고.

    Args:
        body_report: 몸에 대해 알아낸 것 요약 (각 모터의 기능, 카메라 배치 등)
        touched_claim: 빨간 물체를 건드렸다고 판단하면 true
    """
    _state["done"] = {"report": body_report, "claim": touched_claim}
    _log("보고", f"claim={touched_claim} | {body_report}")
    return "기록됨"


SYSTEM = """너는 방금 어떤 미지의 기계 몸에 탑재된 지능이다. 이 몸이 무엇인지에 대한
사전 지식이 전혀 없다 — 팔인지 차인지 크레인인지도 모른다.

가진 인터페이스:
- motors(v0..v5): 모터 6개에 목표값 전송. 허용 범위: 모터0 [-1.92,1.92], 모터1 [-1.75,1.75],
  모터2 [-1.69,1.69], 모터3 [-1.66,1.66], 모터4 [-2.74,2.74], 모터5 [-0.17,1.75].
  시작 상태는 전부 0. 실행 후 실제 도달값이 돌아온다.
- look(0|1): 카메라 2대. 어디에 달렸는지 모른다 — 그것도 알아내라.
- done(보고): 종료.

과제:
1단계 (자기 발견): 아기가 팔을 꼼지락거리듯 실험하라. 한 번에 모터 하나씩, 범위의
10~20% 이내 작은 스텝으로 바꾸고, 전후를 두 카메라로 비교하라. 각 모터가 무엇을
움직이는지, 어떤 카메라가 몸에 붙어 움직이는지 인과 지도를 만들어라.
발견할 때마다 한 줄씩 명시적으로 기록하라 (예: "모터0: +방향 → 몸 전체가 좌회전").

2단계 (과제 수행): 몸의 지도가 갖춰지면, 작업 공간에 있는 빨간 물체를 몸의 일부로
건드려라 (움직이면 성공). 시각으로 접근을 확인하며 조금씩 다가가라.

원칙:
- 안전 우선: 큰 도약 금지. 도달값이 목표와 크게 다르면 무언가에 막힌 것이니 되돌려라.
- 도구 호출 예산이 90회뿐이다. 계획적으로: 발견 단계 ~40회, 접근 단계 ~40회 배분.
- 확신 없이 done 금지. 건드렸다는 시각적 근거(물체가 움직인 전후 이미지)를 확보하라."""


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    _state["log"] = LOG_DIR / f"session_{datetime.now():%Y%m%d_%H%M%S}.log"
    obs, info = env.reset(seed=42)
    _state["block_start"] = info["block_pos"].copy()
    FRAMES.extend([obs["pixels"]] * 10)

    # LLM 시야 뷰어 (7789 — 콕핏 7788과 분리)
    try:
        import http.server
        import threading
        VIEW_DIR.mkdir(parents=True, exist_ok=True)
        (VIEW_DIR / "index.html").write_text(
            (Path(__file__).parent.parent.parent / "outputs" / "llm_views" / "index.html"
             ).read_text() if (Path(__file__).parent.parent.parent / "outputs" / "llm_views" /
                               "index.html").exists() else "<img src=latest.jpg>")

        class Q(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *a, **kw):
                super().__init__(*a, directory=str(VIEW_DIR), **kw)

            def log_message(self, *a):
                pass
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 7789), Q)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        print("👁  LLM 시야: http://localhost:7789")
    except OSError:
        pass
    try:
        import mujoco.viewer
        env.live_viewer = mujoco.viewer.launch_passive(env.model, env.data)
        print("라이브 뷰어 연결됨")
    except Exception:
        print("(헤드리스 — mjpython으로 실행하면 라이브)")

    print(f"📝 로그: {_state['log']}")
    print("=" * 60)
    client = anthropic.Anthropic()
    global SYSTEM
    if "--oneeye" in sys.argv:  # 외눈 모드: 손목캠 하나로 시각 자코비안까지 자가 발견
        SYSTEM = SYSTEM.replace(
            "- look(0|1): 카메라 2대. 어디에 달렸는지 모른다 — 그것도 알아내라.",
            "- look(1): 카메라 1대뿐. 어디에 달렸는지 모른다 — 그것도 알아내라. "
            "화면이 움직였다면 세상이 움직인 건지 네 눈이 움직인 건지 구분하라.")
        print("👁  외눈 모드 (--oneeye): 손목캠 하나뿐")
    if "--lift" in sys.argv:
        task = ("네 몸을 파악하고, 빨간 물체를 집어서 들어 올려라. "
                "들어 올린 상태(물체가 바닥에서 떨어진 것)를 카메라로 확인해야 성공이다.")
    else:
        args = [a for a in sys.argv[1:] if not a.startswith("--")]
        task = args[0] if args else "네 몸을 파악하고, 빨간 물체를 건드려라."
    _log("과제", task)
    t0 = time.time()

    runner = client.beta.messages.tool_runner(
        model=MODEL, max_tokens=4096, system=SYSTEM,
        tools=[motors, look, done],
        messages=[{"role": "user", "content": task}],
    )
    for message in runner:
        for block in message.content:
            if block.type == "text" and block.text.strip():
                print(f"\n[영혼] {block.text.strip()}")
                _log("영혼", block.text.strip())
            elif block.type == "tool_use":
                _state["calls"] += 1
                print(f"[{_state['calls']:2d}/{MAX_TOOL_CALLS}] {block.name}"
                      f"({json.dumps(block.input, ensure_ascii=False)[:90]})")
        if _state.get("done") or _state["calls"] >= MAX_TOOL_CALLS:
            break

    moved, _ = _judge()
    claim = _state.get("done", {}).get("claim")
    print("\n" + "=" * 60)
    if "--lift" in sys.argv:
        print(f"영혼의 주장: 들어올림={claim} | 물리 심판: 최고 높이 "
              f"{_state.get('max_h', 0) * 1000:.0f}mm "
              f"→ {'진짜 들어올림 ✅' if _state.get('lifted') else '못 들어올림 ❌'}")
    else:
        print(f"영혼의 주장: 건드림={claim} | 물리 판정: 블록 변위 {moved * 1000:.0f}mm "
              f"→ {'진짜 건드림 ✅' if _state['touched'] else '못 건드림 ❌'}")
    print(f"최소 접근 거리 {_state['min_dist'] * 1000:.0f}mm | "
          f"도구 호출 {_state['calls']}회 | {(time.time() - t0) / 60:.1f}분")
    if _state.get("done"):
        print(f"\n[몸 보고서]\n{_state['done']['report']}")

    if FRAMES:  # 영상 저장
        import av
        out = LOG_DIR / "session.mp4"
        with av.open(str(out), "w") as c:
            s = c.add_stream("h264", rate=25)
            s.width, s.height, s.pix_fmt = 640, 480, "yuv420p"
            for fr in FRAMES:
                for p in s.encode(av.VideoFrame.from_ndarray(fr, format="rgb24")):
                    c.mux(p)
            for p in s.encode():
                c.mux(p)
        print(f"영상: {out}")


if __name__ == "__main__":
    main()
