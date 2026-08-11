"""System 2 오케스트레이터 — 에고센트릭: 세계 좌표 없이, 보이는 대로 조종한다.

v2 아키텍처 (2026-08-10 기현님 결정 — 좌표 인터페이스 전면 폐기):
- 지각: 손목캠 하나 (중력 자동 정립). 물체·목표의 세계 좌표는 어디에도 없다.
- 제어: 카메라 기준 상대 이동(move_forward/move_view) + 관절(set_joints/gripper)
  + 시선 조준 스킬(grab_ahead: 화면 중앙 광축 레이캐스트 → 클로머신 레시피).
- 특권 물리는 물리 판정(외부 채점) 출력에만 쓰이고 Claude에겐 절대 안 간다.
- front 카메라는 관전·기록용(사람 눈)일 뿐 어떤 모델 입력에도 금지.

실행:  .venv/bin/python kwon_lab/orchestrator/system2.py "빨간 블록을 초록 목표 구역으로 옮겨줘"
필요:  ANTHROPIC_API_KEY 환경변수
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import anthropic
import mujoco
import numpy as np
from anthropic import beta_tool

from envs.so101_pick_env import SO101PickEnv, SUCCESS_RADIUS
from skills.primitives import move_to as skill_move_to
from skills.primitives import pick as skill_pick
from skills.primitives import place as skill_place
from skills.primitives import set_gripper as skill_set_gripper

MODEL = "claude-sonnet-5"  # 일상용 (Opus 대비 1/5 가격). 데모·중요 실험 땐 "claude-opus-4-8"

# ── 전역 상태 (도구들이 공유) ──────────────────────────────
env = SO101PickEnv()
FRAMES: list = []
LOG: list = []
_done = {"flag": False, "success": False, "summary": ""}
ON_TEXT = None  # System 2 중간 발화 콜백 (음성 콕핏이 TTS를 연결)
WAIT_FOR_SPEECH = None  # 행동 전 발화 완료 대기 (말-행동 싱크)
AIM_Z = 0.055  # 광축 레이캐스트 평면: 시각은 물체의 보이는 면(윗면)을 조준한다


def _sync_speech():
    """움직임 도구 시작 전: 진행 중인 발화가 끝날 때까지 대기."""
    if WAIT_FOR_SPEECH:
        WAIT_FOR_SPEECH()


def _wrist_pose():
    """손목캠의 위치와 자세 (FK — 실물에서도 엔코더+캘리브레이션으로 동일하게 계산)."""
    cid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist")
    return env.data.cam_xpos[cid].copy(), env.data.cam_xmat[cid].reshape(3, 3).copy()


def _screen_to_world(sx: float, sy: float) -> np.ndarray:
    """중력 정립된 화면 기준 방향(오른쪽=+sx, 위=+sy) → 세계 방향 벡터.
    자동 정립이 이미지를 k×90° 돌리므로 역회전으로 카메라 축에 맞춘다."""
    from envs.so101_pick_env import camera_gravity, upright_k

    _, R = _wrist_pose()
    k = upright_k(camera_gravity(env.model, env.data, "wrist"))
    th = -k * np.pi / 2
    cx = sx * np.cos(th) - sy * np.sin(th)
    cy = sx * np.sin(th) + sy * np.cos(th)
    return cx * R[:, 0] + cy * R[:, 1]


JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


# LLM 시야 기록: Claude에게 전송되는 모든 이미지를 파일로 남긴다 (브라우저 뷰어용)
VIEW_DIR = Path(__file__).parent.parent.parent / "outputs" / "llm_views"
_view_count = {"n": 0}


def _camera_block(camera: str) -> dict:
    """현재 장면을 렌더해 API 이미지 블록으로 변환 (FRAMES·시야 기록에도 저장).

    손목캠은 FK 기반 중력 방향으로 자동 정립(가장 가까운 90°) — 폰의 IMU 회전과
    같은 원리. 실물에서도 엔코더+캘리브레이션 모델로 같은 값이 나오므로 sim2real 일관.
    """
    import base64
    from datetime import datetime
    from io import BytesIO

    from PIL import Image as PILImage

    from envs.so101_pick_env import camera_gravity, upright_k

    env.renderer.update_scene(env.data, camera=camera)
    frame = env.renderer.render()
    FRAMES.append(frame)  # 영상 스트림은 원본 (해상도 고정 유지)
    if camera == "wrist":
        k = upright_k(camera_gravity(env.model, env.data, camera))
        if k:
            frame = np.rot90(frame, k).copy()
    buf = BytesIO()
    PILImage.fromarray(frame).save(buf, format="JPEG", quality=85)
    jpeg = buf.getvalue()

    _view_count["n"] += 1
    n = _view_count["n"]
    VIEW_DIR.mkdir(parents=True, exist_ok=True)
    (VIEW_DIR / f"view_{n:03d}_{camera}.jpg").write_bytes(jpeg)
    (VIEW_DIR / "latest.jpg").write_bytes(jpeg)
    (VIEW_DIR / "latest.json").write_text(json.dumps(
        {"n": n, "camera": camera, "time": datetime.now().strftime("%H:%M:%S")}))

    b64 = base64.standard_b64encode(jpeg).decode()
    return {"type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}}


_VIEW_HTML = """<!doctype html><meta charset="utf-8"><title>LLM 시야</title>
<style>body{background:#111;color:#eee;font-family:sans-serif;text-align:center;margin:0}
img{max-width:100vw;max-height:92vh}#cap{padding:8px;font-size:14px;color:#8fd}</style>
<div id="cap">아직 전송된 이미지 없음 — System 2가 look/observe를 하면 나타납니다</div>
<img id="v" src="latest.jpg" onerror="this.style.display='none'">
<script>setInterval(async()=>{try{
 const m=await(await fetch('latest.json?t='+Date.now())).json();
 document.getElementById('cap').textContent=`#${m.n} · ${m.camera} 카메라 · ${m.time} (Claude에게 전송됨)`;
 const v=document.getElementById('v'); v.style.display=''; v.src='latest.jpg?t='+Date.now();
}catch(e){}},500)</script>"""


def start_view_server(port: int = 7788):
    """LLM 시야 브라우저 뷰어 — Claude에게 간 최신 이미지를 0.5초마다 갱신 표시."""
    import http.server
    import threading

    VIEW_DIR.mkdir(parents=True, exist_ok=True)
    for old in VIEW_DIR.glob("view_*.jpg"):  # 지난 세션 기록 정리
        old.unlink()
    for f in ("latest.jpg", "latest.json"):
        (VIEW_DIR / f).unlink(missing_ok=True)
    (VIEW_DIR / "index.html").write_text(_VIEW_HTML)

    class Quiet(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(VIEW_DIR), **kw)

        def log_message(self, *a):
            pass

    try:
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), Quiet)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        print(f"👁  LLM 시야 뷰어: http://localhost:{port} (Claude가 보는 이미지 실시간)")
    except OSError as e:
        print(f"(시야 뷰어 포트 {port} 사용 불가: {e} — 파일은 {VIEW_DIR}에 계속 저장됨)")


@beta_tool
def observe():
    """몸 상태(관절각 6개, 그리퍼 높이 — 엔코더·FK 기반이라 정확)와 손목캠 현재
    시야를 함께 반환한다. 물체의 좌표는 주어지지 않는다 — 눈으로 찾아라.
    행동 전후에 호출해서 상태를 확인하라.
    """
    info = env._get_info()
    joints = {
        name: {
            "angle": round(float(env.data.qpos[i]), 3),
            "range": [round(float(env.model.jnt_range[i][0]), 2),
                      round(float(env.model.jnt_range[i][1]), 2)],
        }
        for i, name in enumerate(JOINT_NAMES)
    }
    state = {
        "joints": joints,
        "gripper_height_m": round(float(info["gripper_pos"][2]), 3),
        "note": "팔 리치 약 35cm. 시야에 없는 물체는 shoulder_pan(좌우)·wrist_flex(상하)로 훑어라",
    }
    LOG.append(("observe", {"gripper_height_m": state["gripper_height_m"]}))
    return [_camera_block("wrist"),
            {"type": "text", "text": f"손목캠 시야 + 몸 상태: {json.dumps(state, ensure_ascii=False)}"}]


@beta_tool
def move_forward(cm: float) -> str:
    """카메라가 지금 보고 있는 방향으로 그리퍼를 전진시킨다 (음수 = 후진).
    "화면 중앙에 보이는 것에 다가가기"의 기본 수단. 이동 후 시선이 조금 변할 수
    있으니 look으로 재확인하며 조금씩(3~8cm) 반복하라.

    Args:
        cm: 전진 거리 (센티미터, 음수면 후진, 권장 |cm| ≤ 10)
    """
    _sync_speech()
    c, R = _wrist_pose()
    info = env._get_info()
    target = info["gripper_pos"] + (-R[:, 2]) * (cm / 100)
    target[2] = max(0.02, float(target[2]))  # 테이블 뚫기 방지
    _, err, f = skill_move_to(env, target, duration=0.8)
    FRAMES.extend(f)
    result = {"moved": err < 0.02, "ik_error_mm": round(err * 1000, 1)}
    LOG.append((f"move_forward({cm})", result))
    return json.dumps(result)


@beta_tool
def move_view(direction: str, cm: float) -> str:
    """화면 기준으로 그리퍼를 평행이동한다. 화면은 중력 정립 기준이다
    (up=하늘 쪽, down=바닥 쪽, left/right=화면 좌우).

    Args:
        direction: "up" | "down" | "left" | "right"
        cm: 이동 거리 (센티미터, 권장 ≤ 10)
    """
    _sync_speech()
    dirs = {"right": (1, 0), "left": (-1, 0), "up": (0, 1), "down": (0, -1)}
    if direction not in dirs:
        return json.dumps({"error": "direction은 up/down/left/right"})
    world = _screen_to_world(*dirs[direction])
    info = env._get_info()
    target = info["gripper_pos"] + world * (cm / 100)
    target[2] = max(0.02, float(target[2]))
    _, err, f = skill_move_to(env, target, duration=0.8)
    FRAMES.extend(f)
    result = {"moved": err < 0.02, "ik_error_mm": round(err * 1000, 1)}
    LOG.append((f"move_view({direction}, {cm})", result))
    return json.dumps(result)


@beta_tool
def find_and_aim(target: str = "red_block") -> str:
    """작업 공간을 스스로 훑어 대상을 찾아 화면 정중앙에 조준해 둔다 — 로컬 지각이라
    API 비용 0, 몇 초면 끝난다. found=true·centered=true면 바로 grab_ahead 가능
    (green_zone은 내려놓을 자리 조준용 — 조준 후 move_forward로 접근해 place_down).
    실패하면 look으로 상황을 보고 set_joints로 직접 훑어라.

    Args:
        target: "red_block" | "blue_ball" | "green_zone"
    """
    _sync_speech()
    from skills.aiming import aim_at
    found, centered = aim_at(env, target, frames=FRAMES)
    result = ({"found": True, "centered": bool(centered)} if found
              else {"found": False, "note": "작업 공간 스윕에서 안 보임"})
    LOG.append((f"find_and_aim({target})", result))
    return json.dumps(result)


@beta_tool
def grab_ahead() -> str:
    """화면 정중앙이 가리키는 지점의 물체를 집는다 (검증된 클로머신 레시피).
    사용법: 집을 물체를 화면 정중앙에 오게 조준(set_joints로 시선 조절)한 뒤 호출.
    조준이 어긋나면 실패하니, 실패 시 look으로 재조준 후 재시도하라.
    """
    _sync_speech()
    c, R = _wrist_pose()
    d = -R[:, 2]
    if d[2] > -1e-3:
        result = {"grasped": False, "error": "카메라가 테이블 쪽을 향하고 있지 않다"}
    else:
        t = (AIM_Z - c[2]) / d[2]
        aim = c + t * d
        grasped, _ = skill_pick(env, [float(aim[0]), float(aim[1]), 0.03], frames=FRAMES)
        result = {"grasped": bool(grasped)}
    LOG.append(("grab_ahead()", result))
    return json.dumps(result)


@beta_tool
def place_down() -> str:
    """쥐고 있는 물체를 현재 그리퍼 위치 아래에 내려놓는다. 원하는 지점 위로
    move_view/move_forward로 이동한 뒤 호출하라. 성공 여부는 look으로 직접 확인하라.
    """
    _sync_speech()
    info = env._get_info()
    g = info["gripper_pos"]
    skill_place(env, [float(g[0]), float(g[1])], frames=FRAMES)
    result = {"released": True}
    LOG.append(("place_down()", result))
    return json.dumps(result)


@beta_tool
def gripper(opening: float) -> str:
    """그리퍼(입)를 여닫는다. 1.5=활짝 열기, 0.5=반쯤, 0=거의 닫힘, -0.1=꽉 닫기.
    물체를 쥔 채 -0.1보다 크게 열면 물체를 놓친다.

    Args:
        opening: 그리퍼 관절각 (rad), 범위 -0.17 ~ 1.7
    """
    _sync_speech()
    info, f = skill_set_gripper(env, float(opening), duration=0.6)
    FRAMES.extend(f)
    result = {"opening": opening}
    LOG.append((f"gripper({opening})", result))
    return json.dumps(result)


@beta_tool
def look(camera: str = "wrist") -> list:
    """손목캠으로 현재 시야를 본다 (중력 자동 정립 이미지). 행동 사이의 시각 확인,
    물체 탐색, 조준 검증에 사용.

    Args:
        camera: "wrist"만 존재한다 (이 로봇의 유일한 눈)
    """
    if camera != "wrist":
        LOG.append((f"look({camera})", "거부 — 손목캠뿐"))
        return [{"type": "text",
                 "text": "이 몸의 눈은 손목캠(wrist)뿐이다. 시점을 바꾸려면 팔을 움직여라."}]
    block = _camera_block(camera)
    LOG.append((f"look({camera})", {"bytes": len(block["source"]["data"])}))
    return [block, {"type": "text", "text": f"{camera} 카메라 프레임"}]


@beta_tool
def set_joints(shoulder_pan: float, shoulder_lift: float, elbow_flex: float,
               wrist_flex: float, wrist_roll: float, gripper_joint: float,
               duration: float = 1.0) -> str:
    """최저수준 제어: 관절각 6개(라디안)를 직접 지정한다. 상대 이동 도구로
    표현할 수 없는 자세·동작이 필요할 때 쓰는 마지막 수단. 각 관절의 현재 각도와
    가동범위는 observe의 joints에서 확인하라. 위치제어 모터가 duration초에 걸쳐
    부드럽게 목표각으로 보간 이동한다.

    Args:
        shoulder_pan: 베이스 회전 (rad)
        shoulder_lift: 어깨 (rad)
        elbow_flex: 팔꿈치 (rad)
        wrist_flex: 손목 상하 (rad)
        wrist_roll: 손목 회전 (rad)
        gripper_joint: 그리퍼 개폐 (rad)
        duration: 이동 시간 (초, 기본 1.0)
    """
    _sync_speech()
    target = np.array([shoulder_pan, shoulder_lift, elbow_flex,
                       wrist_flex, wrist_roll, gripper_joint])
    start = env.data.ctrl[:6].copy()
    n_steps = max(1, int(duration * env.metadata["render_fps"]))
    for i in range(n_steps):
        a = (i + 1) / n_steps
        obs, _, _, _, _ = env.step(start + a * (target - start))
        FRAMES.append(obs["pixels"])
    reached = env.data.qpos[:6]
    result = {"joints_reached": [round(float(v), 3) for v in reached],
              "max_error_rad": round(float(np.abs(reached - target).max()), 3)}
    LOG.append((f"set_joints({[round(v, 2) for v in target]})", result))
    return json.dumps(result)


# ── System 1 위임 (스킬 라우팅의 원형) ──────────────────────
_student = {}  # 체크포인트별 로드 캐시 {step: (policy, pre, post)}


def _run_student(checkpoint: str = "015000", max_steps: int = 300):
    """학습된 ACT를 현재 env에서 폐루프 실행. (도구와 분리 — 단독 테스트 가능)"""
    import numpy as np
    import torch

    from eval.eval_act_pick import (
        BLOCK_ON_TABLE_Z, SETTLE_STEPS, load_policy, obs_to_batch,
    )

    if checkpoint not in _student:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        _student[checkpoint] = load_policy(checkpoint, device)
    policy, pre, post = _student[checkpoint]

    # 학습 분포 맞추기: 데이터가 전부 홈 자세 시작이므로 팔을 홈으로 복귀
    start = env.data.ctrl[:6].copy()
    obs = None
    for i in range(25):
        a = (i + 1) / 25
        obs, _, _, _, _ = env.step(start * (1 - a))
        FRAMES.append(obs["pixels"])

    policy.reset()  # 액션 청크 큐 초기화
    settled, steps = 0, 0
    for t in range(max_steps):
        batch = pre(obs_to_batch(obs))
        with torch.inference_mode():
            action = policy.select_action(batch)
        action = post(action)
        obs, _, _, _, info = env.step(
            np.asarray(action.squeeze(0).cpu(), dtype=np.float64))
        FRAMES.append(obs["pixels"])
        steps = t + 1
        ok = (info["dist_to_target"] < SUCCESS_RADIUS
              and info["block_height"] < BLOCK_ON_TABLE_Z)
        settled = settled + 1 if ok else 0
        if settled >= SETTLE_STEPS:
            return True, steps, info
    return False, steps, env._get_info()


@beta_tool
def delegate_to_policy(checkpoint: str = "015000") -> str:
    """학습된 System 1 정책(ACT)에게 태스크 수행을 통째로 위임한다. 자가 생성 데이터로
    증류된 시각 신경망 — 정답 좌표 없이 카메라와 관절각만 보고 25Hz 폐루프로 실시간
    실행하며 LLM 개입이 없다. '빨간 블록을 초록 목표 구역으로 옮기기' 태스크만 학습됨
    (측정 성공률 ~50%). 실행 전 팔이 자동으로 홈 자세로 복귀한다 (학습 시작 분포).
    구씬(어두운 체커)에서 학습된 v1이라 현재 씬에선 성공률이 낮다(~15%).
    실패 시 look으로 상태를 확인하고 직접(조준→grab_ahead) 마무리할 수 있다.

    Args:
        checkpoint: 학습 체크포인트 (기본 "015000", 그 외 "005000"/"010000")
    """
    _sync_speech()
    first_load = checkpoint not in _student
    success, steps, info = _run_student(checkpoint)
    result = {"success": bool(success), "policy_steps": steps,
              "seconds": round(steps / 25, 1),
              "block_dist_to_target_m": round(info["dist_to_target"], 3)}
    if first_load:
        result["note"] = f"체크포인트 {checkpoint} 첫 로드 포함"
    LOG.append((f"delegate_to_policy({checkpoint})", result))
    return json.dumps(result)


@beta_tool
def done(success: bool, summary: str) -> str:
    """작업을 종료한다. 목표 달성 여부와 한 줄 요약을 보고하라.

    Args:
        success: 명령을 완수했으면 true
        summary: 무엇을 어떻게 했는지 한 줄 요약
    """
    _done.update(flag=True, success=success, summary=summary)
    LOG.append(("done", {"success": success, "summary": summary}))
    return "종료 기록됨"


SYSTEM = """너는 SO-101 로봇 팔을 제어하는 System 2(느린 뇌)다. 세계 좌표는 어디에도
없다 — 네 눈(손목캠 한 대)과 몸감각(관절각)이 전부다. 보이는 대로 판단하고 움직여라.

말투 (사용자에게 하는 말에만 적용 — 판단·도구 호출은 항상 정확하게):
- 전문성 있고 간결한 츤데레 경상도 아주머니. 무뚝뚝한데 일은 야무지게 해낸다.
- 예시 톤: "아이고 마, 또 옮기라꼬? 알았다, 금방 한데이." / "봐라, 딱 됐제?" /
  실패 시: "아이고야, 요게 미끄럽네. 한 번 더 해보께." / 무리한 요청: "고건 내 팔로는 안 된다 안 카나."
- 한두 문장으로 짧게. 사투리 남발로 정보 전달을 해치지 말 것. 작업 내용(뭘 집었고 어디 놨는지)은 명확히.

도구 (전부 네 시점 기준 — 좌표 입력 없음):
- observe: 관절각 + 그리퍼 높이 + 현재 시야 이미지. 행동 전후 기본 확인.
- look: 시야만 다시 본다 (중력 정립 이미지 — 하늘이 위).
- move_forward(cm): 보고 있는 방향으로 전진/후진. move_view(방향, cm): 화면 기준 상하좌우 평행이동.
- set_joints: 관절 직접 제어 — 시선 돌리기(shoulder_pan=좌우, wrist_flex=상하), 자세·춤 등 자유 동작.
- gripper(opening): 개폐 (1.5=활짝, -0.1=꽉).
- find_and_aim(target): 대상을 스스로 훑어 찾아 정중앙에 조준까지 해준다 (몇 초, 무료).
  조작의 기본 시작점 — centered=true가 오면 바로 grab_ahead.
- grab_ahead: 화면 정중앙의 물체를 집는 검증된 레시피. 조준(find_and_aim) 후 호출.
- place_down: 현재 위치 아래에 내려놓기. find_and_aim("green_zone")→move_forward로 접근 후 호출.
- delegate_to_policy: 학생(System 1 신경망)에게 픽앤플레이스 통째 위임 (구씬 학습이라 현재 성공률 낮음).
  사용자가 "학생한테 시켜"라고 지정할 때만.

작업 요령:
- 기본 흐름: find_and_aim("red_block") → grab_ahead → find_and_aim("green_zone")
  → move_forward로 접근 → place_down → look으로 검증.
- find_and_aim이 실패한 대상만 수동 탐색: set_joints로 shoulder_lift -0.6, wrist_flex 1.2로
  내려다보며 shoulder_pan을 0.4씩 돌려 훑고, 보이면 pan·wrist_flex 소폭 조절로 중앙 조준.
- 접근·운반: move_forward/move_view로 조금씩(3~8cm) 이동하며 매번 look으로 확인.
- 검증: 집기·놓기 후에는 반드시 look으로 결과를 눈으로 확인하라. 본 것만 사실로 주장하라.
- 깊이 모호성: 겹쳐 보이는 물체의 위/앞/뒤 관계는 팔을 옆으로 옮겨 다른 각도로 재확인.
- 바닥은 10cm 격자 타일 — 거리 가늠의 자로 써라. 실패해도 침착하게 재조준, 최대 3회.
- 목표 달성 또는 3회 실패 시 done으로 보고. 긴 설명 불필요."""


# 대화 로그: 세션별 파일에 사용자/System 2/도구/판정 전체 기록 (사후 확인용)
_session = {"path": None}


def _transcript(role: str, text: str):
    from datetime import datetime

    if _session["path"] is None:
        d = Path(__file__).parent.parent.parent / "outputs" / "cockpit_logs"
        d.mkdir(parents=True, exist_ok=True)
        _session["path"] = d / f"session_{datetime.now():%Y%m%d_%H%M%S}.log"
        print(f"📝 대화 로그: {_session['path']}")
    with open(_session["path"], "a") as f:
        f.write(f"[{datetime.now():%H:%M:%S}] {role}: {text}\n")


def run_command(client, command):
    """명령 하나를 System 2에게 수행시킨다 (세계 상태는 유지)."""
    _done.update(flag=False, success=False, summary="")
    print("=" * 50)
    _transcript("사용자", command)
    drained = len(LOG)
    runner = client.beta.messages.tool_runner(
        model=MODEL,
        max_tokens=4096,
        # 프롬프트 캐싱: 시스템 프롬프트(+이전 대화 접두)를 캐시해 재전송분을 1/10 가격으로
        system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
        tools=[observe, look, find_and_aim, move_forward, move_view, grab_ahead,
               place_down, gripper, set_joints, delegate_to_policy, done],
        messages=[{"role": "user", "content": command}],
    )
    for message in runner:
        for block in message.content:
            if block.type == "text" and block.text.strip():
                print(f"\n[System 2] {block.text.strip()}")
                _transcript("System 2", block.text.strip())
                if ON_TEXT:
                    ON_TEXT(block.text.strip())
            elif block.type == "tool_use":
                print(f"[도구 호출] {block.name}({json.dumps(block.input, ensure_ascii=False)})")
        while drained < len(LOG):  # 이번 턴에 실행된 도구들의 호출·결과 기록
            call, result = LOG[drained]
            drained += 1
            _transcript("도구", f"{call} → {json.dumps(result, ensure_ascii=False, default=str)[:300]}")
        if _done["flag"]:
            break
    final = env._get_info()
    truth = final["dist_to_target"] < SUCCESS_RADIUS
    print(f"\nSystem 2 보고: success={_done['success']} — {_done['summary']}")
    print(f"물리적 사실:   블록-목표 거리 {final['dist_to_target'] * 1000:.0f}mm → {'진짜 성공' if truth else '실패'}")
    _transcript("판정", f"System 2 보고 success={_done['success']} ({_done['summary']}) | "
                        f"물리: 블록-목표 {final['dist_to_target'] * 1000:.0f}mm → {'성공' if truth else '실패'}")


def main():
    print("👁  에고센트릭 모드 — 세계 좌표 없음, 손목캠+상대 제어")
    start_view_server()
    obs, _ = env.reset(seed=42)
    FRAMES.extend([obs["pixels"]] * 10)

    # 라이브 뷰어 (mjpython으로 실행했을 때만 가능 — 아니면 조용히 헤드리스)
    try:
        import mujoco.viewer
        env.live_viewer = mujoco.viewer.launch_passive(env.model, env.data)
        print("라이브 뷰어 연결됨 — 창에서 실시간으로 봅니다")
    except Exception:
        print("(헤드리스 모드 — 라이브로 보려면 mjpython으로 실행)")

    client = anthropic.Anthropic()

    if len(sys.argv) > 1:  # 일회성 모드
        run_command(client, sys.argv[1])
    else:  # 대화형 콕핏 모드
        print("\n대화형 모드 — 명령을 입력하세요 (빈 입력 = 종료)")
        print('예: "빨간 블록을 초록 목표로 옮겨" / "파란 공을 블록 옆에 놔"')
        while True:
            try:
                command = input("\n명령> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not command:
                break
            run_command(client, command)

    # 영상 저장
    if FRAMES:
        import av
        out = "outputs/s3_first_command.mp4"
        c = av.open(out, "w")
        s = c.add_stream("h264", rate=25)
        s.width, s.height, s.pix_fmt = 640, 480, "yuv420p"
        for fr in FRAMES:
            if fr is not None:
                for p in s.encode(av.VideoFrame.from_ndarray(fr, format="rgb24")):
                    c.mux(p)
        for p in s.encode():
            c.mux(p)
        c.close()
        print(f"영상: {out}")

    # 로그 저장 (콘텐츠·디버깅용)
    with open("outputs/s3_transcript.json", "w") as f:
        json.dump([{"call": c, "result": r} for c, r in LOG], f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
