"""시선 조준 공용 모듈 — 손목캠 색 지각으로 대상을 화면 중앙에 놓는다.

콕핏(find_and_aim 도구)·데이터 수집(조준 후 시연 시작)·평가(조준 후 정책 인계)가
전부 이 하나를 쓴다. 픽셀만 사용(비특권) — 실물에선 HSV 임계값만 재튜닝하면 동일.

v2 데이터셋의 교훈이 담긴 모듈: 홈 자세 시작 학생은 5% (첫 관측에 블록이 없어
정보 비대칭으로 붕괴) → 조준 후 시작으로 가시성 100% 보장 + 관절 상태에도
방향 정보 탑재 (2026-08-11).
"""

import numpy as np

CENTER = (320, 240)
SCAN_POSE = (0.0, 0.0, 0.0, 0.0, 0.0)


def locate_color(frame, target: str):
    """색 비율 기반 위치 지각. 반환: (cx, cy) 픽셀 중심 또는 None."""
    r, g, b = (frame[:, :, i].astype(int) for i in range(3))
    masks = {
        "red_block": (r > 70) & (r > 2.5 * g) & (r > 2.5 * b),
        "blue_ball": (b > 130) & (b > 2 * g) & (r < 80),
        "green_zone": (g > 110) & (g > 2 * r) & (g > 2 * b),
    }
    mask = masks.get(target)
    if mask is None or int(mask.sum()) < 150:
        return None
    ys, xs = np.nonzero(mask)
    return float(xs.mean()), float(ys.mean())


def _wrist(env):
    env.renderer.update_scene(env.data, camera="wrist")
    return env.renderer.render()


def _move(env, q, secs, frames=None):
    start = env.data.ctrl[:6].copy()
    n = max(1, int(secs * 25))
    for i in range(n):
        o, _, _, _, _ = env.step(start + (i + 1) / n * (np.asarray(q) - start))
        if frames is not None:
            frames.append(o["pixels"])


def move_to_scan_pose(env, frames=None, duration: float = 0.8):
    """Move to a repeatable camera search pose without using object coordinates."""
    _move(
        env,
        [*SCAN_POSE, float(env.data.ctrl[5])],
        duration,
        frames,
    )


def aim_at(
    env,
    target: str = "red_block",
    frames=None,
    attempts: int = 2,
    allow_scan: bool = True,
):
    """스캔(필요시) → 적응형 센터링. 실패 시 재스캔 재시도. 반환: (found, centered)."""
    found = False
    for _ in range(max(1, attempts)):
        found, centered = _aim_once(env, target, frames, allow_scan=allow_scan)
        if centered:
            return True, True
        if found:  # 찾았는데 센터링 실패 → 스캔부터 다시 (다른 초기 조건)
            _move(env, [0, -0.6, 0.3, 1.2, 0, env.data.ctrl[5]], 0.5, frames)
    return found, False


def _aim_once(env, target, frames, allow_scan=True):
    """1회 시도: 스캔 → 적응형 센터링.

    적응형: 급강하 시점에선 pan이 화면 회전이 되는 등 부호 가정이 깨질 수 있어
    오차가 악화되면 부호를 뒤집는다. 손목(wf) 포화 시 어깨(lift)로 보조.
    """
    ctrl = env.data.ctrl[:6].copy()
    pan, lift, elbow, wf = (float(ctrl[i]) for i in range(4))
    loc = locate_color(_wrist(env), target)
    if loc is None and allow_scan:  # 스캔 자세로 pan 스윕
        lift, elbow, wf = SCAN_POSE[1], SCAN_POSE[2], SCAN_POSE[3]
        for pan in np.linspace(-1.2, 1.2, 7):
            _move(env, [pan, lift, elbow, wf, 0, ctrl[5]], 0.5, frames)
            loc = locate_color(_wrist(env), target)
            if loc is not None:
                break
    if loc is None:
        return False, False

    tol = 50 if target == "green_zone" else 20
    gain, sign_p, sign_w = 0.0015, 1.0, 1.0
    prev = None
    best_score = max(abs(loc[0] - CENTER[0]), abs(loc[1] - CENTER[1]))
    best_q = np.array([pan, lift, elbow, wf, 0, env.data.ctrl[5]], dtype=float)
    dx = dy = 999.0
    for _ in range(14):
        cx, cy = loc
        dx, dy = cx - CENTER[0], cy - CENTER[1]
        if abs(dx) < tol and abs(dy) < tol:
            break
        if prev is not None:
            if abs(dx) > abs(prev[0]) + 10:
                sign_p = -sign_p
            if abs(dy) > abs(prev[1]) + 10:
                sign_w = -sign_w
        prev = (dx, dy)
        pan_old, wf_old = pan, wf
        pan = float(np.clip(pan - sign_p * gain * dx, -1.9, 1.9))
        wf = float(np.clip(wf + sign_w * gain * dy, -1.6, 1.6))
        if abs(wf - wf_old) < 1e-4 and abs(dy) > 50:  # 손목 포화 → 어깨 보조
            # 대상이 화면 아래(dy>0)로 도망 = 너무 가까이 내려다봄 → 팔을 들어(뒤로)
            # 부감각을 완만하게. (숙이면 가까운 대상이 더 아래로 도망 — v2.1 디버깅 교훈)
            lift = float(np.clip(lift - 0.12 * np.sign(dy), -1.7, 1.7))
        _move(env, [pan, lift, elbow, wf, 0, env.data.ctrl[5]], 0.35, frames)
        nxt = locate_color(_wrist(env), target)
        if nxt is None:  # 시야 상실 → 반보 후퇴 + 감쇠
            pan, wf = pan_old, wf_old
            _move(env, [pan, lift, elbow, wf, 0, env.data.ctrl[5]], 0.35, frames)
            gain *= 0.5
            nxt = locate_color(_wrist(env), target)
            if nxt is None:
                _move(env, best_q, 0.35, frames)
                restored = locate_color(_wrist(env), target)
                restored_centered = (
                    restored is not None
                    and max(
                        abs(restored[0] - CENTER[0]),
                        abs(restored[1] - CENTER[1]),
                    ) < tol + 5
                )
                return True, bool(restored_centered)
        loc = nxt
        score = max(abs(loc[0] - CENTER[0]), abs(loc[1] - CENTER[1]))
        if score < best_score:
            best_score = score
            best_q = np.array(
                [pan, lift, elbow, wf, 0, env.data.ctrl[5]], dtype=float
            )
    final_score = max(abs(loc[0] - CENTER[0]), abs(loc[1] - CENTER[1]))
    if best_score < final_score:
        _move(env, best_q, 0.35, frames)
    final = locate_color(_wrist(env), target)
    centered = (
        final is not None
        and max(abs(final[0] - CENTER[0]), abs(final[1] - CENTER[1])) < tol + 5
    )
    return True, bool(centered)
