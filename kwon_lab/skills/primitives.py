"""SO-101 모션 프리미티브: IK 기반 move_to + 그리퍼 제어.

System 2(LLM)가 관절각 대신 이 함수들을 도구로 사용한다 (PROJECT.md 설계 원칙).
IK는 MuJoCo 야코비안 기반 감쇠 최소제곱법(DLS) — 의존성 없음.
인터페이스(move_to 시그니처)만 고정하면 실물에서는 백엔드를 Placo IK로 교체 가능.
"""

import mujoco
import numpy as np

ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
EE_SITE = "gripperframe"
GRIPPER_OPEN, GRIPPER_CLOSED = 1.5, 0.1  # gripper 관절각 (rad)


def _skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def solve_ik(model, target_pos, q_init, point_down=False, pitch_deg=60, n_iter=300,
             tol=0.002, damping=0.05, step=0.5, ori_weight=0.3):
    """목표 위치(x,y,z)에 그리퍼 끝(gripperframe)을 보내는 팔 관절각 5개를 푼다.

    point_down=True면 손가락 방향(사이트 x축)을 수직 아래(0,0,-1)로 정렬하는
    조건을 함께 푼다 — 위에서 내려찍는 그랩 자세.

    Returns: (q(5,), 최종 위치 오차 m)
    """
    data = mujoco.MjData(model)  # 시뮬 상태를 건드리지 않는 작업용 사본
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE)
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_JOINTS]
    dofs = np.array([model.jnt_dofadr[j] for j in joint_ids])
    qadr = np.array([model.jnt_qposadr[j] for j in joint_ids])
    lo = model.jnt_range[joint_ids, 0]
    hi = model.jnt_range[joint_ids, 1]

    q = np.clip(np.asarray(q_init, dtype=float).copy(), lo, hi)
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    target = np.asarray(target_pos, dtype=float)
    # 손가락이 향할 방향: 완전 수직(0,0,-1)은 팔 길이상 도달 불가 영역이 넓어,
    # 베이스에서 목표를 향한 방사 방향으로 60° 기울여 내려찍는 자세를 쓴다.
    radial = target.copy()
    radial[2] = 0.0
    radial /= max(np.linalg.norm(radial), 1e-6)
    a_des = radial * np.cos(np.radians(pitch_deg)) + np.array([0, 0, -np.sin(np.radians(pitch_deg))])

    pos_err = np.inf
    for _ in range(n_iter):
        data.qpos[qadr] = q
        mujoco.mj_forward(model, data)
        e_pos = target - data.site_xpos[site_id]
        pos_err = np.linalg.norm(e_pos)

        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        if point_down:
            a_cur = data.site_xmat[site_id].reshape(3, 3)[:, 0]  # 손가락 방향 = x축
            e_rot = np.cross(a_cur, a_des)  # 정렬에 필요한 회전 (axis*sin)
            if pos_err < tol and np.linalg.norm(e_rot) < 0.05:
                break
            J = np.vstack([jacp[:, dofs], ori_weight * (-_skew(a_cur) @ jacr[:, dofs])])
            e = np.concatenate([e_pos, ori_weight * e_rot])
        else:
            if pos_err < tol:
                break
            J = jacp[:, dofs]
            e = e_pos

        dq = J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(J.shape[0]), e)
        q = np.clip(q + step * dq, lo, hi)

    return q, float(pos_err)


def move_to(env, xyz, gripper=None, duration=1.2, point_down=False, pitch_deg=60):
    """그리퍼 끝을 (x,y,z)로 보낸다. gripper: None=유지 / 값=목표 관절각.

    point_down=True면 손가락을 pitch_deg만큼 아래로 기울인 채 이동 (그랩 자세).
    Returns: (마지막 info, IK 잔여 오차 m, 프레임 리스트)
    """
    q_now = env.data.qpos[:5].copy()
    q_target, ik_err = solve_ik(env.model, xyz, q_now, point_down=point_down, pitch_deg=pitch_deg)
    g_now = float(env.data.ctrl[5])
    g_target = g_now if gripper is None else float(gripper)

    frames, info = [], None
    n_steps = max(1, int(duration * env.metadata["render_fps"]))
    for i in range(n_steps):
        a = (i + 1) / n_steps
        action = np.concatenate([q_now + a * (q_target - q_now), [g_now + a * (g_target - g_now)]])
        obs, _, _, _, info = env.step(action)
        frames.append(obs["pixels"])
    return info, ik_err, frames


def set_gripper(env, value, duration=0.6):
    """그리퍼만 여닫는다 (팔은 현재 목표 유지)."""
    frames, info = [], None
    arm = env.data.ctrl[:5].copy()
    g_now = float(env.data.ctrl[5])
    n_steps = max(1, int(duration * env.metadata["render_fps"]))
    for i in range(n_steps):
        a = (i + 1) / n_steps
        obs, _, _, _, info = env.step(np.concatenate([arm, [g_now + a * (value - g_now)]]))
        frames.append(obs["pixels"])
    return info, frames


# ── 상위 스킬: 오늘 검증된 그랩 레시피를 함수로 봉인 ──────────

POCKET_OFFSET = np.array([-0.03, 0.0, 0.01])  # 실측: 포켓 중심의 site계 오프셋
GRASP_PITCH = 45  # 접근 각도 (deg)


def pick(env, block_pos, frames=None):
    """블록 위치를 받아 집어 올린다. 성공 여부 반환.

    검증된 레시피(2026-08-10): 45도 접근 → 손가락 축 따라 삽입(클로머신) →
    닫기 → 리프트. 10시드 기준 그랩 성공률 10/10.
    """
    import mujoco as _mj
    b = np.asarray(block_pos, dtype=float)
    sid = _mj.mj_name2id(env.model, _mj.mjtObj.mjOBJ_SITE, EE_SITE)
    _, _, f = move_to(env, b + [0, 0, 0.12], gripper=GRIPPER_OPEN, duration=1.3,
                      point_down=True, pitch_deg=GRASP_PITCH)
    if frames is not None: frames += f
    R = env.data.site_xmat[sid].reshape(3, 3)
    gf_target = b - R @ POCKET_OFFSET
    _, _, f = move_to(env, gf_target - R[:, 0] * 0.06, duration=0.9)
    if frames is not None: frames += f
    _, _, f = move_to(env, gf_target, duration=0.8)
    if frames is not None: frames += f
    _, f = set_gripper(env, -0.1, duration=0.7)
    if frames is not None: frames += f
    info, _, f = move_to(env, [b[0], b[1], 0.18], duration=1.2)
    if frames is not None: frames += f
    return info["block_height"] > 0.08, info


def place(env, target_xy, frames=None):
    """쥔 물체를 목표 지점 위로 옮겨 놓는다."""
    _, _, f = move_to(env, [target_xy[0], target_xy[1], 0.12], duration=1.5)
    if frames is not None: frames += f
    info, f = set_gripper(env, GRIPPER_OPEN, duration=0.6)
    if frames is not None: frames += f
    _, _, f2 = move_to(env, [target_xy[0], target_xy[1], 0.20], duration=0.8)
    if frames is not None: frames += f2
    return info["dist_to_target"] < 0.05, info
