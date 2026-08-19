"""Camera-supervised PICK -> TRANSPORT state transitions.

The module never reads simulator object or target poses.  Those privileged
values belong only in benchmark tools.  PLACE/release is intentionally not
part of this stage yet.
"""

from dataclasses import dataclass
from collections import deque

import numpy as np

from skills.block_reacquisition import (
    SIM_OVERHEAD_PIXEL_TO_TABLE,
    SIM_OVERHEAD_TARGET_PIXEL_TO_TABLE,
    locate_overhead_target,
    pick_until_verified,
)
from skills.primitives import (
    EE_SITE,
    GRIPPER_OPEN,
    PLACE_APPROACH_HEIGHT,
    PLACE_RELEASE_HEIGHT,
    hold_position,
    move_to,
    set_gripper,
)
from skills.vision_supervision import (
    GRIPPER_HOLDING_MAX,
    GRIPPER_HOLDING_MIN,
    DualCameraTaskMonitor,
    OverheadTaskMonitor,
    color_masks,
    observe_colors,
)

import mujoco


TRANSPORT_HEIGHT_M = 0.18
TRANSPORT_DURATION_S = 2.0
PLACE_ALIGN_TOLERANCE_PX = 10.0
PLACE_ALIGN_MAX_STEP_M = 0.025
PLACE_ALIGN_MAX_ITERATIONS = 4
PLACE_SETTLE_DURATION_S = 1.2
PLACE_CLEAR_POSITION = (0.30, 0.0, 0.20)
PLACE_FINAL_CONFIRM_FRAMES = 10
PLACE_FINAL_MAX_SPREAD_PX = 3.0
PLACE_CAMERA_SUCCESS_COVERAGE = 0.78


@dataclass
class _TransportDropDetected(Exception):
    step: int
    signals: dict


def _end_effector_position(env):
    site = mujoco.mj_name2id(
        env.model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE
    )
    return env.data.site_xpos[site].copy()


def _red_centroid(frame):
    red, _ = color_masks(frame)
    if int(red.sum()) < 100:
        return None
    ys, xs = np.nonzero(red)
    return np.array([float(xs.mean()), float(ys.mean())], dtype=float)


def transport_verified_pick(
    env,
    target_xy,
    overhead_calibration_frame,
    frames=None,
    on_transport_step=None,
    duration=TRANSPORT_DURATION_S,
):
    """Move a camera-verified grasp near the target and abort on a drop.

    ``on_transport_step`` is an optional benchmark hook.  Deployment code
    leaves it unset.  The gripper stays closed; release belongs to PLACE.
    """
    if frames is None:
        frames = []
    target_xy = np.asarray(target_xy, dtype=float)
    monitor = DualCameraTaskMonitor(overhead_calibration_frame)
    initial_obs = env._get_obs()
    primed = monitor.prime_attached(initial_obs["pixels"])
    report = {
        "state": "TRANSPORT",
        "monitor_primed": bool(primed),
        "target_xy": tuple(map(float, target_xy)),
        "transport_steps": 0,
        "drop_detected": False,
        "drop_detected_step": None,
        "last_signals": None,
        "final_gripper_joint": float(initial_obs["agent_pos"][-1]),
        "final_wrist_block_visible": False,
        "ready_for_place": False,
        "next_state": "PICK" if not primed else "TRANSPORT",
        "failure_reason": None if primed else "transport_handoff_not_visible",
    }
    if not primed:
        return report

    step = 0

    def supervise(obs):
        nonlocal step
        step += 1
        frames.append(obs["pixels"])
        signals = monitor.update(
            obs["pixels"],
            env.render_overhead(),
            obs["agent_pos"],
            step,
        )
        report["last_signals"] = signals
        report["transport_steps"] = step
        if signals["transport_drop"]:
            raise _TransportDropDetected(step, signals)
        if on_transport_step is not None:
            on_transport_step(env, step, signals)

    try:
        # This is the safe, high transport leg only.  Camera-guided descent
        # and the 6 cm release are handled by the following PLACE state.
        move_to(
            env,
            [target_xy[0], target_xy[1], TRANSPORT_HEIGHT_M],
            duration=duration,
            on_observation=supervise,
        )
    except _TransportDropDetected as event:
        report["drop_detected"] = True
        report["drop_detected_step"] = event.step
        report["last_signals"] = event.signals
        report["next_state"] = "PICK"
        report["failure_reason"] = "transport_drop"
        return report

    final_obs = env._get_obs()
    final_gripper = float(final_obs["agent_pos"][-1])
    final_visible = observe_colors(final_obs["pixels"]).red_block.visible
    final_holding = (
        GRIPPER_HOLDING_MIN <= final_gripper <= GRIPPER_HOLDING_MAX
        and final_visible
        and not monitor.transport_drop
    )
    report["final_gripper_joint"] = final_gripper
    report["final_wrist_block_visible"] = bool(final_visible)
    report["ready_for_place"] = bool(final_holding)
    report["next_state"] = "PLACE" if final_holding else "PICK"
    report["failure_reason"] = None if final_holding else "holding_lost_at_target"
    return report


def place_verified_transport(
    env,
    target_pixel,
    overhead_calibration_frame,
    frames=None,
    target_matrix=SIM_OVERHEAD_TARGET_PIXEL_TO_TABLE,
    on_place_step=None,
    conservative=False,
):
    """Descend to 6 cm, visually align, release, and verify 75% coverage.

    The carried block is located only from overhead RGB.  Robot kinematics are
    used for incremental end-effector motion; object and target simulator poses
    are never read.  A drop before intentional release returns to PICK.
    """
    if frames is None:
        frames = []
    target_pixel = np.asarray(target_pixel, dtype=float)
    transport_monitor = DualCameraTaskMonitor(overhead_calibration_frame)
    initial_obs = env._get_obs()
    primed = transport_monitor.prime_attached(initial_obs["pixels"])
    report = {
        "state": "PLACE",
        "monitor_primed": bool(primed),
        "release_height_m": PLACE_RELEASE_HEIGHT,
        "alignment_iterations": 0,
        "alignment_errors_px": [],
        "alignment_skipped_reason": None,
        "place_steps": 0,
        "drop_detected": False,
        "drop_detected_step": None,
        "released": False,
        "stable_success_frames_required": PLACE_FINAL_CONFIRM_FRAMES,
        "camera_success_coverage_required": PLACE_CAMERA_SUCCESS_COVERAGE,
        "image_target_coverage": 0.0,
        "table_area_ratio": 0.0,
        "place_evidence_frames": 0,
        "camera_place_confirmed": False,
        "next_state": "PICK",
        "failure_reason": None if primed else "place_handoff_not_visible",
    }
    if not primed:
        return report

    step = 0

    def supervise_motion(obs):
        nonlocal step
        step += 1
        frames.append(obs["pixels"])
        signals = transport_monitor.update(
            obs["pixels"],
            env.render_overhead(),
            obs["agent_pos"],
            step,
        )
        report["place_steps"] = step
        if signals["transport_drop"]:
            raise _TransportDropDetected(step, signals)
        # During PLACE approach, any confirmed detachment is premature even
        # when the object happens to land inside the target.  Only the explicit
        # gripper-open transition below is an intentional release.
        if signals["wrist_detached"] and signals["overhead_stationary"]:
            raise _TransportDropDetected(step, signals)
        if on_place_step is not None:
            on_place_step(env, step, signals)

    place_monitor = OverheadTaskMonitor(overhead_calibration_frame)
    final_history = deque(maxlen=PLACE_FINAL_CONFIRM_FRAMES)

    def observe_release_motion(obs):
        nonlocal step
        step += 1
        frames.append(obs["pixels"])
        report["place_steps"] = step

    def supervise_final_settle(obs):
        nonlocal step
        step += 1
        frames.append(obs["pixels"])
        overhead_frame = env.render_overhead()
        _, signals = place_monitor.update(
            overhead_frame, obs["agent_pos"], grasp_confirmed=True
        )
        report["place_steps"] = step
        report["table_area_ratio"] = float(signals["table_area_ratio"])
        red, _ = color_masks(overhead_frame)
        red_area = int(red.sum())
        if red_area >= 100:
            ys, xs = np.nonzero(red)
            centroid = np.array([float(xs.mean()), float(ys.mean())])
            coverage = float(
                np.count_nonzero(red & place_monitor.target_mask) / red_area
            )
        else:
            centroid = None
            coverage = 0.0
        final_history.append((centroid, coverage))
        stable = False
        if (
            len(final_history) == PLACE_FINAL_CONFIRM_FRAMES
            and all(item[0] is not None for item in final_history)
            and all(
                item[1] >= PLACE_CAMERA_SUCCESS_COVERAGE
                for item in final_history
            )
        ):
            centroids = np.asarray([item[0] for item in final_history])
            center = np.mean(centroids, axis=0)
            spread = float(
                np.linalg.norm(centroids - center, axis=1).max()
            )
            stable = spread <= PLACE_FINAL_MAX_SPREAD_PX
        report["place_evidence_frames"] = (
            PLACE_FINAL_CONFIRM_FRAMES if stable else 0
        )
        report["image_target_coverage"] = coverage

    def release_clear_and_verify():
        final_history.clear()
        set_gripper(
            env,
            GRIPPER_OPEN,
            duration=0.45,
            on_observation=observe_release_motion,
        )
        report["released"] = True
        ee = _end_effector_position(env)
        move_to(
            env,
            [ee[0], ee[1], PLACE_APPROACH_HEIGHT],
            gripper=GRIPPER_OPEN,
            duration=0.8,
            on_observation=observe_release_motion,
        )
        move_to(
            env,
            PLACE_CLEAR_POSITION,
            gripper=GRIPPER_OPEN,
            duration=1.0,
            on_observation=observe_release_motion,
        )
        hold_position(
            env,
            duration=PLACE_SETTLE_DURATION_S,
            on_observation=supervise_final_settle,
        )
        confirmed = (
            report["place_evidence_frames"] >= PLACE_FINAL_CONFIRM_FRAMES
        )
        report["camera_place_confirmed"] = confirmed
        report["next_state"] = "DONE" if confirmed else "PICK"
        return confirmed

    try:
        # First descend vertically to the safe place approach height.
        ee = _end_effector_position(env)
        move_to(
            env,
            [ee[0], ee[1], PLACE_APPROACH_HEIGHT],
            duration=1.2 if conservative else 0.8,
            on_observation=supervise_motion,
        )

        # Descend to the requested 6 cm release height before final image
        # alignment.  At this height the block remains above the table.
        ee = _end_effector_position(env)
        move_to(
            env,
            [ee[0], ee[1], PLACE_RELEASE_HEIGHT],
            duration=1.4 if conservative else 0.9,
            on_observation=supervise_motion,
        )

        # Image-space visual servo.  The planar map converts only the pixel
        # correction into an incremental robot XY command.  Iteration absorbs
        # height/parallax mismatch instead of reading a privileged block pose.
        linear_map = np.asarray(target_matrix, dtype=float)[:, :2]
        for _ in range(PLACE_ALIGN_MAX_ITERATIONS):
            block_pixel = _red_centroid(env.render_overhead())
            if block_pixel is None:
                # At 6 cm the fingers can fully occlude the block in the fixed
                # view.  Keep the already validated transport XY, release, and
                # let the unobstructed post-retreat 75% check decide DONE/PICK.
                report["alignment_skipped_reason"] = "block_occluded_at_6cm"
                break
            pixel_error = target_pixel - block_pixel
            error_norm = float(np.linalg.norm(pixel_error))
            report["alignment_errors_px"].append(error_norm)
            if error_norm <= PLACE_ALIGN_TOLERANCE_PX:
                break
            delta_xy = linear_map @ pixel_error
            delta_norm = float(np.linalg.norm(delta_xy))
            max_step = 0.015 if conservative else PLACE_ALIGN_MAX_STEP_M
            if delta_norm > max_step:
                delta_xy *= max_step / delta_norm
            ee = _end_effector_position(env)
            move_to(
                env,
                [
                    ee[0] + float(delta_xy[0]),
                    ee[1] + float(delta_xy[1]),
                    PLACE_RELEASE_HEIGHT,
                ],
                duration=0.7 if conservative else 0.45,
                on_observation=supervise_motion,
            )
            report["alignment_iterations"] += 1
    except _TransportDropDetected as event:
        report["drop_detected"] = True
        report["drop_detected_step"] = event.step
        landed_in_target = release_clear_and_verify()
        report["drop_landed_in_target"] = bool(landed_in_target)
        report["failure_reason"] = (
            None if landed_in_target else "drop_during_place_approach"
        )
        return report

    confirmed = release_clear_and_verify()
    report["failure_reason"] = None if confirmed else "place_not_confirmed"
    return report


def pick_then_transport(
    env,
    frames=None,
    block_matrix=SIM_OVERHEAD_PIXEL_TO_TABLE,
    target_matrix=SIM_OVERHEAD_TARGET_PIXEL_TO_TABLE,
    max_pick_attempts=6,
    on_transport_step=None,
):
    """Run the explicit PICK gate, then camera-supervised TRANSPORT."""
    if frames is None:
        frames = []
    overhead_calibration = env.render_overhead()
    target = locate_overhead_target(overhead_calibration, target_matrix)
    result = {
        "target_visible": target.visible,
        "target_pixel": target.pixel,
        "target_xy": target.table_xy,
        "target_pixels": target.pixels,
        "pick": None,
        "transport": None,
        "next_state": "SEARCH_TARGET" if not target.visible else "PICK",
    }
    if not target.visible:
        return result

    pick_report = pick_until_verified(
        env,
        frames=frames,
        matrix=block_matrix,
        max_attempts=max_pick_attempts,
    )
    result["pick"] = pick_report
    if not pick_report["success"]:
        result["next_state"] = "PICK"
        return result

    transport_report = transport_verified_pick(
        env,
        target.table_xy,
        overhead_calibration,
        frames=frames,
        on_transport_step=on_transport_step,
    )
    result["transport"] = transport_report
    result["next_state"] = transport_report["next_state"]
    return result


def _run_pick_transport_place_cycle(
    env,
    overhead_calibration,
    target,
    frames=None,
    block_matrix=SIM_OVERHEAD_PIXEL_TO_TABLE,
    target_matrix=SIM_OVERHEAD_TARGET_PIXEL_TO_TABLE,
    max_pick_attempts=6,
    recovery_pick=False,
    on_transport_step=None,
    on_place_step=None,
):
    """Run one PICK -> TRANSPORT -> PLACE cycle with fixed task context."""
    if frames is None:
        frames = []
    result = {
        "target_visible": target.visible,
        "target_pixel": target.pixel,
        "target_xy": target.table_xy,
        "pick": None,
        "transport": None,
        "place": None,
        "next_state": "SEARCH_TARGET" if not target.visible else "PICK",
    }
    if not target.visible:
        return result

    result["pick"] = pick_until_verified(
        env,
        frames=frames,
        matrix=block_matrix,
        max_attempts=max_pick_attempts,
        recovery=recovery_pick,
    )
    if not result["pick"]["success"]:
        result["next_state"] = "PICK"
        return result

    result["transport"] = transport_verified_pick(
        env,
        target.table_xy,
        overhead_calibration,
        frames=frames,
        on_transport_step=on_transport_step,
    )
    if not result["transport"]["ready_for_place"]:
        result["next_state"] = "PICK"
        return result

    result["place"] = place_verified_transport(
        env,
        target.pixel,
        overhead_calibration,
        frames=frames,
        target_matrix=target_matrix,
        on_place_step=on_place_step,
    )
    result["next_state"] = result["place"]["next_state"]
    return result


def pick_transport_place(
    env,
    frames=None,
    block_matrix=SIM_OVERHEAD_PIXEL_TO_TABLE,
    target_matrix=SIM_OVERHEAD_TARGET_PIXEL_TO_TABLE,
    max_pick_attempts=6,
    on_transport_step=None,
    on_place_step=None,
):
    """Run one camera-gated PICK -> TRANSPORT -> PLACE cycle."""
    overhead_calibration = env.render_overhead()
    target = locate_overhead_target(overhead_calibration, target_matrix)
    return _run_pick_transport_place_cycle(
        env,
        overhead_calibration,
        target,
        frames=frames,
        block_matrix=block_matrix,
        target_matrix=target_matrix,
        max_pick_attempts=max_pick_attempts,
        recovery_pick=False,
        on_transport_step=on_transport_step,
        on_place_step=on_place_step,
    )


def run_pick_place_until_done(
    env,
    frames=None,
    block_matrix=SIM_OVERHEAD_PIXEL_TO_TABLE,
    target_matrix=SIM_OVERHEAD_TARGET_PIXEL_TO_TABLE,
    max_cycles=4,
    max_pick_attempts_per_cycle=6,
    on_transport_step=None,
    on_place_step=None,
    on_cycle_complete=None,
):
    """Repeat recovery cycles until DONE or a bounded safety stop.

    The initial unobstructed target observation is retained across recovery
    cycles.  A dropped block can therefore occlude the green zone without
    changing the task target.  Each recovery still re-runs overhead block
    search, wrist alignment, and camera-verified PICK.
    """
    if frames is None:
        frames = []
    if max_cycles < 1:
        raise ValueError("max_cycles must be at least 1")

    overhead_calibration = env.render_overhead()
    target = locate_overhead_target(overhead_calibration, target_matrix)
    task_report = {
        "target_visible": target.visible,
        "target_pixel": target.pixel,
        "target_xy": target.table_xy,
        "max_cycles": int(max_cycles),
        "cycles_attempted": 0,
        "recovery_cycles": 0,
        "success": False,
        "final_state": "SEARCH_TARGET" if not target.visible else "PICK",
        "stop_reason": (
            "target_not_visible" if not target.visible else None
        ),
        "cycle_reports": [],
    }
    if not target.visible:
        return task_report

    for cycle_index in range(1, max_cycles + 1):
        cycle = _run_pick_transport_place_cycle(
            env,
            overhead_calibration,
            target,
            frames=frames,
            block_matrix=block_matrix,
            target_matrix=target_matrix,
            max_pick_attempts=max_pick_attempts_per_cycle,
            recovery_pick=cycle_index > 1,
            on_transport_step=on_transport_step,
            on_place_step=on_place_step,
        )
        cycle["cycle"] = cycle_index
        task_report["cycle_reports"].append(cycle)
        task_report["cycles_attempted"] = cycle_index
        task_report["final_state"] = cycle["next_state"]
        if on_cycle_complete is not None:
            on_cycle_complete(cycle)
        if cycle["next_state"] == "DONE":
            task_report["success"] = True
            task_report["recovery_cycles"] = cycle_index - 1
            task_report["stop_reason"] = "done"
            return task_report
        if cycle["next_state"] != "PICK":
            task_report["recovery_cycles"] = max(0, cycle_index - 1)
            task_report["stop_reason"] = "non_recoverable_state"
            return task_report

    task_report["recovery_cycles"] = max(0, max_cycles - 1)
    task_report["stop_reason"] = "safety_cycle_limit"
    return task_report
