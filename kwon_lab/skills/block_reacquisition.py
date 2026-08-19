"""Overhead-to-wrist block reacquisition without runtime object coordinates.

The fixed camera supplies a coarse table position. Robot kinematics move the
open gripper to a safe approach height, then wrist RGB performs local aiming.
Simulator object poses are intentionally absent from this module.
"""

from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import numpy as np

from skills.aiming import aim_at, locate_color
from skills.primitives import (
    GRIPPER_OPEN,
    hold_position,
    move_joints,
    pick,
    retreat_vertical,
    set_gripper,
    solve_ik,
)
from skills.vision_supervision import (
    GRIPPER_HOLDING_MAX,
    GRIPPER_HOLDING_MIN,
    VisionPickMonitor,
    color_masks,
    observe_colors,
)


# Calibrated from seven unobstructed table points in the simulated 52 cm
# overhead U20CAM approximation. Real hardware must replace this one-time
# pixel-to-table calibration; runtime tracking uses only the matrix below.
SIM_OVERHEAD_PIXEL_TO_TABLE = np.array(
    [
        [8.89290574e-4, 1.91560868e-7, -3.68790576e-1],
        [2.23295987e-6, -8.89375885e-4, 3.58317319e-1],
    ],
    dtype=float,
)
# The green target lies on the table plane, while the block calibration above
# follows the visible top face of a 6 cm-tall block.  A separate planar map
# avoids the predictable parallax bias between those two heights.  It is
# derived from the simulated overhead camera intrinsics/extrinsics, not from a
# runtime target pose.  Real hardware must calibrate this matrix as well.
SIM_OVERHEAD_TARGET_PIXEL_TO_TABLE = np.array(
    [
        [1.000119753e-3, 0.0, -4.40076642e-1],
        [0.0, -1.000119753e-3, 4.00043111e-1],
    ],
    dtype=float,
)

WORKSPACE_X = (0.10, 0.32)
WORKSPACE_Y = (-0.16, 0.24)
MAX_COARSE_IK_ERROR_M = 0.015
APPROACH_CONFIGS = (
    (0.16, 30),
    (0.14, 40),
    (0.16, 40),
    (0.16, 50),
    (0.18, 50),
    (0.20, 50),
)
LOCAL_SEARCH_OFFSETS_M = (
    (0.0, 0.0),
    (-0.025, 0.0),
    (0.025, 0.0),
    (0.0, -0.025),
    (0.0, 0.025),
)
CAMERA_APPROACH_HEIGHT = 0.14
CAMERA_SEED_REST = (-0.30, -0.10, 1.50, -1.30)
TABLE_BLOCK_CENTER_Z = 0.03
TIPPED_BLOCK_CENTER_Z = 0.02
PICK_START_POSE = np.zeros(5, dtype=float)
OVERHEAD_REACQUIRE_MIN_PIXELS = 100
OVERHEAD_POSE_MIN_PIXELS = 800
UPRIGHT_ASPECT_RATIO_MAX = 1.18
TIPPED_ASPECT_RATIO_MIN = 1.30
# A side occluder can turn a square top face into a narrow visible rectangle.
# The major silhouette extent (sqrt(area * aspect)) remains near the original
# 4 cm edge, while a genuinely tipped 4x6 cm face has a substantially longer
# major extent at the fixed 720p/52 cm setup.
UPRIGHT_OCCLUDED_ASPECT_MAX = 1.55
UPRIGHT_MAJOR_EXTENT_MAX_PX = 54.0
TIPPED_MAJOR_EXTENT_MIN_PX = 58.0
CLEAR_VIEW_ACCEPT_SCORE = 0.75
CLEAR_VIEW_SETTLE_S = 0.20

# Dedicated high, side-clear poses. Both put the end effector at about 24 cm
# and outside the simulated task workspace. A safe vertical retreat always
# precedes either joint-space move. The first pose is chosen from the latest
# visible block side; the alternate is tried only when visual confidence is
# insufficient.
CLEAR_VIEW_POSES = (
    ("CLEAR_VIEW_A", np.array([-1.298, -1.130, 0.807, 0.388, 0.062])),
    ("CLEAR_VIEW_B", np.array([1.290, -1.129, 0.805, 0.389, -0.066])),
)
PICK_RETRY_OFFSETS_M = (
    (0.0, 0.0),
    (0.0, 0.01),
    (0.0, -0.01),
    (0.01, 0.0),
    (-0.01, 0.0),
    (0.01, 0.01),
)
RECOVERY_PICK_CONFIGS = (
    ((0.0, 0.01), 0.02, 0.0),
    ((-0.01, 0.01), 0.02, 0.0),
    ((0.0, 0.01), 0.02, -np.pi / 4),
    ((-0.01, 0.01), 0.02, np.pi / 2),
    ((0.0, 0.0), 0.02, None),
    ((0.0, 0.0), 0.025, 0.0),
    ((0.0, -0.01), 0.02, 0.0),
)
UPRIGHT_RECOVERY_PICK_CONFIGS = (
    ((0.0, 0.0), TABLE_BLOCK_CENTER_Z, 0.0),
    ((0.0, 0.01), TABLE_BLOCK_CENTER_Z, 0.0),
    ((-0.01, 0.0), TABLE_BLOCK_CENTER_Z, 0.0),
    ((0.01, 0.0), TABLE_BLOCK_CENTER_Z, 0.0),
    ((0.0, -0.01), TABLE_BLOCK_CENTER_Z, 0.0),
    ((0.0, 0.0), TABLE_BLOCK_CENTER_Z, np.pi / 2),
)
TIPPED_RECOVERY_PICK_CONFIGS = (
    ((0.0, 0.0), TIPPED_BLOCK_CENTER_Z, None),
    ((0.0, 0.01), TIPPED_BLOCK_CENTER_Z, 0.0),
    ((-0.01, 0.01), TIPPED_BLOCK_CENTER_Z, 0.0),
    ((0.0, 0.01), TIPPED_BLOCK_CENTER_Z, -np.pi / 4),
    ((-0.01, 0.01), TIPPED_BLOCK_CENTER_Z, np.pi / 2),
    ((0.0, 0.0), 0.025, 0.0),
)


@dataclass(frozen=True)
class OverheadBlockLocation:
    visible: bool
    pixel: tuple[float, float] | None
    table_xy: tuple[float, float] | None
    pixels: int
    reachable: bool
    orientation_rad: float | None
    aspect_ratio: float | None
    pose_class: str
    pose_confidence: float
    touches_border: bool


@dataclass(frozen=True)
class OverheadTargetLocation:
    visible: bool
    pixel: tuple[float, float] | None
    table_xy: tuple[float, float] | None
    pixels: int


def _select_foreground_component(mask, min_pixels):
    """Prefer the largest non-border color component over unrelated clutter."""
    binary = np.asarray(mask, dtype=np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    height, width = binary.shape
    candidates = []
    for label in range(1, count):
        x, y, w, h, area = (int(value) for value in stats[label])
        if area < min_pixels:
            continue
        touches_border = x == 0 or y == 0 or x + w == width or y + h == height
        candidates.append((label, area, touches_border))
    if not candidates:
        return np.zeros_like(mask, dtype=bool), 0, False
    non_border = [item for item in candidates if not item[2]]
    selected = max(non_border or candidates, key=lambda item: item[1])
    return labels == selected[0], selected[1], selected[2]


def pixel_to_table(pixel, matrix=SIM_OVERHEAD_PIXEL_TO_TABLE):
    """Convert one overhead pixel to table XY using affine or homography."""
    u, v = map(float, pixel)
    matrix = np.asarray(matrix, dtype=float)
    projected = matrix @ np.array([u, v, 1.0])
    if matrix.shape == (2, 3):
        xy = projected
    elif matrix.shape == (3, 3):
        if abs(projected[2]) < 1e-12:
            raise ValueError("Pixel-to-table homography produced a zero scale")
        xy = projected[:2] / projected[2]
    else:
        raise ValueError(
            "Pixel-to-table matrix must have shape (2, 3) or (3, 3)"
        )
    return float(xy[0]), float(xy[1])


def load_pixel_to_table_matrix(config_path, plane):
    """Load one real-camera plane calibration from a local JSON file."""
    payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
    try:
        matrix = payload["planes"][plane]["pixel_to_table_homography"]
    except KeyError as exc:
        available = sorted(payload.get("planes", {}))
        raise ValueError(
            f"Calibration plane {plane!r} is missing; available={available}"
        ) from exc
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError(f"Calibration plane {plane!r} is not a 3x3 matrix")
    return matrix


def load_real_camera_matrices(config_path):
    """Return pose-aware block maps and the target-table map."""
    block_matrices = {
        "UPRIGHT": load_pixel_to_table_matrix(config_path, "upright_top_6cm"),
        "TIPPED": load_pixel_to_table_matrix(config_path, "tipped_top_4cm"),
    }
    target_matrix = load_pixel_to_table_matrix(config_path, "target_table")
    return block_matrices, target_matrix


def _matrix_for_block_pose(matrix, pose_class):
    if isinstance(matrix, dict):
        selected = matrix.get(pose_class)
        if selected is None:
            selected = matrix.get("UPRIGHT")
        if selected is None:
            raise ValueError(
                "Pose-aware block calibration requires UPRIGHT and TIPPED matrices"
            )
        return np.asarray(selected, dtype=float)
    return np.asarray(matrix, dtype=float)


def locate_overhead_block(frame, matrix=SIM_OVERHEAD_PIXEL_TO_TABLE):
    """Locate the red block and reject estimates outside the robot workspace."""
    red_mask, _ = color_masks(frame)
    red_mask, pixels, touches_border = _select_foreground_component(
        red_mask, OVERHEAD_REACQUIRE_MIN_PIXELS
    )
    if pixels < OVERHEAD_REACQUIRE_MIN_PIXELS:
        return OverheadBlockLocation(
            False, None, None, pixels, False, None, None, "UNKNOWN", 0.0, False
        )
    ys, xs = np.nonzero(red_mask)
    centroid = (float(xs.mean()), float(ys.mean()))
    centered_pixels = np.column_stack(
        [xs.astype(float) - centroid[0], ys.astype(float) - centroid[1]]
    )
    covariance = np.cov(centered_pixels, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, 1e-9)
    aspect_ratio = float(np.sqrt(eigenvalues[-1] / eigenvalues[0]))
    pixel_axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    pose_class = "UNKNOWN"
    pose_confidence = 0.0
    if pixels >= OVERHEAD_POSE_MIN_PIXELS and not touches_border:
        major_extent = float(np.sqrt(pixels * aspect_ratio))
        if aspect_ratio <= UPRIGHT_ASPECT_RATIO_MAX:
            pose_class = "UPRIGHT"
            pose_confidence = float(
                np.clip(
                    (UPRIGHT_ASPECT_RATIO_MAX - aspect_ratio)
                    / (UPRIGHT_ASPECT_RATIO_MAX - 1.0),
                    0.0,
                    1.0,
                )
            )
        elif (
            aspect_ratio <= UPRIGHT_OCCLUDED_ASPECT_MAX
            and major_extent <= UPRIGHT_MAJOR_EXTENT_MAX_PX
        ):
            pose_class = "UPRIGHT"
            pose_confidence = float(
                np.clip(
                    (UPRIGHT_MAJOR_EXTENT_MAX_PX - major_extent) / 14.0,
                    0.15,
                    0.75,
                )
            )
        elif (
            aspect_ratio >= TIPPED_ASPECT_RATIO_MIN
            and major_extent >= TIPPED_MAJOR_EXTENT_MIN_PX
        ):
            pose_class = "TIPPED"
            pose_confidence = float(
                np.clip(
                    (aspect_ratio - 1.20) / 0.28,
                    0.0,
                    1.0,
                )
            )
    pose_matrix = _matrix_for_block_pose(matrix, pose_class)
    xy = pixel_to_table(centroid, pose_matrix)
    axis_pixel = np.asarray(centroid) + 10.0 * pixel_axis
    axis_xy = np.asarray(pixel_to_table(axis_pixel, pose_matrix))
    world_axis = axis_xy - np.asarray(xy)
    orientation = float(np.arctan2(world_axis[1], world_axis[0]))
    reachable = (
        WORKSPACE_X[0] <= xy[0] <= WORKSPACE_X[1]
        and WORKSPACE_Y[0] <= xy[1] <= WORKSPACE_Y[1]
    )
    return OverheadBlockLocation(
        True,
        centroid,
        xy,
        pixels,
        reachable,
        orientation,
        aspect_ratio,
        pose_class,
        pose_confidence,
        touches_border,
    )


def _location_visibility_score(location):
    """Score whether one fixed-camera observation is safe to act on."""
    if not location.visible:
        return 0.0
    area_score = float(
        np.clip(
            (location.pixels - OVERHEAD_REACQUIRE_MIN_PIXELS) / 1700.0,
            0.0,
            1.0,
        )
    )
    pose_score = (
        location.pose_confidence if location.pose_class != "UNKNOWN" else 0.0
    )
    return float(
        0.45 * area_score
        + 0.35 * pose_score
        + 0.10 * float(location.reachable)
        + 0.10 * float(not location.touches_border)
    )


def _observe_clear_view(env, pose_name, pose, matrix, frames):
    """Move to one clear pose and aggregate several stationary observations."""
    _, motion_frames = move_joints(
        env, pose, gripper=GRIPPER_OPEN, duration=0.9
    )
    frames.extend(motion_frames)
    samples = []

    def observe(_obs):
        samples.append(locate_overhead_block(env.render_overhead(), matrix))

    _, settle_frames = hold_position(
        env, duration=CLEAR_VIEW_SETTLE_S, on_observation=observe
    )
    frames.extend(settle_frames)
    if not samples:
        samples.append(locate_overhead_block(env.render_overhead(), matrix))
    best = max(samples, key=_location_visibility_score)
    visible_pixels = [sample.pixel for sample in samples if sample.pixel]
    if len(visible_pixels) >= 2:
        pixels = np.asarray(visible_pixels, dtype=float)
        center = pixels.mean(axis=0)
        spread_px = float(np.linalg.norm(pixels - center, axis=1).max())
    else:
        spread_px = float("inf")
    temporal_score = (
        float(np.clip(1.0 - spread_px / 5.0, 0.0, 1.0))
        if np.isfinite(spread_px)
        else 0.0
    )
    score = 0.90 * _location_visibility_score(best) + 0.10 * temporal_score
    return best, {
        "pose": pose_name,
        "score": float(score),
        "sufficient": bool(
            score >= CLEAR_VIEW_ACCEPT_SCORE
            and best.pose_class != "UNKNOWN"
            and best.reachable
        ),
        "visible": best.visible,
        "pixels": best.pixels,
        "pixel_spread": None if not np.isfinite(spread_px) else spread_px,
        "aspect_ratio": best.aspect_ratio,
        "block_pose_class": best.pose_class,
        "block_pose_confidence": best.pose_confidence,
        "table_xy": best.table_xy,
    }


def select_clear_view_observation(
    env, frames=None, matrix=SIM_OVERHEAD_PIXEL_TO_TABLE
):
    """Choose one of two high poses using fixed-camera visibility evidence."""
    if frames is None:
        frames = []
    initial = locate_overhead_block(env.render_overhead(), matrix)
    poses = list(CLEAR_VIEW_POSES)
    if initial.visible and initial.table_xy is not None:
        # Keep the arm on the opposite side of the last visible block Y.
        preferred = "CLEAR_VIEW_B" if initial.table_xy[1] >= 0.0 else "CLEAR_VIEW_A"
        selection_reason = "opposite_last_visible_block_side"
    else:
        q_now = env.data.qpos[:5].copy()
        preferred = min(
            poses, key=lambda item: float(np.linalg.norm(item[1] - q_now))
        )[0]
        selection_reason = "shorter_joint_path_without_visible_block"
    poses.sort(key=lambda item: item[0] != preferred)

    observations = []
    reports = []
    for pose_name, pose in poses:
        location, candidate = _observe_clear_view(
            env, pose_name, pose, matrix, frames
        )
        observations.append(location)
        reports.append(candidate)
        if candidate["sufficient"]:
            break

    selected_index = int(np.argmax([item["score"] for item in reports]))
    selected = observations[selected_index]
    return selected, {
        "selection_reason": selection_reason,
        "preferred_pose": preferred,
        "selected_pose": reports[selected_index]["pose"],
        "alternate_pose_used": len(reports) > 1,
        "clear_view_sufficient": reports[selected_index]["sufficient"],
        "candidates": reports,
    }


def locate_overhead_target(
    frame, matrix=SIM_OVERHEAD_TARGET_PIXEL_TO_TABLE
):
    """Locate the green placement zone from the fixed overhead camera."""
    _, green_mask = color_masks(frame)
    green_mask, pixels, _ = _select_foreground_component(
        green_mask, OVERHEAD_REACQUIRE_MIN_PIXELS
    )
    if pixels < OVERHEAD_REACQUIRE_MIN_PIXELS:
        return OverheadTargetLocation(False, None, None, pixels)
    ys, xs = np.nonzero(green_mask)
    centroid = (float(xs.mean()), float(ys.mean()))
    return OverheadTargetLocation(
        True,
        centroid,
        pixel_to_table(centroid, matrix),
        pixels,
    )


def _refine_overhead_after_wrist_alignment(env, result, matrix):
    """Prefer a less-occluded overhead observation after the arm has moved."""
    refined = locate_overhead_block(env.render_overhead(), matrix)
    result["overhead_refined"] = False
    result["initial_estimated_table_xy"] = result["estimated_table_xy"]
    if (
        refined.visible
        and refined.reachable
        and refined.pixels > result["overhead_pixels"]
    ):
        result["overhead_refined"] = True
        result["overhead_pixels"] = refined.pixels
        result["overhead_pixel"] = refined.pixel
        result["estimated_table_xy"] = refined.table_xy
        result["overhead_orientation_rad"] = refined.orientation_rad
        result["overhead_aspect_ratio"] = refined.aspect_ratio
        result["overhead_pose_class"] = refined.pose_class
        result["overhead_pose_confidence"] = refined.pose_confidence


def reacquire_block(
    env,
    frames=None,
    matrix=SIM_OVERHEAD_PIXEL_TO_TABLE,
    overhead_only=False,
):
    """Coarsely approach from overhead RGB, then locally aim with wrist RGB.

    Returns a diagnostic dictionary whose ``ready_for_pick`` field is the only
    gate a state machine should use to start a pick attempt.
    """
    if frames is None:
        frames = []
    _, opened_frames = set_gripper(env, GRIPPER_OPEN, duration=0.35)
    frames.extend(opened_frames)
    _, _, retreat_frames = retreat_vertical(env, height=0.20, duration=0.8)
    frames.extend(retreat_frames)
    location, clear_view = select_clear_view_observation(
        env, frames=frames, matrix=matrix
    )
    result = {
        "overhead_visible": location.visible,
        "overhead_pixels": location.pixels,
        "overhead_pixel": location.pixel,
        "estimated_table_xy": location.table_xy,
        "overhead_orientation_rad": location.orientation_rad,
        "overhead_aspect_ratio": location.aspect_ratio,
        "overhead_pose_class": location.pose_class,
        "overhead_pose_confidence": location.pose_confidence,
        "clear_view": clear_view,
        "within_workspace": location.reachable,
        "coarse_ik_error_m": None,
        "wrist_found": False,
        "wrist_centered": False,
        "local_search_attempts": 0,
        "approach_height_m": None,
        "approach_pitch_deg": None,
        "wrist_final_pixel": None,
        "ready_for_pick": False,
        "failure_reason": None,
    }
    if not location.visible:
        result["failure_reason"] = "block_not_visible_overhead"
        return result
    if not location.reachable:
        result["failure_reason"] = "block_outside_workspace"
        return result
    if location.pose_class == "UNKNOWN":
        result["failure_reason"] = "block_pose_unknown_after_clear_views"
        return result
    if overhead_only:
        result["failure_reason"] = "wrist_alignment_skipped_for_recovery"
        return result

    x, y = location.table_xy
    # A free-orientation IK seed keeps the wrist camera looking toward the
    # estimated block throughout the workspace. The pan seed follows the
    # block bearing; acceptance still depends on the actual wrist pixels.
    pan_seed = -float(np.arctan2(y, x)) - 0.08
    camera_seed = np.array([pan_seed, *CAMERA_SEED_REST], dtype=float)
    q_target, ik_error = solve_ik(
        env.model,
        [x, y, CAMERA_APPROACH_HEIGHT],
        camera_seed,
        point_down=False,
        deadline_check=getattr(env, "_skill_deadline_check", None),
    )
    _, camera_frames = move_joints(
        env, q_target, gripper=GRIPPER_OPEN, duration=1.0
    )
    frames.extend(camera_frames)
    result["local_search_attempts"] = 1
    result["coarse_ik_error_m"] = float(ik_error)
    result["approach_height_m"] = CAMERA_APPROACH_HEIGHT
    result["approach_pitch_deg"] = None
    if ik_error <= MAX_COARSE_IK_ERROR_M:
        initial_pixel = locate_color(env.render(), "red_block")
        if (
            initial_pixel is not None
            and max(
                abs(initial_pixel[0] - 320), abs(initial_pixel[1] - 240)
            ) < 25
        ):
            result["wrist_found"] = True
            result["wrist_centered"] = True
            result["wrist_final_pixel"] = initial_pixel
            result["ready_for_pick"] = True
            _refine_overhead_after_wrist_alignment(env, result, matrix)
            return result
        found, centered = aim_at(
            env,
            "red_block",
            frames=frames,
            attempts=1,
            allow_scan=False,
        )
        result["wrist_found"] = bool(found)
        result["wrist_final_pixel"] = locate_color(env.render(), "red_block")
        final_pixel = result["wrist_final_pixel"]
        final_centered = (
            final_pixel is not None
            and max(abs(final_pixel[0] - 320), abs(final_pixel[1] - 240)) < 25
        )
        if centered and final_centered:
            result["wrist_centered"] = True
            result["ready_for_pick"] = True
            _refine_overhead_after_wrist_alignment(env, result, matrix)
            return result

    for attempt, (dx, dy) in enumerate(LOCAL_SEARCH_OFFSETS_M, start=1):
        candidates = []
        q_now = env.data.qpos[:5].copy()
        for height, pitch in APPROACH_CONFIGS:
            q_target, ik_error = solve_ik(
                env.model,
                [x + dx, y + dy, height],
                q_now,
                point_down=True,
                pitch_deg=pitch,
                deadline_check=getattr(env, "_skill_deadline_check", None),
            )
            candidates.append((ik_error, height, pitch, q_target))
        ik_error, height, pitch, q_target = min(
            candidates, key=lambda item: item[0]
        )
        _, approach_frames = move_joints(
            env,
            q_target,
            gripper=GRIPPER_OPEN,
            duration=1.0 if attempt == 1 else 0.45,
        )
        frames.extend(approach_frames)
        result["local_search_attempts"] = attempt + 1
        result["coarse_ik_error_m"] = float(ik_error)
        result["approach_height_m"] = float(height)
        result["approach_pitch_deg"] = int(pitch)
        if ik_error > MAX_COARSE_IK_ERROR_M:
            continue
        found, centered = aim_at(
            env,
            "red_block",
            frames=frames,
            attempts=1,
            allow_scan=False,
        )
        result["wrist_found"] = bool(result["wrist_found"] or found)
        result["wrist_final_pixel"] = locate_color(env.render(), "red_block")
        final_pixel = result["wrist_final_pixel"]
        final_centered = (
            final_pixel is not None
            and max(abs(final_pixel[0] - 320), abs(final_pixel[1] - 240)) < 25
        )
        if centered and final_centered:
            result["wrist_centered"] = True
            result["ready_for_pick"] = True
            _refine_overhead_after_wrist_alignment(env, result, matrix)
            return result

    result["failure_reason"] = (
        "coarse_position_unreachable"
        if result["coarse_ik_error_m"] > MAX_COARSE_IK_ERROR_M
        else "wrist_reacquisition_failed"
    )
    return result


def reacquire_and_pick(
    env,
    frames=None,
    matrix=SIM_OVERHEAD_PIXEL_TO_TABLE,
    pick_xy_offset=(0.0, 0.0),
    pick_center_z=TABLE_BLOCK_CENTER_Z,
    pick_wrist_roll=0.0,
    allow_overhead_pick_fallback=False,
    recovery_overhead_only=False,
    recovery_config_index=None,
):
    """Run SEARCH/ALIGN/PICK and verify grasp from wrist RGB + joints.

    The known table plane and block half-height provide Z. XY comes only from
    the calibrated overhead image. Simulator truth returned by the primitive
    is deliberately ignored when deciding whether transport may start.
    """
    if frames is None:
        frames = []
    report = reacquire_block(
        env,
        frames=frames,
        matrix=matrix,
        overhead_only=recovery_overhead_only,
    )
    if recovery_config_index is not None:
        configs = {
            "UPRIGHT": UPRIGHT_RECOVERY_PICK_CONFIGS,
            "TIPPED": TIPPED_RECOVERY_PICK_CONFIGS,
        }.get(report["overhead_pose_class"], RECOVERY_PICK_CONFIGS)
        config = configs[min(int(recovery_config_index), len(configs) - 1)]
        pick_xy_offset, pick_center_z, pick_wrist_roll = config
    report.update(
        {
            "pick_attempted": False,
            "pick_xy_offset_m": tuple(map(float, pick_xy_offset)),
            "pick_center_z_m": float(pick_center_z),
            "pick_wrist_roll_rad": None,
            "pick_observation_frames": 0,
            "camera_grasp_confirmed": False,
            "grasp_monitor_metrics": None,
            "final_gripper_joint": None,
            "final_wrist_block_visible": False,
            "ready_for_transport": False,
            "overhead_pick_fallback": False,
        }
    )
    if not report["ready_for_pick"]:
        fallback_allowed = (
            allow_overhead_pick_fallback
            and report["overhead_visible"]
            and report["within_workspace"]
            and report["estimated_table_xy"] is not None
            and report["overhead_pose_class"] != "UNKNOWN"
            and (report["wrist_found"] or recovery_overhead_only)
        )
        if not fallback_allowed:
            return report
        report["ready_for_pick"] = True
        report["overhead_pick_fallback"] = True
        report["failure_reason"] = None

    effective_wrist_roll = (
        float(report["overhead_orientation_rad"] + np.pi / 2)
        if pick_wrist_roll is None
        else float(pick_wrist_roll)
    )
    # Wrist roll is periodic over pi for a parallel-jaw gripper.  Keep the
    # command inside the closest half-turn around zero for joint reachability.
    effective_wrist_roll = float(
        (effective_wrist_roll + np.pi / 2) % np.pi - np.pi / 2
    )
    report["pick_wrist_roll_rad"] = effective_wrist_roll

    monitor = VisionPickMonitor()
    observed_steps = 0

    def observe_lift(obs):
        nonlocal observed_steps
        observed_steps += 1
        monitor.update(obs["pixels"], obs["agent_pos"], observed_steps)

    x, y = report["estimated_table_xy"]
    x += float(pick_xy_offset[0])
    y += float(pick_xy_offset[1])
    report["pick_attempted"] = True
    pick_start_pose = PICK_START_POSE.copy()
    pick_start_pose[4] = effective_wrist_roll
    _, reset_frames = move_joints(
        env,
        pick_start_pose,
        gripper=GRIPPER_OPEN,
        duration=0.8,
    )
    frames.extend(reset_frames)
    pick(
        env,
        [x, y, float(pick_center_z)],
        frames=frames,
        on_lift_observation=observe_lift,
    )
    if recovery_config_index is not None:
        # Recovered blocks are often tipped and sit less deeply in the jaws.
        # Re-assert closure and let the contact settle before transport.
        _, secure_frames = set_gripper(
            env,
            -0.1,
            duration=0.25,
            on_observation=observe_lift,
        )
        frames.extend(secure_frames)
        _, settle_frames = hold_position(
            env,
            duration=0.35,
            on_observation=observe_lift,
        )
        frames.extend(settle_frames)
    final_obs = env._get_obs()
    final_gripper = float(final_obs["agent_pos"][-1])
    final_visible = observe_colors(final_obs["pixels"]).red_block.visible
    final_holding = (
        GRIPPER_HOLDING_MIN <= final_gripper <= GRIPPER_HOLDING_MAX
        and final_visible
    )
    confirmed = bool(monitor.grasp_confirmed and final_holding)
    report["pick_observation_frames"] = observed_steps
    report["final_gripper_joint"] = final_gripper
    report["final_wrist_block_visible"] = bool(final_visible)
    report["camera_grasp_confirmed"] = confirmed
    report["grasp_monitor_metrics"] = dict(monitor.last_holding_metrics)
    report["ready_for_transport"] = confirmed
    report["failure_reason"] = (
        None if confirmed else "camera_grasp_not_confirmed"
    )
    return report


def pick_until_verified(
    env,
    frames=None,
    matrix=SIM_OVERHEAD_PIXEL_TO_TABLE,
    max_attempts=6,
    recovery=False,
):
    """Repeat SEARCH/ALIGN/PICK until camera verification or a safety stop."""
    if frames is None:
        frames = []
    attempts = []
    non_retriable = {
        "block_not_visible_overhead",
        "block_outside_workspace",
        "block_pose_unknown_after_clear_views",
    }
    configs = (
        tuple((None, None, None) for _ in range(max_attempts))
        if recovery
        else tuple(
            (offset, TABLE_BLOCK_CENTER_Z, 0.0)
            for offset in PICK_RETRY_OFFSETS_M
        )
    )
    for attempt_index, (offset, center_z, wrist_roll) in enumerate(
        configs[:max_attempts], start=1
    ):
        report = reacquire_and_pick(
            env,
            frames=frames,
            matrix=matrix,
            pick_xy_offset=offset,
            pick_center_z=center_z,
            pick_wrist_roll=wrist_roll,
            allow_overhead_pick_fallback=recovery,
            recovery_overhead_only=recovery,
            recovery_config_index=(attempt_index - 1) if recovery else None,
        )
        report["pick_attempt"] = attempt_index
        attempts.append(report)
        if report["ready_for_transport"]:
            return {
                "success": True,
                "attempts": attempt_index,
                "attempt_reports": attempts,
                "failure_reason": None,
            }
        if report["failure_reason"] in non_retriable:
            break
    return {
        "success": False,
        "attempts": len(attempts),
        "attempt_reports": attempts,
        "failure_reason": (
            attempts[-1]["failure_reason"] if attempts else "no_pick_attempt"
        ),
    }
