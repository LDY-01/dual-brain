"""Camera-only task observations for sim-to-real supervision.

This module deliberately accepts only RGB pixels and robot joint state. It must
not import MuJoCo or consume privileged object poses.
"""

from dataclasses import dataclass
from collections import deque

import numpy as np


MIN_COLOR_PIXELS = 150
MIN_PICK_DECISION_STEP = 75
MISSED_PICK_CONFIRM_FRAMES = 8
GRIPPER_OPEN_THRESHOLD = 1.25
GRIPPER_EMPTY_CLOSED_THRESHOLD = 0.12
GRIPPER_HOLDING_MIN = 0.18
GRIPPER_HOLDING_MAX = 0.65
GRASP_CONFIRM_FRAMES = 12
GRIPPER_STABLE_RANGE = 0.06
MIN_ARM_MOTION_FOR_GRASP = 0.08
MAX_HELD_CENTROID_SPREAD_PX = 70.0
MAX_HELD_AREA_RATIO = 2.5
DROP_CONFIRM_FRAMES = 15
PLACE_CONFIRM_FRAMES = 8
OVERHEAD_TABLE_AREA_MIN_RATIO = 0.65
OVERHEAD_TABLE_AREA_MAX_RATIO = 1.25
BLOCK_TO_TARGET_AREA_RATIO = (0.04 * 0.04) / (np.pi * 0.05 * 0.05)
OVERHEAD_DROP_MAX_CENTROID_SPREAD_PX = 12.0
OVERHEAD_DROP_MIN_ARM_MOTION = 0.08


@dataclass(frozen=True)
class ColorObservation:
    visible: bool
    pixels: int
    centroid: tuple[float, float] | None
    bbox: tuple[int, int, int, int] | None


@dataclass(frozen=True)
class VisionObservation:
    red_block: ColorObservation
    green_target: ColorObservation


def _summarize_mask(mask: np.ndarray) -> ColorObservation:
    count = int(mask.sum())
    if count < MIN_COLOR_PIXELS:
        return ColorObservation(False, count, None, None)
    ys, xs = np.nonzero(mask)
    return ColorObservation(
        True,
        count,
        (float(xs.mean()), float(ys.mean())),
        (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
    )


def observe_colors(frame: np.ndarray) -> VisionObservation:
    """Detect the current red block and green target from an RGB frame."""
    red, green = color_masks(frame)
    return VisionObservation(_summarize_mask(red), _summarize_mask(green))


def color_masks(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return red-block and green-target masks from an RGB image."""
    rgb = np.asarray(frame, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB frame, got {rgb.shape}")
    r, g, b = (rgb[:, :, i].astype(np.int16) for i in range(3))
    red = (r > 100) & (r > 4 * g) & (r > 4 * b)
    green = (g > 100) & (g > 2 * r) & (g > 2 * b)
    return red, green


class OverheadTaskMonitor:
    """Fixed-camera drop/place monitor using RGB pixels and gripper state only.

    The target mask and the block's table-plane pixel area are learned from the
    first unobstructed overhead image. No simulator object coordinates are used.
    """

    def __init__(self, calibration_frame: np.ndarray):
        red, green = color_masks(calibration_frame)
        if int(green.sum()) < MIN_COLOR_PIXELS:
            raise ValueError("Cannot calibrate overhead monitor: green target not visible")
        self.target_mask = green
        # The home pose can hide most of the block from a truly overhead camera.
        # Infer its table-plane footprint from the visible 10 cm target and the
        # known 4x4 cm block instead of trusting the occluded initial red mask.
        self.table_block_area = int(round(green.sum() * BLOCK_TO_TARGET_AREA_RATIO))
        self.drop_evidence = 0
        self.place_evidence = 0
        self.possible_drop = False
        self.possible_place = False
        self.last_coverage = 0.0
        self.last_table_area_ratio = 0.0
        self.drop_history = deque(maxlen=DROP_CONFIRM_FRAMES)

    def update(self, frame: np.ndarray, joint_state, grasp_confirmed: bool):
        red, green = color_masks(frame)
        observation = VisionObservation(_summarize_mask(red), _summarize_mask(green))
        red_area = int(red.sum())
        self.last_table_area_ratio = red_area / max(1, self.table_block_area)
        table_like = (
            observation.red_block.visible
            and OVERHEAD_TABLE_AREA_MIN_RATIO
            <= self.last_table_area_ratio
            <= OVERHEAD_TABLE_AREA_MAX_RATIO
        )
        self.last_coverage = (
            float(np.count_nonzero(red & self.target_mask)) / red_area
            if red_area else 0.0
        )
        gripper = float(np.asarray(joint_state)[-1])
        arm = np.asarray(joint_state, dtype=float)[:5].copy()

        place_like = (
            grasp_confirmed
            and self.last_coverage >= 0.75
            and gripper >= GRIPPER_OPEN_THRESHOLD
        )
        self.place_evidence = self.place_evidence + 1 if place_like else 0
        self.possible_place = self.place_evidence >= PLACE_CONFIRM_FRAMES

        self.drop_history.append((observation.red_block.centroid, arm))
        stable_dropped_block = False
        if len(self.drop_history) == DROP_CONFIRM_FRAMES and all(
            centroid is not None for centroid, _ in self.drop_history
        ):
            centroids = np.asarray([item[0] for item in self.drop_history], dtype=float)
            centroid_spread = float(
                np.linalg.norm(centroids - centroids.mean(axis=0), axis=1).max()
            )
            arm_motion = float(
                np.linalg.norm(self.drop_history[-1][1] - self.drop_history[0][1])
            )
            stable_dropped_block = (
                centroid_spread <= OVERHEAD_DROP_MAX_CENTROID_SPREAD_PX
                and arm_motion >= OVERHEAD_DROP_MIN_ARM_MOTION
            )
        drop_like = (
            grasp_confirmed
            and stable_dropped_block
            and self.last_coverage < 0.75
            and gripper <= GRIPPER_HOLDING_MAX
        )
        self.drop_evidence = self.drop_evidence + 1 if drop_like else 0
        self.possible_drop = self.drop_evidence >= DROP_CONFIRM_FRAMES
        return observation, {
            "possible_drop": self.possible_drop,
            "possible_place": self.possible_place,
            "image_target_coverage": self.last_coverage,
            "table_area_ratio": self.last_table_area_ratio,
        }


def placement_candidate(observation: VisionObservation) -> bool:
    """Approximate block-in-target from visible red/green image regions.

    This is intentionally a shadow estimate, not a deployment success signal.
    The red block occludes part of the green target, so exact 75% world-space
    coverage cannot be recovered from one moving wrist RGB frame.
    """
    red, green = observation.red_block, observation.green_target
    if not (red.visible and green.visible):
        return False
    gx0, gy0, gx1, gy1 = green.bbox
    target_center = np.array([(gx0 + gx1) / 2, (gy0 + gy1) / 2], dtype=float)
    target_radius = max(gx1 - gx0, gy1 - gy0) / 2
    red_center = np.asarray(red.centroid, dtype=float)
    return bool(np.linalg.norm(red_center - target_center) <= 0.75 * target_radius)


class VisionTaskShadow:
    """Camera-only shadow events for pick, possible drop, and possible place."""

    def __init__(self):
        self.pick = VisionPickMonitor()
        self.drop_evidence = 0
        self.place_evidence = 0
        self.possible_drop = False
        self.possible_place = False

    def update(self, frame: np.ndarray, joint_state, step: int):
        observation, pick_event = self.pick.update(frame, joint_state, step)
        gripper = float(np.asarray(joint_state)[-1])
        if self.pick.grasp_confirmed:
            drop_like = (
                not observation.red_block.visible
                and not observation.green_target.visible
                and gripper <= GRIPPER_HOLDING_MAX
            )
            self.drop_evidence = self.drop_evidence + 1 if drop_like else 0
            self.possible_drop = self.drop_evidence >= DROP_CONFIRM_FRAMES
        place_like = placement_candidate(observation) and gripper >= GRIPPER_OPEN_THRESHOLD
        self.place_evidence = self.place_evidence + 1 if place_like else 0
        self.possible_place = self.place_evidence >= PLACE_CONFIRM_FRAMES
        return observation, {
            "pick_event": pick_event,
            "grasp_confirmed": self.pick.grasp_confirmed,
            "possible_drop": self.possible_drop,
            "possible_place": self.possible_place,
        }


class VisionPickMonitor:
    """Infer grasp evidence and empty transport using pixels + gripper state."""

    def __init__(self):
        self.red_missing_frames = 0
        self.grasp_evidence_frames = 0
        self.grasp_confirmed = False
        self.last_event = None
        self.holding_history = deque(maxlen=GRASP_CONFIRM_FRAMES)

    def reset(self):
        self.__init__()

    def update(self, frame: np.ndarray, joint_state, step: int):
        observation = observe_colors(frame)
        gripper = float(np.asarray(joint_state)[-1])

        if observation.red_block.visible:
            self.red_missing_frames = 0
        else:
            self.red_missing_frames += 1

        centroid = observation.red_block.centroid
        self.holding_history.append((
            gripper,
            observation.red_block.visible,
            centroid,
            observation.red_block.pixels,
            np.asarray(joint_state, dtype=float)[:5].copy(),
        ))
        holding_values = [value for value, visible, _, _, _ in self.holding_history if visible]
        visible_history = len(holding_values) == GRASP_CONFIRM_FRAMES
        if visible_history:
            centroids = np.asarray([item[2] for item in self.holding_history], dtype=float)
            areas = np.asarray([item[3] for item in self.holding_history], dtype=float)
            arm_start = self.holding_history[0][4]
            arm_end = self.holding_history[-1][4]
            centroid_spread = float(np.linalg.norm(centroids - centroids.mean(axis=0), axis=1).max())
            area_ratio = float(areas.max() / max(1.0, areas.min()))
            arm_motion = float(np.linalg.norm(arm_end - arm_start))
        else:
            centroid_spread = area_ratio = arm_motion = 0.0
        stable_holding = (
            visible_history
            and all(GRIPPER_HOLDING_MIN <= value <= GRIPPER_HOLDING_MAX for value in holding_values)
            and max(holding_values) - min(holding_values) <= GRIPPER_STABLE_RANGE
            and arm_motion >= MIN_ARM_MOTION_FOR_GRASP
            and centroid_spread <= MAX_HELD_CENTROID_SPREAD_PX
            and area_ratio <= MAX_HELD_AREA_RATIO
        )
        self.grasp_evidence_frames = len(holding_values) if stable_holding else 0
        if stable_holding:
            self.grasp_confirmed = True

        event = None
        if (
            not self.grasp_confirmed
            and step >= MIN_PICK_DECISION_STEP
            and (
                gripper >= GRIPPER_OPEN_THRESHOLD
                or gripper <= GRIPPER_EMPTY_CLOSED_THRESHOLD
            )
            and self.red_missing_frames >= MISSED_PICK_CONFIRM_FRAMES
        ):
            event = "transport_without_block"
        self.last_event = event
        return observation, event
