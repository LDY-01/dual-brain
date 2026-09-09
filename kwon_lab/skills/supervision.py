"""State-based guards shared by rollout supervision and DAgger collection."""

import numpy as np


LIFTED_HEIGHT = 0.08
MIN_DECISION_STEP = 75
FALLBACK_NO_LIFT_STEP = 240
GRIPPER_BLOCK_FAR_M = 0.16
GRIPPER_TARGET_NEAR_M = 0.14


def repick_reason(info, step, lifted_once, fallback_step=FALLBACK_NO_LIFT_STEP):
    """Return why a pick retry is needed, or None while ACT may continue.

    A missed transport is detected from physical state rather than a short
    fixed timeout: the block is still on the table while the gripper has left
    it and moved into the target region. The late fallback catches policies
    that never leave either region.
    """
    if lifted_once or float(info["block_height"]) > LIFTED_HEIGHT:
        return None
    gripper = np.asarray(info["gripper_pos"], dtype=float)
    block = np.asarray(info["block_pos"], dtype=float)
    target = np.asarray(info["target_pos"], dtype=float)
    gripper_block_distance = float(np.linalg.norm(gripper - block))
    gripper_target_xy_distance = float(np.linalg.norm(gripper[:2] - target[:2]))
    if (
        step >= MIN_DECISION_STEP
        and gripper_block_distance >= GRIPPER_BLOCK_FAR_M
        and gripper_target_xy_distance <= GRIPPER_TARGET_NEAR_M
    ):
        return "transport_without_block"
    if step >= fallback_step:
        return "not_lifted_by_deadline"
    return None
