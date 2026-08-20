"""Fail-closed bridge from the MuJoCo policy interface to a physical SO-101.

The adapter intentionally keeps observation/action conversion separate from
task policy.  ``shadow`` mode computes the exact action that active mode would
send, including all clamps, but never calls ``robot.send_action``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Protocol

import cv2
import mujoco
import numpy as np

from hardware.camera_roles import BACKEND_IDS, CAMERA_ROLES, load_camera_registry
from hardware.joint_safety import (
    SHOULDER_PAN_KEY,
    SafetyLimits,
    assert_position_within_limits,
    enforce_action_limits,
)


JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
JOINT_KEYS = tuple(f"{name}.pos" for name in JOINT_NAMES)
VALID_MODES = ("shadow", "active")
DEFAULT_KINEMATICS_SCENE = (
    Path(__file__).resolve().parents[1]
    / "assets/menagerie_so101/pick_scene_real_block.xml"
)


def _load_mj_model(xml_path: str | Path):
    """Load kinematics MJCF even when its Windows path contains non-ASCII."""
    path = Path(xml_path).resolve()
    try:
        return mujoco.MjModel.from_xml_path(str(path))
    except ValueError as exc:
        if "Error opening file" not in str(exc):
            raise
        assets = {
            asset.relative_to(path.parent).as_posix(): asset.read_bytes()
            for asset in path.parent.rglob("*")
            if asset.is_file()
        }
        return mujoco.MjModel.from_xml_string(
            path.read_text(encoding="utf-8"), assets=assets
        )


class CameraSource(Protocol):
    def connect(self) -> None: ...
    def read_rgb(self, role: str) -> np.ndarray: ...
    def disconnect(self) -> None: ...


@dataclass(frozen=True)
class JointMapping:
    robot_units_per_policy_rad: float
    robot_zero: float
    verified_on_physical_robot: bool

    def policy_to_robot(self, value_rad: float) -> float:
        return self.robot_zero + self.robot_units_per_policy_rad * float(value_rad)

    def robot_to_policy(self, value: float) -> float:
        return (float(value) - self.robot_zero) / self.robot_units_per_policy_rad


@dataclass(frozen=True)
class AdapterConfig:
    source_path: Path
    port: str
    robot_id: str
    use_degrees: bool
    max_relative_target: float
    calibration_dir: Path | None
    checkpoint: Path
    image_key: str
    control_hz: float
    n_action_steps: int
    image_width: int
    image_height: int
    image_transform: str
    joint_mappings: dict[str, JointMapping]
    require_verified_mapping: bool
    require_preflight: bool
    confirmation_token: str

    @property
    def mapping_verified(self) -> bool:
        return all(
            self.joint_mappings[name].verified_on_physical_robot
            for name in JOINT_NAMES
        )


def _finite_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number")
    return result


def load_adapter_config(path: str | Path) -> AdapterConfig:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("format_version") != 1:
        raise ValueError("Unsupported real-adapter config format_version")
    robot = payload.get("robot")
    policy = payload.get("policy")
    image = payload.get("policy_image")
    mappings = payload.get("joint_mapping")
    active = payload.get("active_mode")
    if not all(isinstance(item, dict) for item in (robot, policy, image, mappings, active)):
        raise ValueError("Adapter config is missing a required object")
    if policy.get("joint_order") != list(JOINT_NAMES):
        raise ValueError(f"policy.joint_order must equal {list(JOINT_NAMES)!r}")
    if robot.get("use_degrees") is not True:
        raise ValueError("SO-101 arm mapping requires robot.use_degrees=true")
    joint_mappings: dict[str, JointMapping] = {}
    for name in JOINT_NAMES:
        entry = mappings.get(name)
        if not isinstance(entry, dict):
            raise ValueError(f"joint_mapping.{name} is missing")
        scale = _finite_number(
            entry.get("robot_units_per_policy_rad"),
            f"joint_mapping.{name}.robot_units_per_policy_rad",
        )
        if abs(scale) < 1e-9:
            raise ValueError(f"joint_mapping.{name} scale cannot be zero")
        verified = entry.get("verified_on_physical_robot")
        if not isinstance(verified, bool):
            raise ValueError(
                f"joint_mapping.{name}.verified_on_physical_robot must be boolean"
            )
        joint_mappings[name] = JointMapping(
            scale,
            _finite_number(entry.get("robot_zero"), f"joint_mapping.{name}.robot_zero"),
            verified,
        )
    calibration_dir = robot.get("calibration_dir")
    checkpoint = Path(policy.get("checkpoint", ""))
    if not checkpoint.is_absolute():
        checkpoint = Path.cwd() / checkpoint
    return AdapterConfig(
        source.resolve(),
        str(robot.get("port", "")),
        str(robot.get("id", "")),
        True,
        _finite_number(robot.get("max_relative_target"), "robot.max_relative_target"),
        Path(calibration_dir).expanduser() if calibration_dir else None,
        checkpoint.resolve(),
        str(policy.get("image_key", "")),
        _finite_number(policy.get("control_hz"), "policy.control_hz"),
        int(policy.get("n_action_steps", 0)),
        int(image.get("width", 0)),
        int(image.get("height", 0)),
        str(image.get("transform", "")),
        joint_mappings,
        active.get("require_all_joint_mappings_verified") is True,
        active.get("require_motion_authorized_preflight") is True,
        str(active.get("confirmation_token", "")),
    )


def validate_adapter_config(config: AdapterConfig) -> None:
    if not config.port:
        raise ValueError("robot.port is required")
    if not config.robot_id:
        raise ValueError("robot.id is required")
    if config.max_relative_target <= 0:
        raise ValueError("robot.max_relative_target must be positive")
    if config.control_hz <= 0:
        raise ValueError("policy.control_hz must be positive")
    if config.n_action_steps < 1:
        raise ValueError("policy.n_action_steps must be at least 1")
    if config.image_key != "observation.images.wrist":
        raise ValueError("ACT v2.2 requires policy.image_key=observation.images.wrist")
    if config.image_width != 640 or config.image_height != 480:
        raise ValueError("ACT v2.2 requires a 640x480 policy image")
    if config.image_transform != "center_crop_resize":
        raise ValueError("Only center_crop_resize is currently validated")
    if not config.confirmation_token:
        raise ValueError("active_mode.confirmation_token is required")


def prepare_policy_image(rgb: np.ndarray, width: int = 640, height: int = 480) -> np.ndarray:
    """Center-crop without aspect distortion, then resize to ACT's 640x480."""
    frame = np.asarray(rgb)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("RGB frame must have shape HxWx3")
    source_h, source_w = frame.shape[:2]
    target_ratio = width / height
    source_ratio = source_w / source_h
    if source_ratio > target_ratio:
        crop_w = max(1, int(round(source_h * target_ratio)))
        left = (source_w - crop_w) // 2
        frame = frame[:, left:left + crop_w]
    elif source_ratio < target_ratio:
        crop_h = max(1, int(round(source_w / target_ratio)))
        top = (source_h - crop_h) // 2
        frame = frame[top:top + crop_h, :]
    resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(resized)


def robot_state_to_policy(
    observation: Mapping[str, Any], config: AdapterConfig
) -> np.ndarray:
    missing = [key for key in JOINT_KEYS if key not in observation]
    if missing:
        raise KeyError(f"Robot observation is missing joints: {missing}")
    values = [
        config.joint_mappings[name].robot_to_policy(observation[f"{name}.pos"])
        for name in JOINT_NAMES
    ]
    state = np.asarray(values, dtype=np.float64)
    if not np.isfinite(state).all():
        raise ValueError("Robot observation contains non-finite joint values")
    return state


def policy_action_to_robot(action: np.ndarray, config: AdapterConfig) -> dict[str, float]:
    values = np.asarray(action, dtype=np.float64).reshape(-1)
    if values.shape != (len(JOINT_NAMES),) or not np.isfinite(values).all():
        raise ValueError("Policy action must contain six finite joint targets")
    return {
        f"{name}.pos": config.joint_mappings[name].policy_to_robot(values[index])
        for index, name in enumerate(JOINT_NAMES)
    }


def enforce_relative_targets(
    requested: Mapping[str, float],
    current: Mapping[str, float],
    max_delta: float,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    safe = {key: float(value) for key, value in requested.items()}
    interventions = []
    for key, requested_value in safe.items():
        if key not in current:
            raise KeyError(f"Current robot observation is missing {key}")
        present = float(current[key])
        applied = float(np.clip(requested_value, present - max_delta, present + max_delta))
        safe[key] = applied
        if not math.isclose(applied, requested_value, abs_tol=1e-12):
            interventions.append(
                {
                    "joint": key.removesuffix(".pos"),
                    "kind": "relative_target",
                    "present": present,
                    "requested": requested_value,
                    "applied": applied,
                    "max_delta": max_delta,
                }
            )
    return safe, interventions


class OpenCVDualCameraSource:
    """Persistent role-safe camera session; frames are returned as RGB."""

    def __init__(self, camera_config: str | Path, warmup_frames: int = 8):
        self.registry = load_camera_registry(camera_config)
        self.warmup_frames = int(warmup_frames)
        self._captures: dict[str, cv2.VideoCapture] = {}

    def connect(self) -> None:
        if self._captures:
            raise RuntimeError("Camera source is already connected")
        try:
            for role in CAMERA_ROLES:
                entry = self.registry["roles"].get(role)
                if entry is None:
                    raise RuntimeError(f"Camera role {role} is not registered")
                backend = entry.get("backend", "dshow")
                capture = cv2.VideoCapture(int(entry["index"]), BACKEND_IDS[backend])
                if not capture.isOpened():
                    capture.release()
                    raise RuntimeError(f"Could not open {role} camera index {entry['index']}")
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(entry.get("width", 1280)))
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(entry.get("height", 720)))
                self._captures[role] = capture
                for _ in range(max(1, self.warmup_frames)):
                    ok, _ = capture.read()
                    if not ok:
                        raise RuntimeError(f"Could not read warmup frame from {role} camera")
        except Exception:
            self.disconnect()
            raise

    def read_rgb(self, role: str) -> np.ndarray:
        if role not in self._captures:
            raise RuntimeError(f"Camera role {role} is not connected")
        ok, bgr = self._captures[role].read()
        if not ok or bgr is None:
            raise RuntimeError(f"Failed to read current frame from {role} camera")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def disconnect(self) -> None:
        for capture in self._captures.values():
            capture.release()
        self._captures.clear()


def create_lerobot_follower(config: AdapterConfig):
    """Create the installed LeRobot SO-101 follower without opening cameras."""
    from lerobot.robots.so_follower import SO101Follower
    from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig

    robot_config = SO101FollowerConfig(
        port=config.port,
        id=config.robot_id,
        use_degrees=config.use_degrees,
        max_relative_target=config.max_relative_target,
        cameras={},
        calibration_dir=config.calibration_dir,
    )
    return SO101Follower(robot_config)


class RealSO101Adapter:
    """MuJoCo-like environment backed by physical observations and joint targets."""

    metadata = {"render_fps": 25}

    def __init__(
        self,
        config: AdapterConfig,
        *,
        mode: str,
        robot: Any,
        cameras: CameraSource,
        safety_limits: SafetyLimits,
        preflight_report: Mapping[str, Any] | None = None,
        active_confirmation: str | None = None,
        audit_path: str | Path | None = None,
        kinematics_scene: str | Path = DEFAULT_KINEMATICS_SCENE,
    ):
        validate_adapter_config(config)
        if mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}")
        self.config = config
        self.mode = mode
        self.robot = robot
        self.cameras = cameras
        self.safety_limits = safety_limits
        self.preflight_report = dict(preflight_report or {})
        self.active_confirmation = active_confirmation
        self.audit_path = Path(audit_path) if audit_path else None
        self.model = _load_mj_model(kinematics_scene)
        self.data = mujoco.MjData(self.model)
        self.metadata = {"render_fps": int(round(config.control_hz))}
        self._connected = False
        self._shadow_bus_only = False
        self._latest_wrist: np.ndarray | None = None
        self._latest_overhead: np.ndarray | None = None
        self._latest_robot_observation: dict[str, float] | None = None
        self._latest_policy_state: np.ndarray | None = None
        self._last_policy_command = np.zeros(len(JOINT_NAMES), dtype=float)
        self._step_index = 0
        self.goal_position_writes = 0
        self.audit_records: list[dict[str, Any]] = []
        self._assert_mode_gate()

    def _assert_mode_gate(self) -> None:
        if self.mode != "active":
            return
        if self.config.require_preflight and not self.preflight_report.get("motion_authorized"):
            raise RuntimeError("Active mode blocked: real-workspace preflight is not authorized")
        if self.config.require_verified_mapping and not self.config.mapping_verified:
            missing = [
                name for name in JOINT_NAMES
                if not self.config.joint_mappings[name].verified_on_physical_robot
            ]
            raise RuntimeError(f"Active mode blocked: unverified joint mappings {missing}")
        if self.active_confirmation != self.config.confirmation_token:
            raise RuntimeError("Active mode blocked: explicit confirmation token is missing")

    def connect(self) -> None:
        if self._connected:
            raise RuntimeError("Adapter is already connected")
        self._assert_mode_gate()
        if self.mode == "shadow" and all(
            hasattr(self.robot.bus, name) for name in ("connect", "disconnect")
        ):
            # SO101Follower.connect() configures motor registers. Shadow mode
            # needs only calibrated Present_Position reads, so open the bus
            # directly and do not enable/configure torque or Goal_Position.
            self.robot.bus.connect()
            self._shadow_bus_only = True
        else:
            self.robot.connect(calibrate=False)
        try:
            self.cameras.connect()
            self._connected = True
            self._refresh_observation()
            shoulder = self._latest_robot_observation[SHOULDER_PAN_KEY]
            assert_position_within_limits(shoulder, self.safety_limits)
        except Exception:
            self.cameras.disconnect()
            self._disconnect_robot()
            self._connected = False
            raise

    def _disconnect_robot(self) -> None:
        if self._shadow_bus_only:
            # False means do not write Torque_Enable during a read-only shadow
            # disconnect. Active mode keeps LeRobot's normal safe disconnect.
            self.robot.bus.disconnect(False)
            self._shadow_bus_only = False
        else:
            self.robot.disconnect()

    def disconnect(self) -> None:
        if not self._connected:
            return
        try:
            self.cameras.disconnect()
        finally:
            self._disconnect_robot()
            self._connected = False

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.disconnect()

    def _require_connected(self) -> None:
        if not self._connected:
            raise RuntimeError("Real SO-101 adapter is not connected")

    def _refresh_observation(self) -> dict[str, np.ndarray]:
        observation = self.robot.get_observation()
        robot_values = {key: float(observation[key]) for key in JOINT_KEYS}
        policy_state = robot_state_to_policy(robot_values, self.config)
        wrist_raw = self.cameras.read_rgb("wrist")
        overhead = self.cameras.read_rgb("overhead")
        wrist = prepare_policy_image(
            wrist_raw, self.config.image_width, self.config.image_height
        )
        self._latest_robot_observation = robot_values
        self._latest_policy_state = policy_state
        self._latest_wrist = wrist
        self._latest_overhead = np.ascontiguousarray(overhead)
        self.data.qpos[:len(JOINT_NAMES)] = policy_state
        self.data.ctrl[:len(JOINT_NAMES)] = self._last_policy_command
        mujoco.mj_forward(self.model, self.data)
        return {"pixels": wrist, "agent_pos": policy_state.copy()}

    def _get_obs(self) -> dict[str, np.ndarray]:
        self._require_connected()
        return self._refresh_observation()

    def render(self) -> np.ndarray:
        self._require_connected()
        if self._latest_wrist is None:
            self._refresh_observation()
        return self._latest_wrist.copy()

    def render_overhead(self) -> np.ndarray:
        self._require_connected()
        if self._latest_overhead is None:
            self._refresh_observation()
        return self._latest_overhead.copy()

    def _get_info(self) -> dict[str, Any]:
        return {
            "backend": "physical_so101",
            "mode": self.mode,
            "privileged_object_state_available": False,
            "block_height": float("nan"),
            "success": False,
            "goal_position_writes": self.goal_position_writes,
        }

    def _record(self, payload: dict[str, Any]) -> None:
        self.audit_records.append(payload)
        if self.audit_path is not None:
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)
            with self.audit_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def step(self, action):
        self._require_connected()
        policy_action = np.asarray(action, dtype=np.float64).reshape(-1)
        if policy_action.shape != (len(JOINT_NAMES),) or not np.isfinite(policy_action).all():
            raise ValueError("Physical adapter action must contain six finite targets")
        if self._latest_robot_observation is None:
            self._refresh_observation()
        requested = policy_action_to_robot(policy_action, self.config)
        relative_safe, relative_interventions = enforce_relative_targets(
            requested,
            self._latest_robot_observation,
            self.config.max_relative_target,
        )
        absolute_safe, absolute_interventions = enforce_action_limits(
            relative_safe, self.safety_limits
        )
        sent_action = None
        if self.mode == "active":
            self._assert_mode_gate()
            current_pan = float(self.robot.bus.sync_read("Present_Position")["shoulder_pan"])
            assert_position_within_limits(current_pan, self.safety_limits)
            sent_action = {
                key: float(value)
                for key, value in self.robot.send_action(absolute_safe).items()
            }
            self.goal_position_writes += 1
        self._last_policy_command = policy_action.copy()
        self._step_index += 1
        record = {
            "format_version": 1,
            "timestamp": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "step": self._step_index,
            "mode": self.mode,
            "joint_mapping_verified": self.config.mapping_verified,
            "current_policy_state_rad": self._latest_policy_state.tolist(),
            "current_robot_observation": dict(self._latest_robot_observation),
            "policy_action_rad": policy_action.tolist(),
            "requested_robot_action": requested,
            "applied_robot_action": absolute_safe,
            "applied_delta_from_current": {
                key: absolute_safe[key] - self._latest_robot_observation[key]
                for key in JOINT_KEYS
            },
            "relative_interventions": relative_interventions,
            "absolute_interventions": absolute_interventions,
            "send_action_called": self.mode == "active",
            "sent_action": sent_action,
            "goal_position_writes_total": self.goal_position_writes,
        }
        self._record(record)
        if self.mode == "active":
            time.sleep(1.0 / self.config.control_hz)
        obs = self._refresh_observation()
        return obs, 0.0, False, False, self._get_info()


class ACTPolicySession:
    """Load ACT v2.2 and produce one policy-radian target at a time."""

    def __init__(self, config: AdapterConfig, device: str = "cpu"):
        import torch
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.factory import make_pre_post_processors

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if not config.checkpoint.is_dir():
            raise FileNotFoundError(f"ACT checkpoint not found: {config.checkpoint}")
        self.torch = torch
        self.config = config
        self.policy = ACTPolicy.from_pretrained(str(config.checkpoint))
        self.policy.to(device).eval()
        self.policy.config.device = device
        self.policy.config.n_action_steps = config.n_action_steps
        self.pre, self.post = make_pre_post_processors(
            policy_cfg=self.policy.config,
            pretrained_path=str(config.checkpoint),
            preprocessor_overrides={"device_processor": {"device": device}},
        )
        self.reset()

    def reset(self) -> None:
        self.policy.reset()

    def select_action(self, observation: Mapping[str, np.ndarray]) -> np.ndarray:
        image = np.asarray(observation["pixels"])
        state = np.asarray(observation["agent_pos"], dtype=np.float32)
        batch = {
            self.config.image_key: (
                self.torch.from_numpy(image)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .float()
                / 255.0
            ),
            "observation.state": self.torch.from_numpy(state).unsqueeze(0),
        }
        batch = self.pre(batch)
        with self.torch.inference_mode():
            action = self.policy.select_action(batch)
        action = self.post(action)
        return np.asarray(action.squeeze(0).cpu(), dtype=np.float64)
