"""Simulation-only pen-tip kinematics and fail-closed path preflight.

Uses the repository's SO-101 collision model. No hardware backend is imported.
The rigid tool and paper placement are hypotheses, not physical calibration.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone, timedelta
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .planner import PaperSettings, PenStroke, WritingPlan, plan_text


ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
ASSETS = Path(__file__).resolve().parents[1] / "assets" / "menagerie_so101"


def vector(value, size, name):
    if not isinstance(value, (tuple, list, np.ndarray)) or len(value) != size:
        raise ValueError(f"{name} must have {size} numeric elements")
    if any(isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, float, np.number))
           for x in value):
        raise ValueError(f"{name} must contain finite numbers")
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite numbers")
    return result


def positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")


@dataclass(frozen=True)
class ToolMount:
    profile_id: str = "virtual-x-axis-60mm-v1"
    tip_offset_m: tuple = (.060, 0., 0.)
    axis_local: tuple = (1., 0., 0.)  # From barrel toward the nib, in gripperframe.
    shaft_length_m: float = .145
    shaft_radius_m: float = .00625
    nib_length_m: float = .010
    nib_radius_m: float = .0005
    gripper_rad: float = .1
    simulation_only: bool = True
    calibrated: bool = False

    def validate(self):
        if not isinstance(self.profile_id, str) or not self.profile_id.strip():
            raise ValueError("profile_id is required")
        vector(self.tip_offset_m, 3, "tip_offset_m")
        axis = vector(self.axis_local, 3, "axis_local")
        if not np.isclose(np.linalg.norm(axis), 1, atol=1e-7):
            raise ValueError("axis_local must be a unit vector")
        for key in ("shaft_length_m", "shaft_radius_m", "nib_length_m", "nib_radius_m"):
            positive(getattr(self, key), key)
        if self.nib_length_m >= self.shaft_length_m:
            raise ValueError("nib_length_m must be smaller than shaft_length_m")
        vector([self.gripper_rad], 1, "gripper_rad")
        if self.simulation_only is not True or self.calibrated is not False:
            raise ValueError("This backend accepts only uncalibrated simulation profiles")


@dataclass(frozen=True)
class PaperFrame:
    origin_m: tuple = (.200, -.040, .005)
    x_axis: tuple = (0., 1., 0.)
    y_axis: tuple = (1., 0., 0.)
    working_rect_mm: tuple = (8., 8., 150., 80.)  # xmin, ymin, xmax, ymax.

    def validate(self, paper):
        paper.validate()
        vector(self.origin_m, 3, "origin_m")
        x, y = vector(self.x_axis, 3, "x_axis"), vector(self.y_axis, 3, "y_axis")
        if not (np.isclose(np.linalg.norm(x), 1, atol=1e-7)
                and np.isclose(np.linalg.norm(y), 1, atol=1e-7)
                and abs(float(x @ y)) < 1e-7):
            raise ValueError("Paper axes must be orthonormal")
        x0, y0, x1, y1 = vector(self.working_rect_mm, 4, "working_rect_mm")
        if not (paper.margin_mm <= x0 < x1 <= paper.width_mm - paper.margin_mm
                and paper.margin_mm <= y0 < y1 <= paper.height_mm - paper.margin_mm):
            raise ValueError("working_rect_mm must fit inside the paper margins")

    @property
    def rotation(self):
        # Page coordinates are x-right, y-down, z-away: intentionally left-handed.
        return np.column_stack([self.x_axis, self.y_axis, self.normal])

    @property
    def normal(self):
        return -np.cross(self.x_axis, self.y_axis)

    @property
    def geometry_rotation(self):
        # MuJoCo quaternions require a proper rotation, not the page reflection.
        return np.column_stack([self.x_axis, -np.asarray(self.y_axis), self.normal])

    def world(self, point_mm):
        return np.asarray(self.origin_m) + self.rotation @ (vector(point_mm, 3, "point_mm") * .001)

    def paper_mm(self, world_point):
        return self.rotation.T @ (vector(world_point, 3, "world_point") - np.asarray(self.origin_m)) * 1000


@dataclass(frozen=True)
class SimulationSettings:
    position_tolerance_mm: float = .75
    angle_tolerance_deg: float = 3.
    cartesian_step_mm: float = 1.
    max_joint_step_rad: float = .04
    joint_speed_rad_s: float = .5
    tip_speed_mm_s: float = 20.
    max_points: int = 20000
    ik_iterations: int = 200

    def validate(self):
        for key in ("position_tolerance_mm", "angle_tolerance_deg", "cartesian_step_mm",
                    "max_joint_step_rad", "joint_speed_rad_s", "tip_speed_mm_s"):
            positive(getattr(self, key), key)
        if self.angle_tolerance_deg >= 90 or self.cartesian_step_mm > 2 or self.max_joint_step_rad > .1:
            raise ValueError("Simulation tolerances or sampling bounds are too permissive")
        for key in ("max_points", "ik_iterations"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 100000:
                raise ValueError(f"{key} must be an integer in [1, 100000]")


@dataclass(frozen=True)
class TransitSettings:
    # A model-specific airborne pose, not a physical robot's boot/home position.
    ready_q_rad: tuple = (.0992351067, -.2717824768, .2524989785, 1.5859414678, -.1334262657)
    travel_height_mm: float = 16.
    minimum_clearance_mm: float = 12.

    def validate(self, paper, settings):
        vector(self.ready_q_rad, 5, "ready_q_rad")
        positive(self.travel_height_mm, "travel_height_mm")
        positive(self.minimum_clearance_mm, "minimum_clearance_mm")
        if self.travel_height_mm > 1000:
            raise ValueError("travel_height_mm must not exceed 1000 mm")
        if self.travel_height_mm < self.minimum_clearance_mm + settings.position_tolerance_mm:
            raise ValueError("travel_height_mm must include the clearance plus tracking tolerance")
        if self.travel_height_mm < paper.pen_lift_mm:
            raise ValueError("travel_height_mm must not be below pen_lift_mm")


def _xml_vector(value):
    return " ".join(str(float(x)) for x in value)


def paper_camera_pose(frame, paper):
    """Camera on the writing side, with page right screen-right and down screen-down."""
    angle = math.radians(65)
    right = np.asarray(frame.x_axis)
    backward = frame.normal * math.sin(angle) + np.asarray(frame.y_axis) * math.cos(angle)
    up = np.cross(backward, right)
    lookat = frame.world([paper.width_mm / 2, paper.height_mm * .25, 140])
    return lookat + backward * .78, np.column_stack([right, up, backward])


def build_scene(tool=None, frame=None, paper=None):
    tool, frame, paper = tool or ToolMount(), frame or PaperFrame(), paper or PaperSettings()
    tool.validate()
    frame.validate(paper)
    root = ET.parse(ASSETS / "so101.xml").getroot()
    root.set("model", "so101_writing_kinematics")
    ET.SubElement(root.find("visual"), "global", offwidth="800", offheight="600")
    root.find("compiler").set("meshdir", "assets")
    world = root.find("worldbody")
    gripper = root.find(".//body[@name='gripper']")
    grip_site = gripper.find("site[@name='gripperframe']")
    quat = vector([float(x) for x in grip_site.get("quat").split()], 4, "site quaternion")
    quat /= np.linalg.norm(quat)
    mount = ET.SubElement(gripper, "body", name="virtual_pen", pos=grip_site.get("pos"),
                          quat=_xml_vector(quat))
    tip = np.asarray(tool.tip_offset_m)
    axis = np.asarray(tool.axis_local)
    ET.SubElement(mount, "site", name="pen_tip", pos=_xml_vector(tip), size="0.001", rgba="1 0.2 0.2 1")
    ET.SubElement(mount, "site", name="pen_axis", pos=_xml_vector(tip - axis * .01), size="0.0005")
    # Massless rigid proxies: no claim about brush flex, pressure, grip or servo dynamics.
    ET.SubElement(mount, "geom", name="pen_shaft", type="capsule", mass="0", group="0",
                  fromto=_xml_vector(np.r_[tip - axis * tool.shaft_length_m, tip - axis * tool.nib_length_m]),
                  size=str(tool.shaft_radius_m), rgba="0.35 0.08 0.16 1")
    ET.SubElement(mount, "geom", name="pen_nib", type="capsule", mass="0", group="0",
                  fromto=_xml_vector(np.r_[tip - axis * tool.nib_length_m, tip - axis * tool.nib_radius_m]),
                  size=str(tool.nib_radius_m), rgba="0.1 0.1 0.1 1")
    # A mounted tool intentionally overlaps the host jaws; grip mechanics are not simulated.
    contact = ET.SubElement(root, "contact")
    for body in gripper.iter("body"):
        if body.get("name") != "virtual_pen":
            ET.SubElement(contact, "exclude", body1="virtual_pen", body2=body.get("name"))
    rotation = frame.geometry_rotation
    paper_quat = np.zeros(4)
    mujoco.mju_mat2Quat(paper_quat, rotation.ravel())
    center = frame.world([paper.width_mm / 2, paper.height_mm / 2, -.5])
    ET.SubElement(world, "geom", name="writing_paper", type="box", pos=_xml_vector(center),
                  quat=_xml_vector(paper_quat), size=_xml_vector([paper.width_mm / 2000, paper.height_mm / 2000, .0005]),
                  rgba="0.98 0.98 0.98 1")
    ET.SubElement(world, "geom", name="writing_table", type="plane", size="1 1 .01", rgba="0.65 0.69 0.70 1")
    ET.SubElement(world, "light", pos="0 0 1.5", dir="0 0 -1", diffuse="0.8 0.8 0.8")
    camera_pos, camera_rotation = paper_camera_pose(frame, paper)
    ET.SubElement(world, "camera", name="writing_view", pos=_xml_vector(camera_pos),
                  xyaxes=_xml_vector(np.r_[camera_rotation[:, 0], camera_rotation[:, 1]]))
    x0, y0, x1, y1 = frame.working_rect_mm
    center = frame.world([(x0 + x1) / 2, (y0 + y1) / 2, .1])
    ET.SubElement(world, "site", name="candidate_work_area", type="box", pos=_xml_vector(center),
                  quat=_xml_vector(paper_quat), size=_xml_vector([(x1 - x0) / 2000, (y1 - y0) / 2000, .00005]),
                  rgba="0.1 0.65 0.4 0.12", group="1")
    assets = {f"assets/{p.name}": p.read_bytes() for p in (ASSETS / "assets").glob("*.stl")}
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"), assets)
    return model


class SimulationStopped(Exception):
    pass


class PenKinematics:
    def __init__(self, model, tool=None, settings=None):
        self.model, self.tool, self.settings = model, tool or ToolMount(), settings or SimulationSettings()
        self.tool.validate()
        self.settings.validate()
        self.data = mujoco.MjData(model)
        joint_ids = np.array([model.joint(name).id for name in ARM_JOINTS])
        self.qadr, self.dofs = model.jnt_qposadr[joint_ids], model.jnt_dofadr[joint_ids]
        self.limits = model.jnt_range[joint_ids].copy()
        self.grip_adr = model.joint("gripper").qposadr[0]
        grip_limits = model.jnt_range[model.joint("gripper").id]
        if not grip_limits[0] <= self.tool.gripper_rad <= grip_limits[1]:
            raise ValueError("gripper_rad exceeds model joint limits")
        self.tip_id, self.axis_id = model.site("pen_tip").id, model.site("pen_axis").id
        self.jacp, self.jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))

    def pose(self, q):
        q = vector(q, 5, "arm q")
        if np.any(q < self.limits[:, 0]) or np.any(q > self.limits[:, 1]):
            raise ValueError("arm q exceeds model joint limits")
        self.data.qpos[self.qadr] = q
        self.data.qpos[self.grip_adr] = self.tool.gripper_rad
        mujoco.mj_forward(self.model, self.data)
        tip = self.data.site_xpos[self.tip_id].copy()
        axis = tip - self.data.site_xpos[self.axis_id]
        axis /= np.linalg.norm(axis)
        return tip, axis

    def solve(self, target, direction, seed, stop_requested=None):
        target, direction = vector(target, 3, "target"), vector(direction, 3, "direction")
        if not np.isclose(np.linalg.norm(direction), 1, atol=1e-7):
            raise ValueError("direction must be a unit vector")
        q = vector(seed, 5, "seed").copy()
        if np.any(q < self.limits[:, 0]) or np.any(q > self.limits[:, 1]):
            raise ValueError("seed exceeds model limits")
        best, best_score = q.copy(), float("inf")
        for _ in range(self.settings.ik_iterations):
            if stop_requested and stop_requested():
                raise SimulationStopped()
            tip, axis = self.pose(q)
            ep, ea = target - tip, direction - axis
            angle = math.degrees(math.acos(float(np.clip(axis @ direction, -1, 1))))
            score = float(np.linalg.norm(ep) + .08 * np.linalg.norm(ea))
            if score < best_score:
                best, best_score = q.copy(), score
            if np.linalg.norm(ep) * 1000 <= self.settings.position_tolerance_mm and angle <= self.settings.angle_tolerance_deg:
                break
            mujoco.mj_jacSite(self.model, self.data, self.jacp, self.jacr, self.tip_id)
            skew = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
            jac = np.vstack([self.jacp[:, self.dofs], .08 * (-skew @ self.jacr[:, self.dofs])])
            err = np.r_[ep, .08 * ea]
            delta = jac.T @ np.linalg.solve(jac @ jac.T + .005 ** 2 * np.eye(6), err)
            delta *= min(1., .12 / max(float(np.max(np.abs(delta))), 1e-12))
            q = np.clip(q + .7 * delta, self.limits[:, 0], self.limits[:, 1])
        tip, axis = self.pose(q)
        if np.linalg.norm(target - tip) + .08 * np.linalg.norm(direction - axis) < best_score:
            best = q.copy()
        tip, axis = self.pose(best)
        position_mm = float(np.linalg.norm(target - tip) * 1000)
        angle = math.degrees(math.acos(float(np.clip(axis @ direction, -1, 1))))
        return {"q": best, "tip": tip, "position_error_mm": position_mm, "angle_error_deg": angle,
                "converged": position_mm <= self.settings.position_tolerance_mm and angle <= self.settings.angle_tolerance_deg}

    def collisions(self):
        problems = []
        for contact in self.data.contact[:self.data.ncon]:
            names = [self.model.geom(int(g)).name or f"geom_{g}" for g in (contact.geom1, contact.geom2)]
            intended = set(names) == {"pen_nib", "writing_paper"}
            threshold = -.0002 if intended else -.00005
            if contact.dist < threshold:
                problems.append({"geoms": names, "penetration_mm": float(-contact.dist * 1000)})
        return problems


def _targets(plan, settings, ready_point=None, travel_height_mm=None):
    commands = []
    if ready_point is not None:
        first = plan.strokes[0].points_mm[0]
        commands.extend([
            {"op": "approach_lift", "phase": "approach", "stroke": -1,
             "point_mm": [*ready_point[:2], travel_height_mm]},
            {"op": "approach_transfer", "phase": "approach", "stroke": -1,
             "point_mm": [*first, travel_height_mm]},
            {"op": "approach_lower", "phase": "approach", "stroke": -1,
             "point_mm": [*first, plan.paper.pen_lift_mm]},
        ])
    commands.extend({**command, "phase": "writing"} for command in plan.commands())
    if ready_point is not None:
        last = plan.strokes[-1].points_mm[-1]
        commands.extend([
            {"op": "departure_lift", "phase": "departure", "stroke": -1,
             "point_mm": [*last, travel_height_mm]},
            {"op": "departure_transfer", "phase": "departure", "stroke": -1,
             "point_mm": [*ready_point[:2], travel_height_mm]},
            {"op": "departure_lower", "phase": "departure", "stroke": -1,
             "point_mm": ready_point},
        ])
    previous = ready_point
    for command in commands:
        points = command.get("points_mm", [command.get("point_mm")])
        for value in points:
            point = np.asarray(value, dtype=float)
            count = 1 if previous is None else max(1, math.ceil(np.linalg.norm(point - previous) / settings.cartesian_step_mm))
            for index in range(1, count + 1):
                sampled = point if previous is None else previous + (point - previous) * index / count
                yield sampled, command["op"], command["stroke"], command["phase"]
            previous = point


def preflight(plan, tool=None, frame=None, settings=None, stop_requested=None, initial_q=None, transit=None):
    """Return diagnostics only, never a hardware command stream.

    initial_q is a writing-only IK seed. transit starts at a configured virtual
    airborne pose; reaching it from a real robot's state is not validated.
    """
    if not isinstance(plan, WritingPlan):
        raise ValueError("plan must be a WritingPlan")
    tool, frame, settings = tool or ToolMount(), frame or PaperFrame(), settings or SimulationSettings()
    tool.validate()
    frame.validate(plan.paper)
    settings.validate()
    if transit is not None:
        transit.validate(plan.paper, settings)
        if initial_q is not None:
            raise ValueError("Use transit.ready_q_rad, not the writing-only initial_q IK seed")
    report = {"format_version": 2, "kind": "writing_kinematic_preflight",
              "paper_coordinate_convention": "top_left_x_right_y_down_z_away",
              "created_at": datetime.now(timezone(timedelta(hours=9))).isoformat(),
              "mode": "simulation_only", "motion_authorized": False, "physical_calibration_required": True,
              "joint_command_stream_available": False, "dynamics_validated": False,
              "status": "rejected", "trajectory_complete": False, "text": plan.text,
              "tool": asdict(tool), "paper_frame": asdict(frame), "settings": asdict(settings),
              "paper": asdict(plan.paper), "samples": [], "failure": None,
              "scope": "full_cycle" if transit else "writing_only", "phase": "initialization",
              "transit": asdict(transit) if transit else None,
              "initial_pose_validated": False, "approach_validated": False,
              "departure_validated": False, "ready_return_validated": False,
              "start_condition": ("configured_airborne_pose_initialized_in_simulation; physical_home_to_ready_not_validated"
                                  if transit else "first_pose_initialized_in_simulation; approach_from_robot_home_not_validated")}
    x0, y0, x1, y1 = frame.working_rect_mm
    for stroke in plan.strokes:
        if not stroke.points_mm:
            raise ValueError("A stroke must have points")
        for point in stroke.points_mm:
            x, y = vector(point, 2, "stroke point")
            if not x0 <= x <= x1 or not y0 <= y <= y1:
                report["failure"] = {"reason": "outside_candidate_work_area", "point_mm": [x, y, 0]}
                return report, None
    if not plan.strokes:
        report["failure"] = {"reason": "empty_trajectory"}
        return report, None
    model = build_scene(tool, frame, plan.paper)
    kin = PenKinematics(model, tool, settings)
    direction = -frame.normal
    seeds = [np.array([0., -.8, .8, 1., 0.]), np.zeros(5),
             np.array([0., .8, -.8, -1., 0.]), np.array([.3, -.8, 1.2, .8, .5])]
    if initial_q is not None:
        seeds = [vector(initial_q, 5, "initial_q")]
    q, last_world, elapsed = None, None, 0.
    max_interpolated_position_mm, max_interpolated_angle_deg = 0., 0.
    min_transport_tip_height_mm = float("inf")
    max_sampled_tip_speed_mm_s = 0.

    def check_pose(pose, target, clearance, interpolated=False):
        nonlocal max_interpolated_position_mm, max_interpolated_angle_deg, min_transport_tip_height_mm
        if stop_requested and stop_requested():
            raise SimulationStopped()
        tip, axis = kin.pose(pose)
        error_mm = float(np.linalg.norm(tip - target) * 1000)
        angle_deg = math.degrees(math.acos(float(np.clip(axis @ direction, -1, 1))))
        if interpolated:
            max_interpolated_position_mm = max(max_interpolated_position_mm, error_mm)
            max_interpolated_angle_deg = max(max_interpolated_angle_deg, angle_deg)
        actual = frame.paper_mm(tip)
        if transit is not None and clearance == transit.minimum_clearance_mm:
            min_transport_tip_height_mm = min(min_transport_tip_height_mm, float(actual[2]))
        prefix = "interpolated_" if interpolated else ""
        failure = None
        if not x0 <= actual[0] <= x1 or not y0 <= actual[1] <= y1:
            failure = {"reason": prefix + "tip_outside_work_area", "actual_point_mm": actual.tolist()}
        elif error_mm > settings.position_tolerance_mm or angle_deg > settings.angle_tolerance_deg:
            failure = {"reason": prefix + "tracking_error", "position_error_mm": error_mm, "angle_error_deg": angle_deg}
        elif clearance is not None and actual[2] < clearance - 1e-9:
            failure = {"reason": prefix + "insufficient_clearance", "height_mm": float(actual[2]), "required_mm": clearance}
        else:
            collisions = kin.collisions()
            if collisions:
                failure = {"reason": prefix + "collision", "contacts": collisions}
        if failure:
            report["failure"] = {**failure, "phase": report["phase"]}
            return None
        return tip, error_mm, angle_deg

    def accept(new_q, target, point, op, stroke, phase, clearance=None):
        nonlocal q, last_world, elapsed, max_sampled_tip_speed_mm_s
        report["phase"] = phase
        if stop_requested and stop_requested():
            raise SimulationStopped()
        if len(report["samples"]) >= settings.max_points:
            report["failure"] = {"reason": "point_budget_exceeded", "phase": phase}
            return False
        if q is not None:
            interval_tips = [np.asarray(report["samples"][-1]["tip_world_m"])]
            delta = float(np.max(np.abs(new_q - q)))
            if delta > settings.max_joint_step_rad:
                report["failure"] = {"reason": "joint_discontinuity", "max_delta_rad": delta, "phase": phase}
                return False
            # Sample the interior of every joint interval, including approach and return.
            for fraction in (.25, .5, .75):
                expected = last_world + fraction * (target - last_world)
                checked = check_pose(q + fraction * (new_q - q), expected, clearance, True)
                if checked is None:
                    return False
                interval_tips.append(checked[0])
            duration = max(delta / settings.joint_speed_rad_s,
                           float(np.linalg.norm(target - last_world)) * 1000 / settings.tip_speed_mm_s)
        else:
            duration = 0.
        result = check_pose(new_q, target, clearance)
        if result is None:
            return False
        tip, error_mm, angle_deg = result
        if q is not None:
            interval_tips.append(tip)
            max_quarter_distance_mm = max(float(np.linalg.norm(b - a)) * 1000
                                          for a, b in zip(interval_tips, interval_tips[1:]))
            duration = max(duration, 4 * max_quarter_distance_mm / settings.tip_speed_mm_s)
            if duration > 0:
                max_sampled_tip_speed_mm_s = max(max_sampled_tip_speed_mm_s, 4 * max_quarter_distance_mm / duration)
        elapsed += duration
        report["samples"].append({"op": op, "phase": phase, "stroke": stroke, "point_mm": point.tolist(),
                                   "target_world_m": target.tolist(), "tip_world_m": tip.tolist(),
                                   "arm_q_rad": new_q.tolist(), "time_s": elapsed,
                                   "position_error_mm": error_mm, "angle_error_deg": angle_deg})
        q, last_world = new_q.copy(), target.copy()
        return True

    try:
        ready_point, ready_q, ready_tip, travel_height = None, None, None, None
        if transit:
            if stop_requested and stop_requested():
                raise SimulationStopped()
            ready_q = vector(transit.ready_q_rad, 5, "ready_q_rad")
            if np.any(ready_q < kin.limits[:, 0]) or np.any(ready_q > kin.limits[:, 1]):
                report["failure"] = {"reason": "initial_pose_joint_limits", "phase": "initialization"}
                return report, model
            ready_tip, _ = kin.pose(ready_q)
            ready_point = frame.paper_mm(ready_tip)
            if not accept(ready_q, ready_tip, ready_point, "ready", -1, "initialization", transit.minimum_clearance_mm):
                return report, model
            report["initial_pose_validated"] = True
            travel_height = max(transit.travel_height_mm, float(ready_point[2]))
            report["transit_height_mm"] = travel_height
        for point, op, stroke, phase in _targets(plan, settings, ready_point, travel_height):
            if transit and phase == "writing" and report["phase"] == "approach":
                report["approach_validated"] = True
            report["phase"] = phase
            if stop_requested and stop_requested():
                raise SimulationStopped()
            if len(report["samples"]) >= settings.max_points:
                report["failure"] = {"reason": "point_budget_exceeded", "phase": phase}
                return report, model
            target = frame.world(point)
            if q is None:
                candidates = [kin.solve(target, direction, seed, stop_requested) for seed in seeds]
                valid = [r for r in candidates if r["converged"]]
                result = min(valid or candidates, key=lambda r: r["position_error_mm"] + r["angle_error_deg"])
            else:
                result = kin.solve(target, direction, q, stop_requested)
            if not result["converged"]:
                report["failure"] = {"reason": "ik_not_converged", "point_mm": point.tolist(),
                                     "phase": phase, "position_error_mm": result["position_error_mm"], "angle_error_deg": result["angle_error_deg"]}
                return report, model
            clearance = None
            if transit and phase != "writing":
                clearance = (max(0., plan.paper.pen_lift_mm - settings.position_tolerance_mm)
                             if op in ("approach_lower", "departure_lift") else transit.minimum_clearance_mm)
            if not accept(result["q"], target, point, op, stroke, phase, clearance):
                return report, model
        if transit:
            report["departure_validated"] = True
            # Align back to the exact configured joints; a near-ready IK result is not enough.
            start_q, start_world = q.copy(), last_world.copy()
            count = max(1, math.ceil(float(np.max(np.abs(ready_q - start_q))) / (settings.max_joint_step_rad / 2)),
                        math.ceil(float(np.linalg.norm(ready_tip - start_world)) * 1000 / settings.cartesian_step_mm))
            for index in range(1, count + 1):
                fraction = index / count
                target = start_world + fraction * (ready_tip - start_world)
                if not accept(start_q + fraction * (ready_q - start_q), target, frame.paper_mm(target),
                              "return_ready", -1, "return", transit.minimum_clearance_mm):
                    return report, model
    except SimulationStopped:
        report["status"] = "stopped"
        report["failure"] = {"reason": "stop_requested", "phase": report["phase"]}
        return report, model
    report.update(status="passed", phase="complete", trajectory_complete=True,
                  approach_validated=transit is not None, departure_validated=transit is not None,
                  ready_return_validated=transit is not None,
                  min_transport_tip_height_mm=min_transport_tip_height_mm if transit else None,
                  phase_sample_counts={phase: sum(s["phase"] == phase for s in report["samples"])
                                       for phase in ("initialization", "approach", "writing", "departure", "return")},
                  simulated_duration_s=elapsed,
                  max_sampled_tip_speed_mm_s=max_sampled_tip_speed_mm_s,
                  max_interpolated_position_error_mm=max_interpolated_position_mm,
                  max_interpolated_angle_error_deg=max_interpolated_angle_deg,
                  max_position_error_mm=max(s["position_error_mm"] for s in report["samples"]),
                  max_angle_error_deg=max(s["angle_error_deg"] for s in report["samples"]))
    return report, model

def calibration_cases():
    """Offline geometry cases; not physical trials or AI-generated replies."""
    paper = PaperSettings(character_mm=24)
    cases = {
        "line": WritingPlan("직선", paper, (PenStroke("선", 0, ((12., 12.), (40., 12.))),), 1),
        "square": WritingPlan("네모", paper, (PenStroke("네모", 0,
                                ((12., 12.), (36., 12.), (36., 36.), (12., 36.), (12., 12.))),), 1),
        "call": plan_text("네", PaperSettings(character_mm=28)),
        "ga": plan_text("가", paper), "han": plan_text("한", paper), "geul": plan_text("글", paper),
        "sentence": plan_text("안녕하세요.", PaperSettings(character_mm=18)),
    }
    return cases
