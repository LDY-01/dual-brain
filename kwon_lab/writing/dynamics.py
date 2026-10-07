"""Time-stepped MuJoCo actuator checks; never a physical robot controller."""

from bisect import bisect_right
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import math

import mujoco
import numpy as np

from .simulation import ARM_JOINTS, PaperFrame, PenKinematics, TransitSettings, positive, preflight


@dataclass(frozen=True)
class DynamicsSettings:
    timestep_s: float = .005
    time_scale: float = 2.
    settle_s: float = .5
    final_hold_s: float = .3
    tracking_tolerance_mm: float = 1.
    joint_tracking_tolerance_rad: float = .02
    max_joint_speed_rad_s: float = .75
    max_tip_speed_mm_s: float = 30.
    stop_timeout_s: float = 1.
    stop_stable_s: float = .1
    stop_joint_speed_rad_s: float = .02
    stop_tip_speed_mm_s: float = 2.
    max_stop_displacement_mm: float = 2.
    max_steps: int = 40000

    def validate(self):
        for key, value in asdict(self).items():
            if key != "max_steps":
                positive(value, key)
        if not .001 <= self.timestep_s <= .01:
            raise ValueError("timestep_s must be in [0.001, 0.01]")
        if not 1 <= self.time_scale <= 10:
            raise ValueError("time_scale must be in [1, 10]; do not accelerate the preflight schedule")
        if not self.timestep_s <= self.stop_stable_s <= self.stop_timeout_s <= 10:
            raise ValueError("Invalid stop timing bounds")
        if not self.stop_stable_s <= self.settle_s <= 10 or not self.stop_stable_s <= self.final_hold_s <= 10:
            raise ValueError("Invalid settling/hold timing bounds")
        if (self.tracking_tolerance_mm > 2 or self.joint_tracking_tolerance_rad > .1
                or self.max_joint_speed_rad_s > 2 or self.max_tip_speed_mm_s > 100
                or self.max_stop_displacement_mm > 5):
            raise ValueError("Dynamics guard bounds are too permissive")
        if self.stop_joint_speed_rad_s >= self.max_joint_speed_rad_s or self.stop_tip_speed_mm_s >= self.max_tip_speed_mm_s:
            raise ValueError("Stop thresholds must be below motion thresholds")
        if isinstance(self.max_steps, bool) or not isinstance(self.max_steps, int) or not 1 <= self.max_steps <= 200000:
            raise ValueError("max_steps must be an integer in [1, 200000]")


def interpolate_reference(samples, times, time_s):
    """Use the incoming segment's operation, retaining repeated zero-time points."""
    index = min(bisect_right(times, time_s), len(samples) - 1)
    previous = max(0, index - 1)
    start, end = samples[previous], samples[index]
    duration = times[index] - times[previous]
    fraction = 0. if duration <= 0 else float(np.clip((time_s - times[previous]) / duration, 0, 1))
    q = np.asarray(start["arm_q_rad"]) + fraction * (np.asarray(end["arm_q_rad"]) - start["arm_q_rad"])
    target = np.asarray(start["target_world_m"]) + fraction * (np.asarray(end["target_world_m"]) - start["target_world_m"])
    return q, target, end["op"], end["stroke"], end["phase"]


def run_dynamics(plan, tool=None, frame=None, settings=None, transit=None, dynamics=None,
                 stop_requested=None, stop_after_s=None):
    """Preflight the whole cycle, then drive only data.ctrl and integrate mj_step.

    qpos is initialized once. No state teleportation, gravity compensation, gain
    tuning, hardware stream, calibrated pen mass, or pressure model is provided.
    """
    dynamics, transit = dynamics or DynamicsSettings(), transit or TransitSettings()
    dynamics.validate()
    if stop_after_s is not None:
        if isinstance(stop_after_s, bool) or not isinstance(stop_after_s, (int, float)) or not math.isfinite(stop_after_s) or stop_after_s < 0:
            raise ValueError("stop_after_s must be a finite nonnegative number")
    reference, model = preflight(plan, tool, frame, settings, stop_requested=stop_requested, transit=transit)
    report = {"format_version": 2, "kind": "writing_dynamics_rollout",
              "created_at": datetime.now(timezone(timedelta(hours=9))).isoformat(),
              "mode": "simulation_only", "scope": "full_cycle", "status": "rejected",
              "phase": "preflight", "trajectory_complete": False, "motion_authorized": False,
              "joint_command_stream_available": False, "physical_calibration_required": True,
              "dynamics_validated": False, "physical_dynamics_validated": False,
              "pressure_validated": False, "pen_mass_and_compliance_calibrated": False,
              "paper_coordinate_convention": reference["paper_coordinate_convention"],
              "text": plan.text, "paper": reference["paper"], "paper_frame": reference["paper_frame"],
              "tool": reference["tool"], "settings": reference["settings"], "transit": reference["transit"],
              "dynamics_settings": asdict(dynamics), "samples": [], "failure": None,
              "stop_requested": False, "stop_test_passed": False, "stop_settled": False,
              "initial_pose_validated": False, "approach_validated": False,
              "departure_validated": False, "ready_return_validated": False,
              "start_condition": reference["start_condition"],
              "preflight": {key: reference.get(key) for key in
                            ("status", "scope", "failure", "approach_validated", "departure_validated",
                             "ready_return_validated", "max_position_error_mm", "simulated_duration_s")}}
    if (reference["status"] != "passed" or reference.get("scope") != "full_cycle"
            or not reference.get("ready_return_validated") or reference.get("motion_authorized") is not False):
        report["status"] = "stopped" if reference["status"] == "stopped" else "rejected"
        report["failure"] = {"reason": "preflight_not_passed", "detail": reference["failure"]}
        return report, model

    model.opt.timestep = dynamics.timestep_s
    kin = PenKinematics(model, tool, settings)
    actual = mujoco.MjData(model)
    names = (*ARM_JOINTS, "gripper")
    actuator_ids = np.array([model.actuator(name).id for name in names])
    qadr = np.r_[kin.qadr, kin.grip_adr]
    dofs = np.r_[kin.dofs, model.joint("gripper").dofadr[0]]
    limits = model.jnt_range[[model.joint(name).id for name in names]]
    controls = model.actuator_ctrlrange[actuator_ids]
    forces = model.actuator_forcerange[actuator_ids]
    samples = reference["samples"]
    times = [s["time_s"] * dynamics.time_scale for s in samples]
    duration = times[-1]
    report.update(reference_duration_s=samples[-1]["time_s"], execution_reference_duration_s=duration,
                  path_tolerance_mm=dynamics.tracking_tolerance_mm + kin.settings.position_tolerance_mm,
                  model={"mujoco_version": mujoco.__version__, "timestep_s": float(model.opt.timestep),
                         "gravity_m_s2": model.opt.gravity.tolist(), "integrator": int(model.opt.integrator),
                         "actuator_names": list(names), "kp": model.actuator_gainprm[actuator_ids, 0].tolist(),
                         "kv": (-model.actuator_biasprm[actuator_ids, 2]).tolist(),
                         "force_ranges_model_units": forces.tolist(), "control_ranges_rad": controls.tolist(),
                         "pen_proxy_mass_kg": float(model.body_mass[model.body("virtual_pen").id]),
                         "actuator_gains_physically_calibrated": False})
    for sample in samples:
        command = np.r_[sample["arm_q_rad"], kin.tool.gripper_rad]
        if np.any(command < controls[:, 0]) or np.any(command > controls[:, 1]):
            report["failure"] = {"reason": "reference_exceeds_actuator_control_range"}
            return report, model
    initial = np.asarray(samples[0]["arm_q_rad"])
    actual.qpos[kin.qadr] = initial
    actual.qpos[kin.grip_adr] = kin.tool.gripper_rad
    actual.ctrl[actuator_ids] = np.r_[initial, kin.tool.gripper_rad]
    mujoco.mj_forward(model, actual)
    frame = PaperFrame(**reference["paper_frame"])
    normal = frame.normal
    first_tip = actual.site_xpos[kin.tip_id].copy()
    last_tip = first_tip.copy()
    metrics = {"max_tracking_error_mm": 0., "max_path_error_mm": 0., "max_joint_error_rad": 0.,
               "max_axis_error_deg": 0., "max_joint_speed_rad_s": 0., "max_tip_speed_mm_s": 0.,
               "max_actuator_force_model_units": 0., "actuator_saturation_steps": 0}
    step_count, trajectory_start, stopped_at = 0, None, None
    hold_q, hold_tip, stable_s = None, None, 0.
    stop_max_displacement = 0.
    initial_stable_s, final_stable_s = 0., 0.

    def fail(reason, **details):
        report["failure"] = {"reason": reason, "phase": report["phase"],
                             "sim_time_s": float(actual.time) if math.isfinite(actual.time) else None, **details}

    def advance(q, target, op, stroke, phase, clearance=None, reference_q=None):
        nonlocal step_count, last_tip
        report["phase"] = phase
        if step_count >= dynamics.max_steps:
            fail("dynamics_step_budget_exceeded")
            return None
        ctrl = np.r_[q, kin.tool.gripper_rad]
        if np.any(ctrl < controls[:, 0]) or np.any(ctrl > controls[:, 1]):
            fail("control_range_violation")
            return None
        actual.ctrl[actuator_ids] = ctrl
        before = actual.time
        mujoco.mj_step(model, actual)
        step_count += 1
        if (not np.all(np.isfinite(actual.qpos)) or not np.all(np.isfinite(actual.qvel))
                or not math.isfinite(actual.time) or abs(actual.time - before - dynamics.timestep_s) > 1e-8):
            fail("nonfinite_state_or_engine_reset")
            return None
        mujoco.mj_forward(model, actual)
        if any(w.number for w in actual.warning) or not np.all(np.isfinite(actual.qacc)):
            fail("mujoco_numerical_warning")
            return None
        tip = actual.site_xpos[kin.tip_id].copy()
        axis = tip - actual.site_xpos[kin.axis_id]
        axis /= np.linalg.norm(axis)
        reference_q = q if reference_q is None else reference_q
        nominal, _ = kin.pose(reference_q)
        track = float(np.linalg.norm(tip - nominal) * 1000)
        path_error = float(np.linalg.norm(tip - target) * 1000)
        joint_error = float(np.max(np.abs(actual.qpos[qadr] - np.r_[reference_q, kin.tool.gripper_rad])))
        axis_error = math.degrees(math.acos(float(np.clip(axis @ (-normal), -1, 1))))
        joint_speed = float(np.max(np.abs(actual.qvel[dofs])))
        step_tip_speed = float(np.linalg.norm(tip - last_tip) * 1000 / dynamics.timestep_s)
        mujoco.mj_jacSite(model, actual, kin.jacp, kin.jacr, kin.tip_id)
        instantaneous_tip_speed = float(np.linalg.norm(kin.jacp @ actual.qvel) * 1000)
        tip_speed = max(step_tip_speed, instantaneous_tip_speed)
        peak_force = float(np.max(np.abs(actual.actuator_force[actuator_ids])))
        for key, value in (("max_tracking_error_mm", track), ("max_path_error_mm", path_error),
                           ("max_joint_error_rad", joint_error), ("max_axis_error_deg", axis_error),
                           ("max_joint_speed_rad_s", joint_speed), ("max_tip_speed_mm_s", tip_speed),
                           ("max_actuator_force_model_units", peak_force)):
            metrics[key] = max(metrics[key], value)
        if np.any(np.abs(actual.actuator_force[actuator_ids]) >= .99 * np.max(np.abs(forces), axis=1)):
            metrics["actuator_saturation_steps"] += 1
        actual_point = frame.paper_mm(tip)
        collisions = []
        for contact in actual.contact[:actual.ncon]:
            geom_names = [model.geom(int(g)).name or f"geom_{g}" for g in (contact.geom1, contact.geom2)]
            threshold = -.0002 if set(geom_names) == {"pen_nib", "writing_paper"} else -.00005
            if contact.dist < threshold:
                collisions.append({"geoms": geom_names, "penetration_mm": float(-contact.dist * 1000)})
        checks = [(bool(np.any(actual.qpos[qadr] < limits[:, 0] - 1e-7)
                        or np.any(actual.qpos[qadr] > limits[:, 1] + 1e-7)), "actual_joint_limit_violation"),
                  (bool(collisions), "actual_collision"),
                  (track > dynamics.tracking_tolerance_mm, "tip_tracking_error"),
                  (path_error > report["path_tolerance_mm"], "path_tracking_error"),
                  (joint_error > dynamics.joint_tracking_tolerance_rad, "joint_tracking_error"),
                  (axis_error > kin.settings.angle_tolerance_deg, "actual_axis_error"),
                  (joint_speed > dynamics.max_joint_speed_rad_s, "actual_joint_overspeed"),
                  (tip_speed > dynamics.max_tip_speed_mm_s, "actual_tip_overspeed"),
                  (not frame.working_rect_mm[0] <= actual_point[0] <= frame.working_rect_mm[2]
                   or not frame.working_rect_mm[1] <= actual_point[1] <= frame.working_rect_mm[3], "actual_tip_outside_work_area"),
                  (clearance is not None and actual_point[2] < clearance, "actual_insufficient_clearance")]
        report["samples"].append({"time_s": float(actual.time), "phase": phase, "op": op, "stroke": stroke,
                                   "arm_q_rad": actual.qpos[kin.qadr].tolist(),
                                   "gripper_q_rad": float(actual.qpos[kin.grip_adr]),
                                   "arm_qvel_rad_s": actual.qvel[kin.dofs].tolist(),
                                   "command_q_rad": q.tolist(), "point_mm": frame.paper_mm(target).tolist(),
                                   "reference_q_rad": reference_q.tolist(),
                                   "target_world_m": target.tolist(), "tip_world_m": tip.tolist(),
                                   "tracking_error_mm": track, "path_error_mm": path_error,
                                   "joint_error_rad": joint_error, "angle_error_deg": axis_error,
                                   "tip_speed_mm_s": tip_speed, "joint_speed_rad_s": joint_speed,
                                   "instantaneous_tip_speed_mm_s": instantaneous_tip_speed,
                                   "step_tip_speed_mm_s": step_tip_speed,
                                   "actuator_force_model_units": actual.actuator_force[actuator_ids].tolist()})
        last_tip = tip
        for violated, reason in checks:
            if violated:
                fail(reason, contacts=collisions) if reason == "actual_collision" else fail(reason)
                return None
        return tip, joint_speed, tip_speed

    try:
        while True:
            if stop_requested and stop_requested():
                requested = True
            else:
                requested = (trajectory_start is not None and stop_after_s is not None
                             and actual.time - trajectory_start >= stop_after_s - 1e-9)
            if requested and stopped_at is None:
                stopped_at = actual.time
                hold_q = actual.qpos[kin.qadr].copy()
                hold_tip = actual.site_xpos[kin.tip_id].copy()
                report.update(stop_requested=True, stop_request_sim_time_s=float(stopped_at),
                              stop_request_trajectory_time_s=float(stopped_at - trajectory_start) if trajectory_start is not None else None,
                              stop_command_q_rad=hold_q.tolist(), stop_strategy="hold_current_position_without_resetting_velocity")
            if stopped_at is not None:
                result = advance(hold_q, hold_tip, "hold_stop", -1, "stopping")
                if result is None:
                    break
                tip, joint_speed, tip_speed = result
                stop_max_displacement = max(stop_max_displacement, float(np.linalg.norm(tip - hold_tip) * 1000))
                if stop_max_displacement > dynamics.max_stop_displacement_mm:
                    fail("stop_displacement_exceeded")
                    break
                stable_s = stable_s + dynamics.timestep_s if (joint_speed <= dynamics.stop_joint_speed_rad_s
                                                              and tip_speed <= dynamics.stop_tip_speed_mm_s) else 0.
                if stable_s >= dynamics.stop_stable_s - 1e-9:
                    report.update(status="stopped", stop_test_passed=True, stop_settled=True,
                                  stop_settling_time_s=float(actual.time - stopped_at))
                    break
                if actual.time - stopped_at >= dynamics.stop_timeout_s - 1e-9:
                    fail("stop_not_settled")
                    break
            elif trajectory_start is None:
                result = advance(initial, first_tip, "initial_hold", -1, "settling", transit.minimum_clearance_mm)
                if result is None:
                    break
                _, joint_speed, tip_speed = result
                initial_stable_s = initial_stable_s + dynamics.timestep_s if (
                    joint_speed <= dynamics.stop_joint_speed_rad_s and tip_speed <= dynamics.stop_tip_speed_mm_s) else 0.
                if actual.time >= dynamics.settle_s - 1e-9:
                    if initial_stable_s < dynamics.stop_stable_s - 1e-9:
                        fail("initial_pose_not_settled")
                        break
                    trajectory_start = actual.time
                    report["initial_pose_validated"] = True
                    report["trajectory_start_sim_time_s"] = float(trajectory_start)
            else:
                elapsed = actual.time - trajectory_start
                if elapsed >= duration + dynamics.final_hold_s - 1e-9:
                    if final_stable_s < dynamics.stop_stable_s - 1e-9:
                        fail("final_pose_not_settled")
                    else:
                        report.update(status="passed", phase="complete", trajectory_complete=True, dynamics_validated=True,
                                      approach_validated=True, departure_validated=True, ready_return_validated=True)
                    break
                # Command at the start of the step, then inspect the post-step state.
                q, _, _, _, _ = interpolate_reference(samples, times, min(elapsed, duration))
                goal_q, target, op, stroke, phase = interpolate_reference(samples, times, min(elapsed + dynamics.timestep_s, duration))
                if elapsed >= duration:
                    phase, op, stroke = "final_hold", "hold_ready", -1
                clearance = None
                if phase in ("approach", "departure", "return", "final_hold"):
                    clearance = (max(0., plan.paper.pen_lift_mm - report["path_tolerance_mm"])
                                 if op in ("approach_lower", "departure_lift") else transit.minimum_clearance_mm)
                result = advance(q, target, op, stroke, phase, clearance, goal_q)
                if result is None:
                    break
                if phase == "final_hold":
                    _, joint_speed, tip_speed = result
                    final_stable_s = final_stable_s + dynamics.timestep_s if (
                        joint_speed <= dynamics.stop_joint_speed_rad_s and tip_speed <= dynamics.stop_tip_speed_mm_s) else 0.
    except KeyboardInterrupt:
        # Freeze the simulation immediately; this is not a tested physical E-stop.
        report["status"] = "stopped"
        fail("keyboard_interrupt_simulation_aborted")
    except mujoco.FatalError as error:
        fail("mujoco_fatal_error", message=str(error))
    report.update(metrics, simulation_steps=step_count,
                  actual_simulated_duration_s=float(actual.time) if math.isfinite(actual.time) else None,
                  stop_max_displacement_mm=stop_max_displacement if stopped_at is not None else None)
    return report, model
