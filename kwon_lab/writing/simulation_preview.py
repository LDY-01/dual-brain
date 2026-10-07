"""Render either planned kinematic poses or states recorded by a dynamics rollout."""

from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from .simulation import PaperFrame, PenKinematics, ToolMount


def render_replay(model, report, output, frame_count=48):
    if report.get("format_version") != 2:
        raise ValueError("Re-run preflight: replay requires the corrected v2 page coordinate convention")
    samples = report["samples"]
    if not samples:
        raise ValueError("Cannot render a report without diagnostic samples")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    kin = PenKinematics(model, ToolMount(**report["tool"]))
    paper_frame = PaperFrame(**report["paper_frame"])
    x0, y0, x1, y1 = paper_frame.working_rect_mm
    corners = [paper_frame.world([x, y, .2]) for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0))]
    option = mujoco.MjvOption()
    option.geomgroup[3:] = 0
    option.sitegroup[1:] = 0
    traces = []
    for index, (previous, current) in enumerate(zip(samples, samples[1:]), 1):
        if current["op"] == "stroke" and previous["stroke"] == current["stroke"]:
            start, end = np.array(previous["tip_world_m"]), np.array(current["tip_world_m"])
            if np.linalg.norm(end - start) > 1e-8:
                traces.append((index, start, end))
    page_traces = [(i, paper_frame.paper_mm(a)[:2], paper_frame.paper_mm(b)[:2])
                   for i, a, b in traces]
    frames = []
    dynamic = report.get("kind") == "writing_dynamics_rollout"
    indices = np.unique(np.linspace(0, len(samples) - 1, min(frame_count, len(samples))).astype(int))
    with mujoco.Renderer(model, height=600, width=800, max_geom=max(2000, len(traces) + model.ngeom + 20)) as renderer:
        for index in indices:
            if dynamic:
                kin.data.qpos[kin.qadr] = samples[index]["arm_q_rad"]
                kin.data.qpos[kin.grip_adr] = samples[index]["gripper_q_rad"]
                mujoco.mj_forward(model, kin.data)
            else:
                kin.pose(samples[index]["arm_q_rad"])
            renderer.update_scene(kin.data, camera="writing_view", scene_option=option)
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
            for start, end in zip(corners, corners[1:]):
                geom = renderer.scene.geoms[renderer.scene.ngeom]
                mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE,
                                   np.zeros(3), np.zeros(3), np.eye(3).ravel(), np.array([.05, .55, .30, 1.]))
                mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, .0003, start, end)
                renderer.scene.ngeom += 1
            for sample_index, start, end in traces:
                if sample_index > index:
                    break
                geom = renderer.scene.geoms[renderer.scene.ngeom]
                mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE,
                                   np.zeros(3), np.zeros(3), np.eye(3).ravel(), np.array([.08, .10, .10, 1.]))
                mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, .0005, start, end)
                renderer.scene.ngeom += 1
            frame = Image.fromarray(renderer.render().copy())
            draw = ImageDraw.Draw(frame)
            draw.rectangle((0, 0, 800, 44), fill="#ffffff")
            kind = "DYNAMICS ROLLOUT" if dynamic else "KINEMATIC REPLAY"
            draw.text((14, 8), f"SO-101 | {kind} | UNCALIBRATED SIMULATION", fill="#20272c")
            phase = samples[index].get("phase", "writing")
            draw.text((14, 25), f"{report['status'].upper()} | {phase.upper()} | {index + 1}/{len(samples)} samples | no hardware / no pressure model", fill="#535d64")
            # Same solved TCP data, without the arm occluding the page. No glyph re-rendering.
            left, top = 26, 380
            scale = min(252 / (x1 - x0), 168 / (y1 - y0))
            bottom = top + (y1 - y0) * scale
            draw.rectangle((14, 345, 290, bottom + 12), fill="white", outline="#bfc9c5")
            draw.text((26, 357), "PAGE TOP VIEW | ACTUAL TCP PATH", fill="#20272c")
            draw.rectangle((left, top, left + (x1 - x0) * scale, bottom), outline="#39875e")
            for sample_index, start, end in page_traces:
                if sample_index > index:
                    break
                points = [(left + (p[0] - x0) * scale, top + (p[1] - y0) * scale) for p in (start, end)]
                draw.line(points, fill="#172326", width=2)
            frames.append(frame)
    frames[-1].save(output / "replay.png")
    frames[0].save(output / "replay.gif", save_all=True, append_images=frames[1:], duration=100, loop=0)
    return {"png": str(output / "replay.png"), "gif": str(output / "replay.gif")}
