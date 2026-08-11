"""watch_scene.py의 자식 프로세스 — 뷰어 창 하나를 담당. 직접 실행하지 말 것."""

import sys
import time

import mujoco
import mujoco.viewer

m = mujoco.MjModel.from_xml_path(sys.argv[1])  # XML 오류면 여기서 비정상 종료 → 부모가 처리
d = mujoco.MjData(m)

with mujoco.viewer.launch_passive(m, d) as v:
    while v.is_running():
        t0 = time.time()
        mujoco.mj_step(m, d)
        v.sync()
        time.sleep(max(0, m.opt.timestep - (time.time() - t0)))
