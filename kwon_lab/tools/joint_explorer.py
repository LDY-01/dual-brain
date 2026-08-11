"""SO-101 관절 탐색기 — 브라우저 슬라이더로 시뮬 로봇을 움직여보는 도구.

인터랙티브 MuJoCo 뷰어가 이 맥에서 크래시하는 문제(PROJECT.md 실무 정보 참조)의
우회로, 오프스크린 렌더 + 로컬 웹서버로 같은 경험을 제공한다.

실행:  .venv/bin/python kwon_lab/tools/joint_explorer.py
접속:  http://localhost:7799
"""

import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import mujoco
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from envs.so101_pick_env import load_mj_model

# 기본은 픽업 씬(블록+목표 구역). 다른 씬을 보려면 인자로 xml 경로 전달.
DEFAULT_SCENE = Path(__file__).parent.parent / "assets" / "so101" / "pick_scene.xml"
SCENE = sys.argv[1] if len(sys.argv) > 1 else str(DEFAULT_SCENE)
PORT = 7799
W, H = 960, 720

model = load_mj_model(SCENE)
model.vis.global_.offwidth, model.vis.global_.offheight = W, H
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, height=H, width=W)

cam = mujoco.MjvCamera()
mujoco.mjv_defaultFreeCamera(model, cam)
cam.distance *= 0.7  # 조금 가까이

# 팔의 회전/슬라이드 관절만 노출 (블록의 freejoint 등은 제외)
JOINTS = [
    {
        "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i),
        "lo": float(model.jnt_range[i][0]),
        "hi": float(model.jnt_range[i][1]),
        "qposadr": int(model.jnt_qposadr[i]),
    }
    for i in range(model.njnt)
    if model.jnt_type[i] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
]


def render_frame(qpos: list[float], azimuth: float, elevation: float) -> bytes:
    for j, val in zip(JOINTS, qpos):
        data.qpos[j["qposadr"]] = val
    mujoco.mj_forward(model, data)
    cam.azimuth, cam.elevation = azimuth, elevation
    renderer.update_scene(data, camera=cam)
    buf = BytesIO()
    Image.fromarray(renderer.render()).save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def build_html() -> str:
    sliders = "\n".join(
        f"""<label>{j["name"]}
  <input type="range" id="q{i}" min="{j["lo"]:.3f}" max="{j["hi"]:.3f}"
         step="0.01" value="0" oninput="update()">
  <span id="v{i}">0.00</span></label>"""
        for i, j in enumerate(JOINTS)
    )
    return f"""<!doctype html>
<meta charset="utf-8"><title>SO-101 Joint Explorer</title>
<style>
  body {{ font-family: -apple-system, sans-serif; display: flex; gap: 24px;
         padding: 20px; background: #1c1c1e; color: #eee; }}
  #panel {{ width: 300px; }}
  label {{ display: block; margin: 14px 0; font-size: 13px; }}
  input[type=range] {{ width: 100%; }}
  img {{ max-width: 100%; border-radius: 8px; }}
  h2 {{ font-size: 16px; }} .cam {{ color: #999; }}
</style>
<div id="panel">
  <h2>SO-101 관절 (모터 1→6 순서)</h2>
  {sliders}
  <h2 class="cam">카메라</h2>
  <label class="cam">azimuth
    <input type="range" id="az" min="-180" max="180" value="{cam.azimuth:.0f}" oninput="update()"></label>
  <label class="cam">elevation
    <input type="range" id="el" min="-89" max="0" value="{cam.elevation:.0f}" oninput="update()"></label>
  <button onclick="reset()">홈 자세로</button>
</div>
<div><img id="view" src="/render" width="{W}"></div>
<script>
const N = {len(JOINTS)};
let pending = false, dirty = false;
function params() {{
  const q = [];
  for (let i = 0; i < N; i++) {{
    const v = document.getElementById('q' + i).value;
    document.getElementById('v' + i).textContent = (+v).toFixed(2);
    q.push(v);
  }}
  return `/render?q=${{q.join(',')}}&az=${{az.value}}&el=${{el.value}}`;
}}
function update() {{
  if (pending) {{ dirty = true; return; }}
  pending = true;
  const img = document.getElementById('view');
  const next = new window.Image();
  next.onload = () => {{ img.src = next.src; pending = false; if (dirty) {{ dirty = false; update(); }} }};
  next.src = params() + '&t=' + Date.now();
}}
function reset() {{
  for (let i = 0; i < N; i++) document.getElementById('q' + i).value = 0;
  update();
}}
</script>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # 콘솔 스팸 방지
        pass

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/render":
            qs = parse_qs(url.query)
            qpos = [float(x) for x in qs.get("q", ["0," * (len(JOINTS) - 1) + "0"])[0].split(",")]
            az = float(qs.get("az", [cam.azimuth])[0])
            el = float(qs.get("el", [cam.elevation])[0])
            body = render_frame(qpos, az, el)
            ctype = "image/jpeg"
        else:
            body = build_html().encode()
            ctype = "text/html; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    print(f"SO-101 관절 탐색기: http://localhost:{PORT}")
    HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
