"""로컬 VLM 물체 탐지 벤치마크 — "중간 눈" 후보 검증.

Claude(0.3Hz, 유료) 대신 로컬 VLM에게 "빨간 블록 픽셀 좌표 찍기"를 맡길 수 있는지
정량 측정한다: 탐지율 / 픽셀 오차(정답은 카메라 투영으로 자동 계산) / 실측 지연·Hz.

실행:  .venv/bin/python kwon_lab/tools/vlm_bench.py [모델명=gemma4:e4b] [장면수=8]
출력:  콘솔 표 + outputs/vlm_bench/ (장면 이미지, 정답 십자+예측 원 표시 이미지, 결과 JSON)
필요:  Ollama 서버 실행 중 (localhost:11434), vision 지원 모델
"""

import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import base64
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from envs.so101_pick_env import SO101PickEnv

OUT = Path("outputs/vlm_bench")
W, H = 640, 480
FOVY_DEG = 45.0  # front 카메라 (fovy 미지정 → MuJoCo 기본값)

PROMPT = (
    "이 이미지는 640x480 로봇 작업 공간 사진이다. 빨간 블록(작은 빨간 상자)을 찾아라. "
    '반드시 JSON만 출력: {"visible": true/false, "cx": 중심x픽셀(0~639), "cy": 중심y픽셀(0~479)}. '
    "안 보이면 visible=false, cx=cy=0."
)


def project(env, cam_name, p_world):
    """월드 점 → front 카메라 픽셀 (fovy 기반 핀홀 투영)."""
    cid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    c = env.data.cam_xpos[cid]
    R = env.data.cam_xmat[cid].reshape(3, 3)  # 열: 카메라 x(우), y(상), z(뒤)
    p = R.T @ (np.asarray(p_world) - c)       # 카메라 좌표계 (시선은 -z)
    fy = 0.5 * H / np.tan(np.deg2rad(FOVY_DEG) / 2)
    u = W / 2 + fy * (p[0] / -p[2])
    v = H / 2 - fy * (p[1] / -p[2])
    return float(u), float(v)


def ask_ollama(model, jpeg_bytes, timeout=120):
    req = urllib.request.Request(
        "http://localhost:11434/api/chat",
        data=json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": PROMPT,
                          "images": [base64.b64encode(jpeg_bytes).decode()]}],
            "stream": False, "format": "json",
            "options": {"temperature": 0},
        }).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.load(r)
    return json.loads(body["message"]["content"]), time.time() - t0


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "gemma4:e4b"
    n_scenes = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    OUT.mkdir(parents=True, exist_ok=True)
    env = SO101PickEnv()

    results = []
    for i in range(n_scenes):
        obs, info = env.reset(seed=7000 + i)
        frame = env.render()  # front 카메라
        gt_u, gt_v = project(env, "front", info["block_pos"])

        img = Image.fromarray(frame)
        from io import BytesIO
        buf = BytesIO(); img.save(buf, "JPEG", quality=90)

        try:
            pred, dt = ask_ollama(model, buf.getvalue())
        except Exception as e:
            print(f"장면 {i}: 호출 실패 — {type(e).__name__}: {e}")
            continue

        err = None
        if pred.get("visible"):
            err = float(np.hypot(pred["cx"] - gt_u, pred["cy"] - gt_v))
        results.append({"seed": 7000 + i, "gt": [round(gt_u), round(gt_v)],
                        "pred": pred, "err_px": err, "sec": round(dt, 2)})

        # 시각화: 정답=초록 십자, 예측=자홍 원
        d = ImageDraw.Draw(img)
        d.line([(gt_u - 12, gt_v), (gt_u + 12, gt_v)], fill=(0, 255, 0), width=3)
        d.line([(gt_u, gt_v - 12), (gt_u, gt_v + 12)], fill=(0, 255, 0), width=3)
        if pred.get("visible"):
            d.ellipse([pred["cx"] - 10, pred["cy"] - 10, pred["cx"] + 10, pred["cy"] + 10],
                      outline=(255, 0, 255), width=3)
        img.save(OUT / f"scene_{i}_result.jpg")

        e_txt = f"오차 {err:.0f}px" if err is not None else "미탐지"
        print(f"장면 {i}: 정답 ({gt_u:.0f},{gt_v:.0f}) → 예측 {pred} | {e_txt} | {dt:.1f}s")

    det = [r for r in results if r["err_px"] is not None]
    errs = [r["err_px"] for r in det]
    times = [r["sec"] for r in results[1:]]  # 첫 호출은 모델 로드 포함 → 제외
    print("\n" + "=" * 56)
    print(f"모델 {model} | 탐지 {len(det)}/{len(results)}")
    if errs:
        print(f"픽셀 오차: 중앙값 {np.median(errs):.0f}px, 평균 {np.mean(errs):.0f}px "
              f"(블록 폭 ≈ 30px 안팎이면 그랩 가능권)")
    if times:
        print(f"지연: 평균 {np.mean(times):.1f}s → {1 / np.mean(times):.1f}Hz "
              f"(첫 호출 {results[0]['sec']}s는 모델 로드 포함, 제외)")
    print("=" * 56)
    (OUT / "results.json").write_text(json.dumps(
        {"model": model, "results": results}, ensure_ascii=False, indent=1))
    print(f"결과: {OUT}/ (scene_*_result.jpg — 초록 십자=정답, 자홍 원=예측)")


if __name__ == "__main__":
    main()
