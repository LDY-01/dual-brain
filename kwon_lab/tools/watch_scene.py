"""픽업 씬 실시간 뷰어 + XML 자동 리로드.

XML을 저장하면 0.5초 안에 감지 → 뷰어 창을 새 씬으로 다시 띄운다.
문법 오류로 저장해도 죽지 않고, 고쳐서 저장하면 이어서 뜬다.
(뷰어 창은 자식 프로세스로 분리 — 리로드 시 확실하게 닫고 새로 연다)

실행:  .venv/bin/python kwon_lab/tools/watch_scene.py [씬.xml]
종료:  뷰어 창을 직접 닫기 (또는 Ctrl+C)
"""

import subprocess
import sys
import time
from pathlib import Path

DEFAULT = Path(__file__).parent.parent / "assets" / "menagerie_so101" / "pick_scene.xml"
scene = Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT).resolve()
mjpython = Path(sys.executable).parent / "mjpython"
child_script = Path(__file__).parent / "_viewer_child.py"


def wait_for_change(last_mtime):
    while True:
        time.sleep(0.5)
        try:
            mt = scene.stat().st_mtime
            if mt != last_mtime:
                return mt
        except FileNotFoundError:
            pass


mtime = scene.stat().st_mtime
print(f"감시 시작 — {scene.name} 저장하면 자동 리로드, 뷰어 창을 닫으면 종료")

while True:
    child = subprocess.Popen([str(mjpython), str(child_script), str(scene)])
    reloaded = False
    while child.poll() is None:
        time.sleep(0.5)
        try:
            mt = scene.stat().st_mtime
        except FileNotFoundError:
            continue
        if mt != mtime:
            mtime = mt
            print("변경 감지 → 리로드!")
            child.terminate()
            child.wait()
            reloaded = True
            break

    if reloaded:
        time.sleep(0.3)  # 창 정리 여유
        continue

    if child.returncode == 0:
        print("뷰어 종료")
        break  # 사용자가 창을 직접 닫음

    # 자식이 비정상 종료 = XML 오류 등 → 고칠 때까지 대기
    print("씬 로드 실패 (위 오류 참조) — 고쳐서 저장하면 다시 시도합니다")
    mtime = wait_for_change(mtime)
    print("변경 감지 → 재시도!")
