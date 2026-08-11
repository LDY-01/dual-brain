"""음성 콕핏 — 말로 명령하고, System 2가 음성으로 대답한다.

구성: 🎤 → mlx-whisper(로컬 STT, M4 최적화) → Claude System 2 → macOS say(TTS) → 🔊
- STT: 완전 로컬·무료. 첫 실행 시 모델(~1.5GB) 자동 다운로드
- TTS: v1은 맥 내장 유나(Yuna) 음성. v2에서 ElevenLabs로 교체 예정 (speak() 함수만 갈면 됨)

실행 (기현님 터미널에서 — 마이크 권한 필요):
    .venv/bin/mjpython kwon_lab/orchestrator/voice_cockpit.py

사용법: Enter를 누르면 녹음 시작 → 말한 뒤 1초 침묵하면 자동 종료 → 실행
"""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000
WHISPER_MODEL = "mlx-community/whisper-large-v3-turbo"
MIN_RMS = 0.003  # 이보다 조용하면 "소리 없음"으로 판정 (Whisper 환각 방지)


def flush_stdin():
    """TTS가 말하는 동안 눌린 Enter들이 다음 프롬프트를 자동 통과시키는 것 방지."""
    import termios

    try:
        termios.tcflush(sys.stdin, termios.TCIFLUSH)
    except Exception:
        pass


class Mic:
    """세션 내내 열려 있는 단일 마이크 스트림 (콜백 기반).

    녹음마다 스트림을 열고 닫으면 macOS에서 두 번째부터 무음이 오는 문제가
    있어(TTS 재생과의 오디오 라우팅 충돌, 특히 블루투스), 실전 음성 앱 방식대로
    스트림은 하나만 유지하고 캡처 여부만 플래그로 제어한다. (2026-08-10)
    """

    def __init__(self):
        # 블루투스 라우팅 문제 회피: 내장 마이크가 있으면 명시적으로 그걸 사용
        device = None
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0 and "MacBook" in d["name"]:
                device = i
                break
        self.device_name = sd.query_devices(device if device is not None else sd.default.device[0])["name"]
        self.capturing = False
        self.chunks: list = []
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32",
            device=device, callback=self._callback,
        )
        self.stream.start()

    def _callback(self, indata, frames, time_info, status):
        if self.capturing:
            self.chunks.append(indata[:, 0].copy())

    def start(self):
        self.chunks = []
        self.capturing = True

    def stop(self) -> np.ndarray:
        self.capturing = False
        if not self.chunks:
            return np.zeros(SAMPLE_RATE // 10, dtype="float32")
        return np.concatenate(self.chunks)


def mic_selftest(mic: Mic) -> bool:
    """마이크 확인 — 준비된 상태에서 말하게 하고, 실패해도 재시도/건너뛰기 가능."""
    import time

    print(f"🎙  입력 장치: {mic.device_name}")
    while True:
        flush_stdin()
        input("   [Enter] 누른 뒤 2초간 또렷하게 말해보세요 → ")
        mic.start()
        time.sleep(2.0)
        rec = mic.stop()
        rms = float(np.sqrt((rec ** 2).mean()))
        if rms > MIN_RMS:
            print(f"   입력 레벨: {rms:.4f} ✅ 정상")
            return True
        print(f"   입력 레벨: {rms:.4f} ❌ 너무 조용함")
        print("   → 다시: [Enter] / 무시하고 진행: c / 종료: q")
        ans = input("   > ").strip().lower()
        if ans == "c":
            return True
        if ans == "q":
            return False
        # 그 외(Enter 포함)는 재시도. 계속 실패하면:
        # 시스템 설정→개인정보 보호→마이크에서 터미널 앱 권한 확인 후 앱 재시작


def record_push_to_talk(mic: Mic) -> np.ndarray:
    """Enter 토글 녹음 — 지속 스트림에서 캡처만 켜고 끈다."""
    mic.start()
    flush_stdin()
    input("🎤 녹음 중 — 말이 끝나면 [Enter] ")
    return mic.stop()


def transcribe(audio: np.ndarray) -> str:
    import mlx_whisper
    result = mlx_whisper.transcribe(
        audio, path_or_hf_repo=WHISPER_MODEL, language="ko", fp16=True
    )
    return result["text"].strip()


# ── TTS v2: ElevenLabs (환경변수 ELEVENLABS_API_KEY 있으면 사용, 없으면 유나 폴백) ──
ELEVEN_VOICE_ID = "aIyfYczcAioGTbdEA7R1"  # Xunzi (기현님 선택, 2026-08-10)
ELEVEN_MODEL = "eleven_multilingual_v2"   # style(과장) 파라미터 지원 모델
ELEVEN_SETTINGS = {"style": 1.0, "stability": 0.35, "similarity_boost": 0.75, "speed": 1.2}  # 과장 100% + 최고 속도(1.2가 상한)


def _synth_and_play(text: str):
    """ElevenLabs 합성 + 재생(끝날 때까지 대기). 실패 시 유나 폴백."""
    import os
    import tempfile

    key = os.environ.get("ELEVENLABS_API_KEY")
    if key:
        try:
            import httpx

            r = httpx.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_VOICE_ID}",
                headers={"xi-api-key": key},
                json={"text": text, "model_id": ELEVEN_MODEL, "voice_settings": ELEVEN_SETTINGS},
                timeout=15,
            )
            if r.status_code == 200:
                f = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
                f.write(r.content)
                f.close()
                subprocess.run(["afplay", f.name])
                return
            print(f"(ElevenLabs 오류 {r.status_code} — 유나로 대체)")
        except Exception as e:
            print(f"(ElevenLabs 실패: {type(e).__name__} — 유나로 대체)")
    subprocess.run(["say", "-v", "Yuna", text])


# 발화 큐: 로봇이 움직이는 동안 System 2의 중계가 순서대로 흘러나온다 (자비스 모드)
import queue as _queue
import threading as _threading

_say_q: "_queue.Queue[str]" = _queue.Queue()


def _tts_worker():
    while True:
        text = _say_q.get()
        try:
            _synth_and_play(text[:250])  # 과도하게 긴 발화 방지
        finally:
            _say_q.task_done()


_threading.Thread(target=_tts_worker, daemon=True).start()


def speak(text: str):
    """비동기 발화 — 큐에 넣으면 워커가 순서대로 재생 (겹침 없음)."""
    _say_q.put(text)


def wait_for_speech():
    """진행 중·대기 중인 발화가 전부 끝날 때까지 블록 (말-행동 싱크)."""
    _say_q.join()


def main():
    # System 2 콕핏 재사용 (라이브 뷰어 포함)
    from orchestrator import system2 as s2

    # 모드 선택: 음성(STT+TTS) / 채팅(텍스트만 — 마이크·Whisper·TTS 완전 꺼짐, 빠름)
    if "--chat" in sys.argv:
        voice = False
    elif "--voice" in sys.argv:
        voice = True
    else:
        flush_stdin()
        voice = input("모드 선택 — [Enter]=음성 / c=채팅(음성 끔) > ").strip().lower() != "c"

    print("👁  에고센트릭 모드 — 세계 좌표 없음, 손목캠+상대 제어")
    s2.start_view_server()  # Claude가 보는 이미지 실시간 뷰어 (localhost:7788)
    obs, _ = s2.env.reset(seed=42)
    s2.FRAMES.extend([obs["pixels"]] * 10)
    try:
        import mujoco.viewer
        s2.env.live_viewer = mujoco.viewer.launch_passive(s2.env.model, s2.env.data)
        print("라이브 뷰어 연결됨")
    except Exception:
        print("(헤드리스 — mjpython으로 실행하면 라이브)")

    import anthropic
    client = anthropic.Anthropic()

    mic = None
    if voice:
        s2.ON_TEXT = speak  # System 2의 중간 발화도 목소리로 (자비스 모드)
        s2.WAIT_FOR_SPEECH = wait_for_speech  # 말 끝나야 다음 행동 (말-행동 싱크)
        mic = Mic()
        if not mic_selftest(mic):
            print("권한 해결 후 다시 실행해주세요.")
            return
        print("\n음성 콕핏 준비 완료. [Enter]=녹음 시작/종료 토글, q+Enter=종료")
        speak("음성 콕핏 준비 완료. 명령을 말씀하세요.")
    else:
        print("\n💬 채팅 모드 — 음성 입출력 꺼짐. 명령을 입력하세요 (q=종료)")

    while True:
        flush_stdin()
        if voice:
            key = input("\n[Enter]=녹음 시작 (q=종료) > ").strip().lower()
            if key == "q":
                break
            audio = record_push_to_talk(mic)
            rms = float(np.sqrt((audio ** 2).mean()))
            if rms <= MIN_RMS:
                print(f"(무음 감지 — 레벨 {rms:.4f}. 인식 건너뜀)")
                speak("소리가 안 들렸어요.")
                continue
            text = transcribe(audio)
            if not text:
                print("(인식된 말 없음)")
                speak("잘 못 들었어요.")
                continue
            print(f"🗣  인식: \"{text}\"")
        else:
            text = input("\n명령> ").strip()
            if text.lower() == "q":
                break
            if not text:
                continue
        s2.run_command(client, text)
        if voice:
            # 최종 요약도 페르소나 그대로 (중간 발화는 ON_TEXT로 이미 나감)
            if s2._done["summary"]:
                speak(s2._done["summary"])
            wait_for_speech()  # 말 끝난 뒤 다음 녹음 프롬프트 (목소리가 마이크에 안 섞이게)


if __name__ == "__main__":
    main()
