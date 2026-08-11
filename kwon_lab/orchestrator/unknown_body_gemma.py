"""미지의 몸 — Gemma 드라이버: 같은 몸, 다른 영혼 (최소 사양 측정).

unknown_body.py의 몸(모터/카메라/심판)을 그대로 재사용하고, 뇌만
Claude(opus) → gemma4:e4b(로컬 8B, Ollama 툴콜링)로 교체한다.
Claude 성공(17회, 2.6분)과 같은 조건에서 8B가 어디까지 가는지 잰다.

8B 대응 하네스:
- 슬라이딩 이미지 창: 최근 3장만 컨텍스트 유지 (구형은 텍스트로 대체)
- num_ctx 16384 명시 (Ollama 기본값이 짧음)
- think 모드 선택 (--think: 깊은 추론, 호출당 ~1분 / 기본: 빠름 ~5초)

실행:  .venv/bin/python kwon_lab/orchestrator/unknown_body_gemma.py [--lift] [--think]
"""

import json
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# 몸은 unknown_body의 것을 그대로 (env, 도구 실행부, 특권 심판, 로그)
from orchestrator import unknown_body as body

MODEL = "gemma4:e4b"
MAX_CALLS = 60
KEEP_IMAGES = 3  # 컨텍스트에 유지할 최근 이미지 수

TOOLS = [
    {"type": "function", "function": {
        "name": "motors",
        "description": ("모터 6개에 목표값을 보낸다. 1.5초에 걸쳐 이동 후 실제 도달값을 "
                        "돌려준다. 도달값이 목표와 다르면 물리적 저항(막힘)이 있었다는 뜻."),
        "parameters": {"type": "object", "required": ["v0", "v1", "v2", "v3", "v4", "v5"],
                       "properties": {k: {"type": "number"} for k in
                                      ("v0", "v1", "v2", "v3", "v4", "v5")}}}},
    {"type": "function", "function": {
        "name": "look",
        "description": "카메라 이미지를 본다. camera는 0 또는 1.",
        "parameters": {"type": "object", "required": ["camera"],
                       "properties": {"camera": {"type": "integer"}}}}},
    {"type": "function", "function": {
        "name": "done",
        "description": "실험 종료 보고. body_report에 알아낸 것 요약, touched_claim에 과제 성공 여부.",
        "parameters": {"type": "object", "required": ["body_report", "touched_claim"],
                       "properties": {"body_report": {"type": "string"},
                                      "touched_claim": {"type": "boolean"}}}}},
]


def chat(messages, think):
    req = urllib.request.Request(
        "http://localhost:11434/api/chat",
        data=json.dumps({
            "model": MODEL, "messages": messages, "tools": TOOLS,
            "stream": False, "think": think,
            "options": {"temperature": 0.2, "num_ctx": 16384},
        }).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)["message"]


def trim_images(messages):
    """오래된 이미지를 컨텍스트에서 제거 (최근 KEEP_IMAGES장만 유지)."""
    with_img = [m for m in messages if m.get("images")]
    for m in with_img[:-KEEP_IMAGES]:
        m.pop("images", None)
        m["content"] = "[이전 카메라 이미지 — 컨텍스트 절약을 위해 제거됨. 필요하면 다시 look하라]"


def main():
    lift = "--lift" in sys.argv
    think = "--think" in sys.argv
    body.LOG_DIR.mkdir(parents=True, exist_ok=True)
    body._state["log"] = body.LOG_DIR / f"gemma_{datetime.now():%Y%m%d_%H%M%S}.log"
    obs, info = body.env.reset(seed=42)
    body._state["block_start"] = info["block_pos"].copy()
    body.FRAMES.extend([obs["pixels"]] * 10)

    task = ("네 몸을 파악하고, 빨간 물체를 집어서 들어 올려라. 들어 올린 상태를 "
            "카메라로 확인해야 성공이다.") if lift else \
           "네 몸을 파악하고, 빨간 물체를 건드려라."
    body._log("과제", f"{task} (드라이버: {MODEL}, think={think})")
    print(f"[미지의 몸 · Gemma] {task} | think={think} | 예산 {MAX_CALLS}회")
    print(f"📝 로그: {body._state['log']}")

    messages = [{"role": "system", "content": body.SYSTEM},
                {"role": "user", "content": task}]
    calls, t0 = 0, time.time()
    llm_sec = 0.0

    while calls < MAX_CALLS and not body._state.get("done"):
        t1 = time.time()
        try:
            msg = chat(messages, think)
        except Exception as e:
            print(f"(Ollama 오류: {type(e).__name__}: {e} — 종료)")
            break
        llm_sec += time.time() - t1
        messages.append({k: v for k, v in msg.items() if k in
                         ("role", "content", "tool_calls", "thinking") and v})

        if msg.get("content", "").strip():
            print(f"\n[영혼-G] {msg['content'].strip()[:400]}")
            body._log("영혼", msg["content"].strip())

        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            # 도구 없이 말만 함 → 행동 재촉
            messages.append({"role": "user",
                             "content": "말이 아니라 도구를 호출하라. motors/look/done 중 하나."})
            continue

        for tc in tool_calls:
            fn = tc["function"]["name"]
            args = tc["function"].get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            calls += 1
            print(f"[{calls:2d}/{MAX_CALLS}] {fn}({json.dumps(args, ensure_ascii=False)[:80]})")
            try:
                if fn == "motors":
                    result = body.motors.call({k: float(args.get(k, 0)) for k in
                                               ("v0", "v1", "v2", "v3", "v4", "v5")})
                    messages.append({"role": "tool", "tool_name": fn, "content": result})
                elif fn == "look":
                    blocks = body.look.call({"camera": int(args.get("camera", 0))})
                    if blocks and blocks[0].get("type") == "image":
                        b64 = blocks[0]["source"]["data"]
                        messages.append({"role": "tool", "tool_name": fn,
                                         "content": "이미지가 다음 메시지에 첨부됨"})
                        messages.append({"role": "user",
                                         "content": f"[카메라 {args.get('camera')} 이미지]",
                                         "images": [b64]})
                        trim_images(messages)
                    else:
                        messages.append({"role": "tool", "tool_name": fn,
                                         "content": blocks[0].get("text", "오류")})
                elif fn == "done":
                    body.done.call({"body_report": str(args.get("body_report", "")),
                                    "touched_claim": bool(args.get("touched_claim"))})
                    messages.append({"role": "tool", "tool_name": fn, "content": "기록됨"})
                else:
                    messages.append({"role": "tool", "tool_name": fn,
                                     "content": f"알 수 없는 도구 {fn}"})
            except Exception as e:
                messages.append({"role": "tool", "tool_name": fn,
                                 "content": f"도구 오류: {type(e).__name__}: {e}"})
            if body._state.get("done"):
                break

    moved, _ = body._judge()
    st = body._state
    claim = st.get("done", {}).get("claim")
    print("\n" + "=" * 60)
    if lift:
        print(f"영혼(Gemma) 주장: 들어올림={claim} | 물리 심판: 최고 높이 "
              f"{st.get('max_h', 0) * 1000:.0f}mm → "
              f"{'진짜 들어올림 ✅' if st.get('lifted') else '못 들어올림 ❌'}")
    else:
        print(f"영혼(Gemma) 주장: 건드림={claim} | 물리 심판: 변위 {moved * 1000:.0f}mm → "
              f"{'진짜 건드림 ✅' if st.get('touched') else '못 건드림 ❌'}")
    print(f"최소 접근 {st['min_dist'] * 1000:.0f}mm | 호출 {calls}회 "
          f"(LLM {llm_sec / max(1, calls):.1f}s/회) | 전체 {(time.time() - t0) / 60:.1f}분")
    if st.get("done"):
        print(f"\n[몸 보고서]\n{st['done']['report']}")
    body._log("최종", f"심판 touched={st.get('touched')} lifted={st.get('lifted')} "
                     f"min_dist={st['min_dist']*1000:.0f}mm calls={calls}")


if __name__ == "__main__":
    main()
