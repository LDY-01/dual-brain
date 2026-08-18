"""Role-safe OpenCV camera registration for wrist + overhead RGB."""

import json
from pathlib import Path

import cv2


CAMERA_ROLES = ("wrist", "overhead")
BACKEND_IDS = {
    "auto": cv2.CAP_ANY,
    "dshow": cv2.CAP_DSHOW,
    "msmf": cv2.CAP_MSMF,
}


def probe_camera(index, backend="dshow", width=1280, height=720, warmup=8):
    """Open one index, read a frame, and return JSON-safe diagnostics."""
    if backend not in BACKEND_IDS:
        raise ValueError(f"Unsupported camera backend: {backend}")
    capture = cv2.VideoCapture(int(index), BACKEND_IDS[backend])
    report = {
        "index": int(index),
        "backend": backend,
        "opened": bool(capture.isOpened()),
        "read_ok": False,
        "frame_shape": None,
        "reported_width": None,
        "reported_height": None,
        "reported_fps": None,
    }
    if not report["opened"]:
        capture.release()
        return report
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
    frame = None
    try:
        for _ in range(max(1, int(warmup))):
            ok, candidate = capture.read()
            if ok:
                frame = candidate
        report.update(
            {
                "read_ok": frame is not None,
                "frame_shape": list(frame.shape) if frame is not None else None,
                "reported_width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "reported_height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "reported_fps": float(capture.get(cv2.CAP_PROP_FPS)),
            }
        )
        return report
    finally:
        capture.release()


def load_camera_registry(path):
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format_version") != 1:
        raise ValueError("Unsupported camera-role config format_version")
    roles = payload.get("roles")
    if not isinstance(roles, dict):
        raise ValueError("Camera-role config must contain a roles object")
    unknown = set(roles) - set(CAMERA_ROLES)
    if unknown:
        raise ValueError(f"Unknown camera roles: {sorted(unknown)}")
    indices = []
    for role in CAMERA_ROLES:
        entry = roles.get(role)
        if entry is None:
            continue
        if not isinstance(entry, dict) or not isinstance(entry.get("index"), int):
            raise ValueError(f"Role {role!r} must be null or contain an integer index")
        if entry.get("backend", "dshow") not in BACKEND_IDS:
            raise ValueError(f"Role {role!r} has an unsupported backend")
        indices.append(entry["index"])
    if len(indices) != len(set(indices)):
        raise ValueError("Wrist and overhead roles cannot use the same camera index")
    return payload


def camera_registry_status(path, probe=True):
    """Validate role uniqueness and optionally verify current frame access."""
    payload = load_camera_registry(path)
    status = {"roles": {}, "dual_camera_ready": True}
    for role in CAMERA_ROLES:
        entry = payload["roles"].get(role)
        if entry is None:
            status["roles"][role] = {
                "registered": False,
                "available": False,
                "reason": "not_registered",
            }
            status["dual_camera_ready"] = False
            continue
        item = {
            "registered": True,
            "index": entry["index"],
            "backend": entry.get("backend", "dshow"),
            "expected_view": entry.get("expected_view"),
            "physical_usb_port": entry.get("physical_usb_port"),
        }
        if probe:
            diagnostics = probe_camera(
                entry["index"],
                backend=entry.get("backend", "dshow"),
                width=entry.get("width", 1280),
                height=entry.get("height", 720),
            )
            item["probe"] = diagnostics
            item["available"] = bool(diagnostics["read_ok"])
            if not item["available"]:
                status["dual_camera_ready"] = False
        else:
            item["available"] = None
        status["roles"][role] = item
    return status


def require_dual_camera_ready(path):
    """Fail closed before robot motion when either camera is missing/unreadable."""
    status = camera_registry_status(path, probe=True)
    if not status["dual_camera_ready"]:
        reasons = []
        for role, item in status["roles"].items():
            if not item["registered"]:
                reasons.append(f"{role}=not_registered")
            elif not item["available"]:
                reasons.append(f"{role}=unavailable")
        raise RuntimeError(
            "Dual-camera startup guard blocked robot motion: " + ", ".join(reasons)
        )
    return status
