"""Role-safe OpenCV camera registration for wrist + overhead RGB."""

import json
import platform
import shutil
import subprocess
from pathlib import Path

import cv2


CAMERA_ROLES = ("wrist", "overhead")
BACKEND_IDS = {
    "auto": cv2.CAP_ANY,
    "dshow": cv2.CAP_DSHOW,
    "msmf": cv2.CAP_MSMF,
}


def capture_camera_frame(index, backend="dshow", width=1280, height=720, warmup=8):
    """Open one index and return its last warm frame plus diagnostics."""
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
        return None, report
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
        return frame, report
    finally:
        capture.release()


def probe_camera(index, backend="dshow", width=1280, height=720, warmup=8):
    """Open one index, read a frame, and return JSON-safe diagnostics."""
    _, report = capture_camera_frame(index, backend, width, height, warmup)
    return report


def enumerate_windows_camera_devices():
    """Return stable Windows PnP identifiers when the operating system exposes them.

    OpenCV indices are not stable across USB reconnects. PnP instance IDs are
    therefore recorded as an additional audit signal, but visual confirmation
    remains mandatory because Windows does not expose a reliable index mapping.
    """
    if platform.system() != "Windows":
        return {"supported": False, "devices": [], "reason": "not_windows"}
    shell = shutil.which("pwsh") or shutil.which("powershell")
    if shell is None:
        return {"supported": False, "devices": [], "reason": "powershell_missing"}
    command = (
        "[Console]::OutputEncoding=[Text.Encoding]::UTF8; "
        "Get-CimInstance Win32_PnPEntity | "
        "Where-Object { $_.PNPClass -in @('Camera','Image') } | "
        "Select-Object Name,PNPClass,DeviceID,Status | ConvertTo-Json -Compress"
    )
    try:
        completed = subprocess.run(
            [shell, "-NoProfile", "-Command", command],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=10,
        )
        raw = completed.stdout.strip()
        payload = [] if not raw else json.loads(raw)
        if isinstance(payload, dict):
            payload = [payload]
        devices = [
            {
                "name": item.get("Name"),
                "pnp_class": item.get("PNPClass"),
                "device_instance_id": item.get("DeviceID"),
                "status": item.get("Status"),
            }
            for item in payload
        ]
        return {"supported": True, "devices": devices, "reason": None}
    except (subprocess.SubprocessError, json.JSONDecodeError, OSError) as exc:
        return {
            "supported": False,
            "devices": [],
            "reason": f"enumeration_failed:{type(exc).__name__}",
        }


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
    device_ids = []
    for role in CAMERA_ROLES:
        entry = roles.get(role)
        if entry is None:
            continue
        if not isinstance(entry, dict) or not isinstance(entry.get("index"), int):
            raise ValueError(f"Role {role!r} must be null or contain an integer index")
        if entry.get("backend", "dshow") not in BACKEND_IDS:
            raise ValueError(f"Role {role!r} has an unsupported backend")
        indices.append(entry["index"])
        device_id = entry.get("device_instance_id")
        if device_id is not None:
            if not isinstance(device_id, str) or not device_id.strip():
                raise ValueError(f"Role {role!r} has an invalid device_instance_id")
            device_ids.append(device_id.casefold())
    if len(indices) != len(set(indices)):
        raise ValueError("Wrist and overhead roles cannot use the same camera index")
    if len(device_ids) != len(set(device_ids)):
        raise ValueError("Wrist and overhead roles cannot use the same PnP device")
    return payload


def camera_registry_status(path, probe=True, include_devices=True):
    """Validate role uniqueness and optionally verify current frame access."""
    payload = load_camera_registry(path)
    devices = (
        enumerate_windows_camera_devices()
        if include_devices
        else {"supported": False, "devices": [], "reason": "not_requested"}
    )
    present_ids = {
        str(item.get("device_instance_id", "")).casefold()
        for item in devices["devices"]
    }
    status = {
        "roles": {},
        "dual_camera_ready": True,
        "windows_camera_devices": devices,
    }
    require_identity = payload.get("startup_policy", {}).get(
        "require_view_confirmation_after_usb_change", True
    )
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
            "device_name": entry.get("device_name"),
            "device_instance_id": entry.get("device_instance_id"),
        }
        expected_id = entry.get("device_instance_id")
        if expected_id is not None:
            item["pnp_device_present"] = expected_id.casefold() in present_ids
            if devices["supported"] and not item["pnp_device_present"]:
                status["dual_camera_ready"] = False
        else:
            item["pnp_device_present"] = None
        item["device_identity_complete"] = bool(
            expected_id
            and entry.get("physical_usb_port")
            and entry.get("physical_usb_port") != "unassigned"
            and entry.get("confirmation") == "user_visually_confirmed"
        )
        if require_identity and not item["device_identity_complete"]:
            status["dual_camera_ready"] = False
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
            elif item.get("pnp_device_present") is False:
                reasons.append(f"{role}=registered_pnp_device_missing")
            elif not item.get("device_identity_complete"):
                reasons.append(f"{role}=device_identity_incomplete")
        raise RuntimeError(
            "Dual-camera startup guard blocked robot motion: " + ", ".join(reasons)
        )
    return status
