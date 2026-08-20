"""Probe and register wrist/overhead cameras without assuming stable indices."""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from hardware.camera_roles import (
    CAMERA_ROLES,
    camera_registry_status,
    capture_camera_frame,
    enumerate_windows_camera_devices,
    load_camera_registry,
    probe_camera,
)


DEFAULT_CONFIG = Path("config/real_camera_roles.local.json")


def empty_registry():
    return {
        "format_version": 1,
        "roles": {"wrist": None, "overhead": None},
        "startup_policy": {
            "require_distinct_indices": True,
            "require_view_confirmation_after_usb_change": True,
            "block_robot_motion_when_incomplete": True,
        },
    }


def save_registry(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--probe", action="store_true")
    action.add_argument("--devices", action="store_true")
    action.add_argument("--status", action="store_true")
    action.add_argument("--register-role", choices=CAMERA_ROLES)
    action.add_argument("--clear-role", choices=CAMERA_ROLES)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--camera-index", type=int)
    parser.add_argument("--max-index", type=int, default=5)
    parser.add_argument(
        "--skip-index",
        type=int,
        action="append",
        default=[],
        help="Camera index to leave unopened; repeat for multiple indices.",
    )
    parser.add_argument("--backend", choices=("auto", "dshow", "msmf"), default="dshow")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--snapshot-dir", type=Path)
    parser.add_argument("--physical-usb-port", default="unassigned")
    parser.add_argument("--device-instance-id")
    parser.add_argument(
        "--confirm-view",
        help="For registration, type WRIST or OVERHEAD after visually checking the feed.",
    )
    args = parser.parse_args()

    if args.devices:
        print(json.dumps(enumerate_windows_camera_devices(), indent=2))
        return

    if args.probe:
        reports = []
        for index in range(args.max_index + 1):
            if index in set(args.skip_index):
                continue
            frame, report = capture_camera_frame(
                index,
                backend=args.backend,
                width=args.width,
                height=args.height,
                warmup=3,
            )
            if not report["opened"]:
                continue
            if args.snapshot_dir is not None and frame is not None:
                import cv2

                args.snapshot_dir.mkdir(parents=True, exist_ok=True)
                snapshot = args.snapshot_dir / f"camera_index_{index}.png"
                if not cv2.imwrite(str(snapshot), frame):
                    raise SystemExit(f"Failed to write snapshot: {snapshot}")
                report["snapshot"] = str(snapshot.resolve())
            reports.append(report)
        if args.snapshot_dir is not None:
            manifest = {
                "opencv_cameras": reports,
                "windows_camera_devices": enumerate_windows_camera_devices(),
                "warning": "OpenCV index to PnP device mapping still requires visual confirmation.",
            }
            (args.snapshot_dir / "probe_manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )
        print(json.dumps(reports, indent=2))
        return

    if args.status:
        print(json.dumps(camera_registry_status(args.config), indent=2))
        return

    payload = (
        load_camera_registry(args.config)
        if args.config.exists()
        else empty_registry()
    )
    if args.clear_role:
        payload["roles"][args.clear_role] = None
        save_registry(args.config, payload)
        print(f"Cleared {args.clear_role} registration in {args.config}")
        return

    role = args.register_role
    if args.camera_index is None:
        raise SystemExit("--camera-index is required with --register-role")
    if args.confirm_view != role.upper():
        raise SystemExit(
            f"Refusing registration: visually check the feed and pass --confirm-view {role.upper()}"
        )
    diagnostics = probe_camera(
        args.camera_index,
        backend=args.backend,
        width=args.width,
        height=args.height,
    )
    if not diagnostics["read_ok"]:
        raise SystemExit(f"Camera probe failed: {json.dumps(diagnostics)}")
    windows_devices = enumerate_windows_camera_devices()
    if windows_devices["supported"] and not args.device_instance_id:
        raise SystemExit(
            "Windows PnP device identity is required. Run --devices, identify the camera "
            "by unplugging one at a time if necessary, then pass --device-instance-id."
        )
    if not args.physical_usb_port or args.physical_usb_port == "unassigned":
        raise SystemExit("Assign and label a physical USB port with --physical-usb-port")
    selected_device = None
    if args.device_instance_id:
        selected_device = next(
            (
                item
                for item in windows_devices["devices"]
                if item["device_instance_id"].casefold()
                == args.device_instance_id.casefold()
            ),
            None,
        )
        if windows_devices["supported"] and selected_device is None:
            raise SystemExit("--device-instance-id is not present in Windows camera devices")
    other_role = "overhead" if role == "wrist" else "wrist"
    other = payload["roles"].get(other_role)
    if other is not None and other["index"] == args.camera_index:
        raise SystemExit(
            f"Camera index {args.camera_index} is already registered as {other_role}"
        )
    payload["roles"][role] = {
        "index": args.camera_index,
        "backend": args.backend,
        "width": diagnostics["reported_width"],
        "height": diagnostics["reported_height"],
        "expected_view": "gripper_and_object" if role == "wrist" else "top_down_workspace",
        "physical_usb_port": args.physical_usb_port,
        "device_name": selected_device["name"] if selected_device else None,
        "device_instance_id": args.device_instance_id,
        "registered_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "confirmation": "user_visually_confirmed",
    }
    save_registry(args.config, payload)
    print(json.dumps(camera_registry_status(args.config), indent=2))


if __name__ == "__main__":
    main()
