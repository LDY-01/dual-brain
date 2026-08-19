"""Interactively calibrate real overhead pixels to robot-table coordinates.

Repeat this tool for the target/table plane, the upright block top (6 cm),
and the tipped block top (4 cm). Hardware-specific output should use a
``config/*.local.json`` path so it stays out of Git.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np

from skills.block_reacquisition import pixel_to_table


PLANE_CHOICES = ("target_table", "upright_top_6cm", "tipped_top_4cm")


def load_reference_points(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    points = payload.get("points", [])
    if len(points) < 4:
        raise ValueError("At least four non-collinear calibration points are required")
    names = [str(item["name"]) for item in points]
    table_xy = np.asarray([item["table_xy_m"] for item in points], dtype=float)
    if table_xy.shape != (len(points), 2) or not np.isfinite(table_xy).all():
        raise ValueError("Every table_xy_m must contain two finite numbers")
    return payload, names, table_xy


def fit_homography(pixel_points, table_xy):
    pixel_points = np.asarray(pixel_points, dtype=float)
    table_xy = np.asarray(table_xy, dtype=float)
    if pixel_points.shape != (len(table_xy), 2) or len(table_xy) < 4:
        raise ValueError("Pixel and table points must be matching Nx2 arrays, N>=4")
    matrix, _ = cv2.findHomography(pixel_points, table_xy, method=0)
    if matrix is None or not np.isfinite(matrix).all():
        raise ValueError("Homography fit failed; check for duplicate/collinear points")
    predicted = np.asarray(
        [pixel_to_table(pixel, matrix) for pixel in pixel_points], dtype=float
    )
    errors = np.linalg.norm(predicted - table_xy, axis=1)
    return matrix, errors


def derive_parallel_plane_homography(
    table_matrix, image_size, lens_height_m, plane_height_m
):
    """Approximate a plane above the table for a near-vertical fixed camera."""
    if not 0.0 <= plane_height_m < lens_height_m:
        raise ValueError("Plane height must be in [0, lens height)")
    width, height = image_size
    camera_table_xy = np.asarray(
        pixel_to_table((width / 2.0, height / 2.0), table_matrix)
    )
    scale = (lens_height_m - plane_height_m) / lens_height_m
    plane_transform = np.array(
        [
            [scale, 0.0, (1.0 - scale) * camera_table_xy[0]],
            [0.0, scale, (1.0 - scale) * camera_table_xy[1]],
            [0.0, 0.0, 1.0],
        ]
    )
    return plane_transform @ np.asarray(table_matrix, dtype=float)


def capture_frame(camera_index, width, height, backend):
    backend_id = {
        "auto": cv2.CAP_ANY,
        "dshow": cv2.CAP_DSHOW,
        "msmf": cv2.CAP_MSMF,
    }[backend]
    capture = cv2.VideoCapture(camera_index, backend_id)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open camera index {camera_index} via {backend}")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    frame = None
    try:
        for _ in range(30):
            ok, candidate = capture.read()
            if ok:
                frame = candidate
        if frame is None:
            raise RuntimeError(f"Camera index {camera_index} returned no frames")
        return frame
    finally:
        capture.release()


def collect_clicks(frame, names):
    window = "Overhead calibration: click in listed order"
    clicks = []

    def on_mouse(event, x, y, _flags, _userdata):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < len(names):
            clicks.append((float(x), float(y)))

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    try:
        while True:
            canvas = frame.copy()
            for index, (x, y) in enumerate(clicks):
                cv2.circle(canvas, (round(x), round(y)), 7, (0, 255, 255), -1)
                cv2.putText(
                    canvas,
                    str(index + 1),
                    (round(x) + 9, round(y) - 9),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )
            next_name = names[len(clicks)] if len(clicks) < len(names) else "DONE"
            cv2.putText(
                canvas,
                f"Next: {next_name} | ENTER=fit R=reset ESC=cancel",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 0, 0),
                3,
            )
            cv2.putText(
                canvas,
                f"Next: {next_name} | ENTER=fit R=reset ESC=cancel",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                1,
            )
            cv2.imshow(window, canvas)
            key = cv2.waitKey(20) & 0xFF
            if key == 27:
                raise RuntimeError("Calibration cancelled")
            if key in (ord("r"), ord("R")):
                clicks.clear()
            if key in (10, 13) and len(clicks) == len(names):
                return np.asarray(clicks, dtype=float)
    finally:
        cv2.destroyWindow(window)


def save_calibration(args, reference_payload, names, table_xy, pixels, matrix, errors):
    if args.output.exists():
        output = json.loads(args.output.read_text(encoding="utf-8"))
    else:
        output = {"format_version": 1, "planes": {}}
    output.update(
        {
            "camera_index": args.camera_index,
            "capture_backend": args.backend,
            "requested_resolution": [args.width, args.height],
            "lens_height_m": args.lens_height_m,
            "layout_id": args.layout_id,
            "coordinate_frame": reference_payload.get("coordinate_frame"),
            "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
    )
    output.setdefault("planes", {})[args.plane] = {
        "pixel_to_table_homography": matrix.tolist(),
        "reference_points": [
            {
                "name": name,
                "pixel": pixel.tolist(),
                "table_xy_m": xy.tolist(),
                "fit_error_m": float(error),
            }
            for name, pixel, xy, error in zip(
                names, pixels, table_xy, errors, strict=True
            )
        ],
        "rmse_m": float(np.sqrt(np.mean(errors**2))),
        "max_error_m": float(errors.max()),
    }
    if args.plane == "target_table":
        for plane_name, height_m in (
            ("upright_top_6cm", 0.06),
            ("tipped_top_4cm", 0.04),
        ):
            derived = derive_parallel_plane_homography(
                matrix,
                (args.width, args.height),
                args.lens_height_m,
                height_m,
            )
            output["planes"][plane_name] = {
                "pixel_to_table_homography": derived.tolist(),
                "derived_from": "target_table",
                "plane_height_m": height_m,
                "assumption": "fixed near-vertical camera with centered principal point",
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


def run_self_test():
    truth = np.array(
        [[8.7e-4, 2.0e-5, -0.35], [1.0e-5, -8.9e-4, 0.36], [2e-5, -1e-5, 1.0]]
    )
    pixels = np.array(
        [[180, 160], [640, 120], [1100, 180], [220, 610], [640, 650], [1060, 590]],
        dtype=float,
    )
    table_xy = np.asarray([pixel_to_table(point, truth) for point in pixels])
    fitted, errors = fit_homography(pixels, table_xy)
    test_pixels = np.array([[320, 240], [800, 500], [500, 600]], dtype=float)
    max_test_error = max(
        np.linalg.norm(
            np.asarray(pixel_to_table(point, fitted))
            - np.asarray(pixel_to_table(point, truth))
        )
        for point in test_pixels
    )
    derived = derive_parallel_plane_homography(
        fitted, (1280, 720), lens_height_m=0.52, plane_height_m=0.06
    )
    center = np.asarray(pixel_to_table((640, 360), fitted))
    scale = (0.52 - 0.06) / 0.52
    max_plane_error = max(
        np.linalg.norm(
            np.asarray(pixel_to_table(point, derived))
            - (center + scale * (np.asarray(pixel_to_table(point, fitted)) - center))
        )
        for point in test_pixels
    )
    report = {
        "fit_max_error_m": float(errors.max()),
        "held_out_max_error_m": float(max_test_error),
        "derived_plane_max_error_m": float(max_plane_error),
        # OpenCV solves the homography in finite precision. One micrometre is
        # already far below the millimetre-scale accuracy required here.
        "passed": bool(
            errors.max() < 1e-6
            and max_test_error < 1e-6
            and max_plane_error < 1e-12
        ),
    }
    print(json.dumps(report, indent=2))
    return report["passed"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--backend", choices=("auto", "dshow", "msmf"), default="dshow")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--lens-height-m", type=float, default=0.52)
    parser.add_argument(
        "--layout-id",
        default="UNSET",
        help="Stable name for this physical robot/pillar/camera layout.",
    )
    parser.add_argument(
        "--points",
        type=Path,
        default=Path("config/overhead_calibration_points.example.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("config/overhead_camera_calibration.local.json"),
    )
    parser.add_argument("--plane", choices=PLANE_CHOICES, default="target_table")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        raise SystemExit(0 if run_self_test() else 1)

    reference, names, table_xy = load_reference_points(args.points)
    if args.image:
        frame = cv2.imread(str(args.image))
        if frame is None:
            raise SystemExit(f"Cannot read image: {args.image}")
    else:
        frame = capture_frame(
            args.camera_index, args.width, args.height, args.backend
        )
    print("Click points in this exact order:")
    for index, (name, xy) in enumerate(zip(names, table_xy, strict=True), start=1):
        print(f"  {index}. {name}: XY={xy.tolist()} m")
    pixels = collect_clicks(frame, names)
    matrix, errors = fit_homography(pixels, table_xy)
    save_calibration(args, reference, names, table_xy, pixels, matrix, errors)
    print(f"Saved {args.plane} calibration to {args.output}")
    print(f"RMSE={np.sqrt(np.mean(errors**2))*1000:.2f} mm, max={errors.max()*1000:.2f} mm")
    if errors.max() > 0.010:
        print("WARNING: max calibration error exceeds 10 mm; repeat the clicks/measurements.")


if __name__ == "__main__":
    main()
