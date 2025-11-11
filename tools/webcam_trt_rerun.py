"""Webcam TensorRT demo that streams detections/keypoints to Rerun."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from rtmlib import PoseTracker, Wholebody
from rtmlib.tools.object_detection.yolox import YOLOX
from rtmlib.tools.pose_estimation.rtmpose3d import RTMPose3d


def parse_args() -> argparse.Namespace:
    """Return parsed CLI arguments for the webcam demo."""
    default_root = Path.home() / ".cache" / "rtmlib" / "hub" / "checkpoints"
    parser = argparse.ArgumentParser(description="TensorRT webcam demo streaming to Rerun")
    parser.add_argument("--det-model", type=Path,
                        default=default_root / "yolox_m_8xb8-300e_humanart-c2c7a14a.onnx",
                        help="Path to YOLOX ONNX model")
    parser.add_argument("--pose-plan", type=Path,
                        default=default_root / "rtmw3d-x_8xb64_cocktail14-384x288-b0a0eab7_20240626_fp16.plan",
                        help="Path to RTMW3D TensorRT plan (used when --pose-mode=3d)")
    parser.add_argument("--pose-model", type=Path,
                        default=default_root / "rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.onnx",
                        help="Wholebody 2D ONNX path (used when --pose-mode=2d)")
    parser.add_argument("--pose-mode", choices=["3d", "2d"], default="3d",
                        help="Select pose solution: 3d (Wholebody3d) or 2d (Wholebody)")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index")
    parser.add_argument("--max-det", type=int, default=8, help="Maximum detections per frame")
    parser.add_argument("--kpt-thr", type=float, default=0.3, help="Keypoint confidence threshold")
    parser.add_argument("--max-seconds", type=float, default=120.0, help="Stop after this many seconds (0 = unlimited)")
    return parser.parse_args()


def log_boxes(boxes: np.ndarray, ratio: float) -> None:
    """Log detector outputs into the Rerun timeline as 2D boxes."""
    if boxes.size == 0:
        return
    mins = boxes[:, :2]
    sizes = boxes[:, 2:4] - boxes[:, :2]
    rr.log(
        "world/detections",
        rr.Boxes2D(mins=mins * ratio, sizes=sizes * ratio, colors=[0, 255, 0], labels=["person"] * len(boxes)),
    )


def log_keypoints(kpts: np.ndarray, scores: np.ndarray, thr: float) -> None:
    """Push 2D keypoints to Rerun after thresholding low-score entries."""
    if kpts is None or scores is None:
        return
    mask = scores > thr
    if not np.any(mask):
        return
    pts = kpts[mask]
    rr.log(
        "world/keypoints2d",
        rr.Points2D(positions=pts, colors=[255, 0, 0], radii=2.0),
    )


def main() -> None:
    """Run the end-to-end webcam loop (detector + pose + Rerun logging)."""
    args = parse_args()
    det_model = args.det_model.expanduser()
    pose_plan = args.pose_plan.expanduser()
    pose_model = args.pose_model.expanduser()
    if not det_model.exists():
        raise FileNotFoundError(f"Detector ONNX not found: {det_model}")
    if args.pose_mode == "3d" and not pose_plan.exists():
        raise FileNotFoundError(f"Pose plan not found: {pose_plan}")
    if args.pose_mode == "2d" and not pose_model.exists():
        raise FileNotFoundError(f"Pose ONNX not found: {pose_model}")

    rr.init("rtmlib-webcam-trt", spawn=True)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
    blueprint = rrb.Blueprint(
        rrb.Vertical(
            contents=[
                rrb.Horizontal(
                    contents=[
                        rrb.Spatial2DView(name="Camera", origin="world/camera"),
                        rrb.Spatial2DView(name="Detections", origin="world"),
                    ]
                ),
                rrb.TimeSeriesView(name="Timings", origin="timings"),
            ]
        )
    )
    rr.send_blueprint(blueprint)

    detector = YOLOX(str(det_model), backend="onnxruntime", device="cuda")
    if args.pose_mode == "3d":
        poser = RTMPose3d(str(pose_plan), backend="tensorrt_plan", device="cuda")
    else:
        poser = PoseTracker(
            Wholebody,
            det_frequency=1,
            tracking=False,
            backend="onnxruntime",
            device="cuda",
            pose=str(pose_model),
        )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {args.camera}")

    prev = time.perf_counter()
    start_run = prev
    frame_idx = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_idx += 1
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        det_start = time.perf_counter()
        boxes = detector(frame_rgb)
        det_time = time.perf_counter() - det_start
        boxes = np.array(boxes[:args.max_det], dtype=np.float32)

        pose_time = 0.0
        kpts2d = None
        scores = None
        if boxes.size:
            pose_start = time.perf_counter()
            result = poser(frame_rgb, bboxes=boxes.tolist())
            pose_time = time.perf_counter() - pose_start
            if args.pose_mode == "3d":
                _, score3d, _, kpts2d = result
                scores = score3d
            else:
                kpts2d, scores = result

        now = time.perf_counter()
        fps = 1.0 / (now - prev)
        prev = now

        rr.set_time(timeline="frame", timestamp=now)
        rr.log("world/camera", rr.Image(frame_rgb))
        log_boxes(boxes, ratio=1.0)
        if kpts2d is not None and scores is not None:
            if kpts2d.ndim == 3:
                pts2d = kpts2d.reshape(-1, 2)
                scr = scores.reshape(-1)
            else:
                pts2d = kpts2d
                scr = scores if scores.ndim == 1 else scores.reshape(-1)
            log_keypoints(pts2d, scr, args.kpt_thr)

        rr.log("timings/fps", rr.TextLog(f"fps={fps:.1f}"))
        rr.log("timings/det", rr.TextLog(f"det={det_time*1000:.1f} ms"))
        rr.log("timings/pose", rr.TextLog(f"pose={pose_time*1000:.1f} ms"))

        if cv2.waitKey(1) == ord('q'):
            break
        if args.max_seconds > 0 and (time.perf_counter() - start_run) > args.max_seconds:
            break

    cap.release()


if __name__ == "__main__":
    main()
