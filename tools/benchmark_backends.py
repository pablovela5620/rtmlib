"""Benchmark RTM pose models (2D/3D) across backends."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from rtmlib.tools.pose_estimation.rtmpose import RTMPose
from rtmlib.tools.pose_estimation.rtmpose3d import RTMPose3d


@dataclass
class BackendConfig:
    label: str
    backend: str
    device: str
    model_path: Path


def _normalize(outputs) -> Tuple[np.ndarray, ...]:
    """Convert nested backend outputs into a homogeneous tuple of float32 arrays."""
    arrays: List[np.ndarray] = []
    if isinstance(outputs, np.ndarray):
        arrays.append(outputs.astype(np.float32))
    elif isinstance(outputs, (list, tuple)):
        for item in outputs:
            if isinstance(item, (list, tuple, np.ndarray)):
                arrays.append(np.asarray(item, dtype=np.float32))
    return tuple(arrays)


def _make_pose(cfg: BackendConfig, pose_kind: str):
    """Instantiate the appropriate RTM pose wrapper for the requested backend."""
    pose_cls = RTMPose3d if pose_kind == "3d" else RTMPose
    return pose_cls(
        onnx_model=str(cfg.model_path),
        backend=cfg.backend,
        device=cfg.device,
    )


def _run_backend(
    image: np.ndarray,
    bbox: List[float],
    cfg: BackendConfig,
    pose_kind: str,
    warmup: int,
    runs: int,
    duration: float,
) -> dict:
    """Execute one backend (CPU/CUDA/TRT) and collect timing plus outputs."""
    pose = _make_pose(cfg, pose_kind)

    def infer():
        return _normalize(pose(image, bboxes=[bbox]))

    outputs: Tuple[np.ndarray, ...] = tuple()
    for _ in range(warmup):
        outputs = infer()

    if duration > 0:
        start = time.perf_counter()
        count = 0
        while time.perf_counter() - start < duration:
            outputs = infer()
            count += 1
        elapsed = time.perf_counter() - start
        used = count
    else:
        start = time.perf_counter()
        for _ in range(runs):
            outputs = infer()
        elapsed = time.perf_counter() - start
        used = runs

    fps = used / elapsed if elapsed > 0 else float("inf")
    return {
        "label": cfg.label,
        "elapsed": elapsed,
        "runs": used,
        "fps": fps,
        "outputs": outputs,
    }


def _compare(ref: Tuple[np.ndarray, ...], other: Tuple[np.ndarray, ...], rtol: float, atol: float) -> Tuple[bool, float]:
    """Check ``other`` against ``ref`` and report whether ``np.allclose`` passes."""
    if len(ref) != len(other):
        return False, float("inf")
    max_diff = 0.0
    for a, b in zip(ref, other):
        if a.shape != b.shape:
            return False, float("inf")
        if a.size == 0:
            continue
        diff = float(np.max(np.abs(a - b)))
        max_diff = max(max_diff, diff)
        if not np.allclose(a, b, rtol=rtol, atol=atol):
            return False, max_diff
    return True, max_diff


def main() -> None:
    """Benchmark entrypoint used by ``pixi run python tools/benchmark_backends.py``."""
    parser = argparse.ArgumentParser(description="Benchmark RTM pose backends on a single image.")
    parser.add_argument("--image", type=Path, default=Path("demo.jpg"))
    parser.add_argument("--pose-kind", choices=["2d", "3d"], default="3d",
                        help="Choose between the RTMW 2D model or the RTMW3D model.")
    parser.add_argument("--pose-2d-model", type=Path,
                        default=Path.home() / ".cache/rtmlib/hub/checkpoints/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.onnx",
                        help="ONNX checkpoint for the 2D RTMW model.")
    parser.add_argument("--pose-3d-model", type=Path,
                        default=Path.home() / ".cache/rtmlib/hub/checkpoints/rtmw3d-x_8xb64_cocktail14-384x288-b0a0eab7_20240626.onnx",
                        help="ONNX checkpoint for the 3D RTMW model.")
    parser.add_argument("--pose-2d-plan", type=Path,
                        default=Path.home() / ".cache/rtmlib/hub/checkpoints/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122_fp16.plan",
                        help="TensorRT plan for the 2D RTMW model (optional).")
    parser.add_argument("--pose-plan", type=Path,
                        default=Path.home() / ".cache/rtmlib/hub/checkpoints/rtmw3d-x_8xb64_cocktail14-384x288-b0a0eab7_20240626_fp16.plan",
                        help="TensorRT plan for the 3D RTMW model.")
    parser.add_argument("--pose-model", dest="pose_model_deprecated", type=Path, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=None,
                        help="Absolute tolerance for np.allclose checks. Defaults to 0.1 for 2D, 5.0 for 3D if unspecified.")
    args = parser.parse_args()

    if args.pose_model_deprecated is not None:
        args.pose_3d_model = args.pose_model_deprecated

    pose_kind = args.pose_kind.lower()
    pose_model_path = args.pose_3d_model if pose_kind == "3d" else args.pose_2d_model
    pose_model_path = pose_model_path.expanduser()

    if args.atol is None:
        atol = 5.0 if pose_kind == "3d" else 2.5
    else:
        atol = args.atol

    image = cv2.imread(str(args.image))
    if image is None:
        raise FileNotFoundError(args.image)
    bbox = [0.0, 0.0, float(image.shape[1]), float(image.shape[0])]

    configs = [
        BackendConfig("ORT-CPU", "onnxruntime", "cpu", pose_model_path),
        BackendConfig("ORT-CUDA", "onnxruntime", "cuda", pose_model_path),
    ]

    pose_plan = args.pose_plan.expanduser()
    pose_plan_2d = args.pose_2d_plan.expanduser()
    if pose_kind == "3d" and pose_plan.exists():
        configs.append(BackendConfig("TensorRT Plan", "tensorrt_plan", "cuda", pose_plan))
    if pose_kind == "2d" and pose_plan_2d.exists():
        configs.append(BackendConfig("TensorRT Plan", "tensorrt_plan", "cuda", pose_plan_2d))

    results = []
    reference: Optional[Tuple[np.ndarray, ...]] = None

    for cfg in configs:
        print(f"Benchmarking {cfg.label} ...")
        res = _run_backend(image, bbox, cfg, pose_kind, args.warmup, args.runs, args.duration)
        outputs = res.get("outputs")
        if reference is None and cfg.label == "ORT-CPU" and outputs is not None:
            reference = outputs
        elif reference is not None and outputs is not None and cfg.label != "ORT-CPU":
            ok, diff = _compare(reference, outputs, args.rtol, atol)
            res["max_diff"] = diff
            if not ok:
                raise AssertionError(f"{cfg.label} diverged from ORT-CPU (max diff {diff:.3e})")
        results.append(res)

    cpu_elapsed = next((r["elapsed"] for r in results if r.get("label") == "ORT-CPU" and "elapsed" in r), None)

    print(f"\nPose benchmark ({pose_kind.upper()} | {args.runs} runs, warmup {args.warmup}, duration {args.duration}s, atol={atol})")
    for res in results:
        if "error" in res:
            print(f"- {res['label']}: FAILED -> {res['error']}")
            continue
        speedup = None
        if cpu_elapsed and res["label"] != "ORT-CPU":
            speedup = cpu_elapsed / res["elapsed"] if res["elapsed"] > 0 else float("inf")
        line = f"- {res['label']}: {res['fps']:.2f} FPS ({res['elapsed']:.2f}s total)"
        if speedup:
            line += f" | {speedup:.2f}x vs CPU"
        if "max_diff" in res:
            line += f" | max diff {res['max_diff']:.3e}"
        print(line)


if __name__ == "__main__":
    main()
