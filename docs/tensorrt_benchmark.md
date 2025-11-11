# TensorRT Benchmark Notes

## Current Scope

| Component (demo.jpg) | ORT-CPU | ORT-CUDA | TensorRT Plan |
|----------------------|---------|----------|----------------|
| RTMW3D pose (1×3×384×288) | 13.9 FPS | 126.5 FPS (~9×) | 361.0 FPS (~26×) |
| RTMW 2D pose (1×3×384×288) | 20.3 FPS | 170.0 FPS (~8.4×) | 460.4 FPS (~22.7×) |

- Single-image benchmarks use `demo.jpg` and a full-frame bbox. Heavy-load runs (120 s duration) keep the same FPS ordering (CPU 19.7 FPS, CUDA 166.9 FPS, TRT 464.4 FPS).
- Outputs stay within the `np.allclose` guard rails relative to ORT‑CPU (≤ ~2 px deviation for 2D, ≤ ~5 px/mm for 3D) before logging results.
- Commands:
  - 3D pose (default): `pixi run python tools/benchmark_backends.py --pose-kind 3d --image demo.jpg --runs 10`
  - 2D pose: `pixi run python tools/benchmark_backends.py --pose-kind 2d --image demo.jpg --runs 20`
  - Long-duration soak: `pixi run python tools/benchmark_backends.py --pose-kind 2d --duration 120`

## Detector Status (YOLOX)

- The public ONNX includes TopK/NMS with `-1` dimensions in multiple reshape constants (tensors 637/1305/1329/1351). TensorRT 10 requires those to be fixed-size.
- Clamping `TopK` to 1024 (via `tools/patch_topk.py`) enables plan building but runtime still fails because buffer shapes stay dynamic; Myelin reports `CUDA error 400` when it hits the `CastConc` kernel.
- ONNXRuntime symbolic shape inference does not resolve these `-1`s without a full re-export.

> Note: `tools/benchmark_full.py` has been dropped for now so that the benchmarking surface matches the supported scope (pose-only). Re-add it once YOLOX has a stable static export and TensorRT backend again.

## Paths Forward

1. **Fold dynamic reshapes into constants**
   - Use Polygraphy / ONNX Graph Surgeon to evaluate the subgraph that produces the reshape tensors. With `k=1024`, replace each `-1` with its computed value so the volumes match (e.g., derive from the upstream tensor size divided by the known factors).
   - After folding, rebuild the FP16 plan (`tools/build_trt_plan.py`). The existing TensorRTPlanRunner already supports v3 engines.

2. **Truncate YOLOX graph (DONE)**
   - `tools/extract_yolox_head.py` rewires the graph outputs to tensors `1436` (decoded boxes) and `1438` (class scores). `tools/build_trt_plan.py` then produces a static FP16 plan (`…_head_fp16.plan`).
   - `YOLOX` now detects this two-output format and performs NMS in Python, so ORT-CPU, ORT-CUDA, and TensorRT plan all run the same NumPy post-processing.

3. **Detector status (YOLOX)**
   - TensorRT experiments were retired. Dynamic TopK/reshape nodes made the head-only plan fragile, and the native ONNXRuntime path proved stable enough. Detector benchmarking has been removed from the default suite; reintroduce it only after exporting a fully static graph.

## YOLOX TensorRT (Retired)

- Attempted approach: clamp `TopK` to 1024, expose head outputs, rebuild FP16 plans, and run post-processing in Python.
- Practical issues: TensorRT EP still recompiles the original graph (fails on large `TopK`), maintaining duplicate ONNX/plan variants adds complexity, and the end-to-end FPS is still limited by the detector even under TRT.
- Decision: keep YOLOX on ONNX Runtime (CPU/CUDA) until we have a fully static export; focus optimizations on the pose head where TRT provides clear wins.
