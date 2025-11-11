"""Convert ONNX checkpoints to TensorRT plan files."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import tensorrt as trt

from rtmlib.tools.file import download_checkpoint


def parse_args() -> argparse.Namespace:
    """Return the CLI arguments for the plan builder."""
    parser = argparse.ArgumentParser(description="Build TensorRT plan files from ONNX models")
    parser.add_argument("onnx", help="Path or URL to the ONNX model")
    parser.add_argument(
        "--output",
        type=Path,
        help="Where to save the TensorRT engine (.plan). Default: alongside the ONNX file",
    )
    parser.add_argument(
        "--input-shape",
        type=int,
        nargs=4,
        metavar=("B", "C", "H", "W"),
        required=True,
        help="Explicit (batch, channel, height, width) dims used for optimization/profile",
    )
    parser.add_argument(
        "--workspace-gb",
        type=float,
        default=2.0,
        help="Workspace size (GB) to allocate for TensorRT builder (default: 2.0)",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=True,
        help="Enable FP16 precision (default: on)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output plan if it already exists",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Use TensorRT info-level logging",
    )
    return parser.parse_args()


def resolve_paths(onnx_arg: str, output: Path | None) -> tuple[Path, Path]:
    """Resolve the ONNX source (path or URL) and the output plan destination."""
    resolved = download_checkpoint(onnx_arg) if onnx_arg.startswith("http") else onnx_arg
    onnx_path = Path(resolved).expanduser().resolve()
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")
    if output is None:
        output = onnx_path.with_suffix(".plan")
    else:
        output = output.expanduser().resolve()
    return onnx_path, output


def _shape_tuple(values: Sequence[int]) -> tuple[int, ...]:
    """Validate the B×C×H×W tuple supplied on the command line."""
    if len(values) != 4:
        raise ValueError("Expected four integers for input shape (B C H W)")
    return tuple(int(v) for v in values)


def build_plan(
    onnx_path: Path, plan_path: Path, shape: tuple[int, ...], workspace_gb: float, fp16: bool, verbose: bool
) -> None:
    """Parse the ONNX graph and emit a serialized TensorRT engine at ``plan_path``."""
    logger = trt.Logger(trt.Logger.INFO if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    parser = trt.OnnxParser(network, logger)
    with onnx_path.open("rb") as f:
        if not parser.parse(f.read()):
            errors = [parser.get_error(i) for i in range(parser.num_errors)]
            raise RuntimeError("Failed to parse ONNX:\n" + "\n".join(str(err) for err in errors))

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1024**3)))
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    input_tensor = network.get_input(0)
    profile = builder.create_optimization_profile()
    min_shape = tuple(shape)
    opt_shape = tuple(shape)
    max_shape = tuple(shape)
    profile.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

    # Compilation can fail silently (returning ``None``) if TensorRT cannot
    # satisfy workspace/precision constraints, so fail loudly here.
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT failed to build the engine")

    plan_path.parent.mkdir(parents=True, exist_ok=True)
    with plan_path.open("wb") as f:
        f.write(serialized)


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    onnx_path, plan_path = resolve_paths(args.onnx, args.output)
    if plan_path.exists() and not args.force:
        raise FileExistsError(f"Plan file already exists: {plan_path} (use --force to overwrite)")

    shape = _shape_tuple(args.input_shape)
    build_plan(
        onnx_path=onnx_path,
        plan_path=plan_path,
        shape=shape,
        workspace_gb=args.workspace_gb,
        fp16=args.fp16,
        verbose=args.verbose,
    )
    print(f"Saved TensorRT plan -> {plan_path}")


if __name__ == "__main__":
    main()
