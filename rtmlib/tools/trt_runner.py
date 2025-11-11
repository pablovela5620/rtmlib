"""TensorRT plan runner backed by PyCUDA for fixed-shape engines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pycuda.driver as cuda
import tensorrt as trt


@dataclass
class _Binding:
    name: str
    dtype: np.dtype
    shape: tuple[int, ...]
    is_input: bool
    host_mem: np.ndarray
    device_mem: int


class TensorRTPlanRunner:
    """Minimal TensorRT wrapper that runs fixed-shape engines via PyCUDA."""

    def __init__(self, plan_path: str | Path, device_id: int = 0) -> None:
        self.plan_path = Path(plan_path)
        if not self.plan_path.exists():
            raise FileNotFoundError(f"TensorRT plan not found: {self.plan_path}")

        cuda.init()
        if device_id >= cuda.Device.count():
            raise ValueError(f"CUDA device {device_id} unavailable (found {cuda.Device.count()} devices)")
        self.device = cuda.Device(device_id)
        self.cuda_context = self.device.make_context()

        # TensorRT insists on a logger instance (even if we only care about warnings),
        # so keep one around rather than letting it fall back to the global singleton.
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with self.plan_path.open("rb") as f:
            engine_data = f.read()
        self.engine = runtime.deserialize_cuda_engine(engine_data)
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {self.plan_path}")

        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.bindings: List[_Binding] = []
        try:
            self._prepare_bindings()
        finally:
            self.cuda_context.pop()

    def _push_ctx(self) -> None:
        self.cuda_context.push()

    def _pop_ctx(self) -> None:
        self.cuda_context.pop()

    def _prepare_bindings(self) -> None:
        if not hasattr(self.engine, "num_io_tensors"):
            raise RuntimeError("TensorRT runtime API v3 required. Rebuild the plan with TensorRT 10.0 or newer.")

        for idx in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(idx)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                _, opt_shape, _ = self.engine.get_tensor_profile_shape(name, 0)
                self.context.set_input_shape(name, opt_shape)
        if hasattr(self.context, "infer_shapes"):
            self.context.infer_shapes()

        for idx in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(idx)
            mode = self.engine.get_tensor_mode(name)
            dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
            shape = tuple(self.context.get_tensor_shape(name))
            if any(dim <= 0 for dim in shape):
                raise ValueError(f"Tensor {name} has invalid shape {shape}; rebuild plan with fixed shapes")
            size = int(trt.volume(shape))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(
                _Binding(name, dtype, shape, mode == trt.TensorIOMode.INPUT, host_mem, device_mem)
            )

    def run(self, inputs: List[np.ndarray]) -> List[np.ndarray]:
        self._push_ctx()
        try:
            # The engine only exposes one input (left/right are cat'd upstream), but
            # keeping this generic makes it usable for other plans too.
            input_bindings = [b for b in self.bindings if b.is_input]
            if len(inputs) != len(input_bindings):
                raise ValueError(f"Expected {len(input_bindings)} inputs, got {len(inputs)}")

            for arr, binding in zip(inputs, input_bindings):
                payload = np.ascontiguousarray(arr.astype(binding.dtype, copy=False))
                if payload.shape != binding.shape:
                    raise ValueError(f"Input {binding.name} expected shape {binding.shape}, got {payload.shape}")
                binding.host_mem[:] = payload.ravel()
                cuda.memcpy_htod_async(binding.device_mem, binding.host_mem, self.stream)

            for binding in self.bindings:
                self.context.set_tensor_address(binding.name, int(binding.device_mem))
            if not self.context.execute_async_v3(self.stream.handle):
                raise RuntimeError("TensorRT execution failed")

            outputs: List[np.ndarray] = []
            for binding in self.bindings:
                if binding.is_input:
                    continue
                cuda.memcpy_dtoh_async(binding.host_mem, binding.device_mem, self.stream)
                outputs.append(binding.host_mem.reshape(binding.shape).astype(np.float32, copy=False))

            self.stream.synchronize()
            return outputs
        finally:
            self._pop_ctx()


__all__ = ["TensorRTPlanRunner"]
