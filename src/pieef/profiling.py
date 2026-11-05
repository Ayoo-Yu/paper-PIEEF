from __future__ import annotations

import platform
import statistics
import time
from typing import Any, Callable

import torch


def count_parameters(model: torch.nn.Module, trainable_only: bool = False) -> int:
    """Count distinct model parameters, optionally excluding frozen parameters."""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if not trainable_only or parameter.requires_grad
    )


def profile_predictor(
    predict: Callable[[], Any],
    device: torch.device | str,
    warmup: int = 10,
    repeats: int = 100,
) -> dict[str, Any]:
    """Measure one forecast-origin callable on its actual CPU or CUDA device.

    The caller supplies the real inputs and puts any neural model in evaluation
    mode. No model modes, parameters, inputs, or outputs are changed here.
    CUDA memory is the allocator's peak allocated memory, including resident
    tensors and model weights. CPU memory is left unspecified.
    """
    if not callable(predict):
        raise TypeError("predict must be callable")
    if not isinstance(warmup, int) or isinstance(warmup, bool) or warmup < 0:
        raise ValueError("warmup must be a non-negative integer")
    if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    actual_device = torch.device(device)
    if actual_device.type not in {"cpu", "cuda"}:
        raise ValueError("Profiling supports CPU and CUDA devices")

    is_cuda = actual_device.type == "cuda"
    if is_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA profiling requested but CUDA is unavailable")
        index = actual_device.index
        if index is None:
            index = torch.cuda.current_device()
        if index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device index {index} is unavailable")
        actual_device = torch.device("cuda", index)
        hardware_identifier = torch.cuda.get_device_name(actual_device)
    else:
        hardware_identifier = platform.processor() or platform.machine() or "CPU"

    def synchronize() -> None:
        if is_cuda:
            torch.cuda.synchronize(actual_device)

    latencies_ms = []
    with torch.inference_mode():
        for _ in range(warmup):
            predict()
        synchronize()
        if is_cuda:
            torch.cuda.reset_peak_memory_stats(actual_device)
        for _ in range(repeats):
            synchronize()
            start_ns = time.perf_counter_ns()
            predict()
            synchronize()
            latencies_ms.append((time.perf_counter_ns() - start_ns) / 1_000_000.0)

    return {
        "latency_mean_ms": statistics.fmean(latencies_ms),
        "latency_std_ms": statistics.pstdev(latencies_ms),
        "peak_allocated_mb": torch.cuda.max_memory_allocated(actual_device) / 1_000_000.0 if is_cuda else None,
        "device": str(actual_device),
        "hardware_identifier": hardware_identifier,
        "warmup": warmup,
        "repeats": repeats,
    }
