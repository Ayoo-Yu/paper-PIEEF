"""Measure completed epoch work separately from validation and artifact I/O."""
from __future__ import annotations

from datetime import datetime, timezone
import math
import os
from pathlib import Path
import subprocess
import time

import torch

from .artifacts import write_json_atomic


class TrainingTimer:
    def __init__(self, device, *, clock=None):
        self.device = torch.device(device)
        self.clock = clock or time.perf_counter
        self.records = []
        self.active = None

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def start(self, phase, stage, epoch):
        if self.active is not None or phase not in {"train", "validation"}:
            raise ValueError("Training timing scopes must be sequential train/validation phases")
        self._sync()
        self.active = (phase, str(stage), int(epoch), self.clock())

    def stop(self, batches):
        if self.active is None:
            raise ValueError("No active training timing scope")
        self._sync()
        phase, stage, epoch, started = self.active
        elapsed = self.clock() - started
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("Invalid measured epoch duration")
        self.records.append(dict(phase=phase, stage=stage, epoch=epoch,
                                 batches=int(batches), wall_clock_s=float(elapsed)))
        self.active = None
        return elapsed

    def result(self):
        if self.active is not None:
            raise ValueError("An incomplete timing scope cannot be published")
        return {
            "version": 1, "device": str(self.device), "epochs": list(self.records),
            "train_only_s": sum(row["wall_clock_s"] for row in self.records if row["phase"] == "train"),
            "validation_only_s": sum(row["wall_clock_s"] for row in self.records if row["phase"] == "validation"),
            "scope": "Epoch data loading, transfers and forward/backward/optimizer work; validation separate. Model/optimizer setup, checkpoint I/O and final evaluations excluded.",
            "cuda_synchronized_boundaries": self.device.type == "cuda",
        }


def gpu_runtime(device):
    device = torch.device(device)
    result = {"torch": str(torch.__version__), "cuda_runtime": str(torch.version.cuda),
              "device": str(device), "gpu": None, "gpu_uuid": None, "driver_version": None}
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        result.update(gpu=properties.name, gpu_memory_bytes=int(properties.total_memory))
        try:
            index = device.index if device.index is not None else torch.cuda.current_device()
            line = subprocess.run(["nvidia-smi", "-i", str(index), "--query-gpu=uuid,driver_version",
                                   "--format=csv,noheader,nounits"], check=True, capture_output=True,
                                  text=True, timeout=10).stdout.strip()
            uuid, driver = [value.strip() for value in line.split(",")]
            result.update(gpu_uuid=uuid, driver_version=driver)
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
    return result


def save_training_timing(run_dir, model, cfg, seed, horizon, device):
    timing = getattr(model, "training_timing", None)
    if timing is None:
        raise ValueError("Completed neural training lacks measured epoch timing")
    run_dir = Path(run_dir)
    payload = dict(timing, status="completed", run_id=run_dir.name,
                   config_id=cfg["tracking"]["config_id"], model=cfg["experiment"]["model"],
                   seed=int(seed), horizon=int(horizon), pid=os.getpid(),
                   worker_id=cfg["tracking"].get("worker_id"), runtime=gpu_runtime(device),
                   recorded_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    write_json_atomic(run_dir / "training_timing.json", payload)
    return payload
