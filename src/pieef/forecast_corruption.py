"""F2: masked-history corruption on the Stage-3 forecast loss path only.

Every batch is corrupted with probability 0.5. Inside a corrupted batch each sample
draws its own anchor turbine, block length, block position and channel set, following
附录 C.4 of `第三轮统一实验方案_20261019.md`. Injection always goes through
`pieef.faults.apply_fault`, so the training-time corruption and the evaluation-time
fault matrix share one definition of "missing" (value, explicit mask channel and
causal forward fill).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .faults import apply_fault

POLICIES = ("off", "f2_block")
BATCH_PROBABILITY = 0.5
TURBINE_RANGE = (1, 10)
BLOCK_RANGE = (4, 96)
CHANNELS = ("power", "both")


@dataclass
class ForecastCorruption:
    """Independent corruption stream; it never draws from the global torch RNG."""

    policy: str
    coords: object
    scada_wind_std: float
    seed: int
    num_turbines: int
    seq_len: int
    generator: np.random.Generator = field(init=False, repr=False)
    counts: dict = field(init=False, default_factory=dict)
    last_scenarios: list = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        if self.policy not in POLICIES:
            raise ValueError("stage3_forecast_corruption must be off or f2_block")
        if self.num_turbines < 1 or self.seq_len < 1:
            raise ValueError("F2 corruption requires a positive turbine count and history length")
        # A dedicated stream: adding F2 must not move the data order or any initialization.
        self.generator = np.random.default_rng(int(self.seed))
        self.coords = np.asarray(self.coords, dtype=np.float64)
        if self.coords.shape[0] != self.num_turbines or self.coords.ndim != 2 or self.coords.shape[1] < 2:
            raise ValueError("F2 corruption requires finite [turbines, >=2] coordinates")
        if not np.isfinite(self.coords).all():
            raise ValueError("F2 corruption requires finite turbine coordinates")
        self.counts = {"batches": 0, "corrupted_batches": 0, "corrupted_samples": 0}
        # Distance-sorted turbine order per anchor, computed once. It reproduces
        # fault_jobs._nearest_turbines(anchor, count) for every count without recomputing
        # the distance matrix inside the training loop.
        from .faults import _nearest_turbines
        self.neighbor_order = [
            _nearest_turbines(self.coords, self.num_turbines, anchor, self.num_turbines)
            for anchor in range(self.num_turbines)
        ]

    def scenario(self, sample_id) -> dict:
        """Draw one continuous corruption pattern (never a fixed test scenario)."""
        turbines = min(max(TURBINE_RANGE[0], 1), self.num_turbines)
        high = min(TURBINE_RANGE[1], self.num_turbines)
        count = int(self.generator.integers(turbines, high + 1)) if high >= turbines else turbines
        low_block = min(BLOCK_RANGE[0], self.seq_len)
        high_block = min(BLOCK_RANGE[1], self.seq_len)
        length = int(self.generator.integers(low_block, high_block + 1))
        # Half of the blocks end flush against the forecast origin, half float freely.
        if self.generator.random() < 0.5 or length >= self.seq_len:
            start = self.seq_len - length
        else:
            start = int(self.generator.integers(0, self.seq_len - length + 1))
        channels = CHANNELS[0] if self.generator.random() < 0.5 else CHANNELS[1]
        return dict(
            kind="missing", channels=channels, turbine_count=count,
            duration_steps=length, start_index=start,
            anchor_turbine=int(self.generator.integers(0, self.num_turbines)),
            sample_ids=[sample_id],
        )

    def corrupt(self, batch):
        """Return (batch, power_mask, wind_mask); identities when the batch is skipped."""
        if self.policy != "f2_block":
            raise ValueError("F2 corruption is only defined for the f2_block policy")
        x_p, x_w, x_hist_nwp, x_target_nwp, target = batch
        if x_p.ndim != 4 or x_p.shape[-1] != 1 or x_w.shape != x_p.shape:
            raise ValueError("F2 corruption needs equal [batch, turbines, seq_len, 1] SCADA inputs")
        bsz = x_p.shape[0]
        if x_p.shape[1] != self.num_turbines or x_p.shape[2] != self.seq_len:
            raise ValueError("F2 corruption window does not match the configured turbines/seq_len")
        self.counts["batches"] += 1
        if self.generator.random() >= BATCH_PROBABILITY:
            self.last_scenarios = []
            return batch, None, None
        power, wind = x_p.clone(), x_w.clone()
        p_mask = torch.zeros_like(x_p)
        w_mask = torch.zeros_like(x_w)
        scenarios = [self.scenario(f"f2:{self.counts['batches']}:{index}") for index in range(bsz)]
        # Vectorised assembly, identical in semantics to faults.apply_fault for a 'missing'
        # SCADA fault: the block is replaced by the causal value at the block start, the
        # explicit missing channel is flagged, and no other value moves.
        for index, scenario in enumerate(scenarios):
            anchor = scenario["anchor_turbine"]
            count = scenario["turbine_count"]
            selected = self.neighbor_order[anchor][:count]
            sel = torch.as_tensor(selected, dtype=torch.long, device=x_p.device)
            start = scenario["start_index"]
            length = scenario["duration_steps"]
            end = start + length
            channels = scenario["channels"]

            def _causal_vector(series: torch.Tensor) -> torch.Tensor:
                # series: [k, start]; the last finite value per row, zero when none exists.
                if start == 0:
                    return torch.zeros(sel.numel(), dtype=series.dtype, device=series.device)
                prefix = series
                finite = torch.isfinite(prefix)
                positions = torch.arange(start, device=prefix.device).expand_as(prefix)
                last = torch.where(finite, positions, torch.full_like(positions, -1)).max(dim=1).values
                gathered = prefix.gather(1, last.clamp_min(0).unsqueeze(1)).squeeze(1)
                return torch.where(last >= 0, gathered, torch.zeros_like(gathered))

            if channels in ("power", "both"):
                values = _causal_vector(x_p[index, sel, :start, 0])
                power[index, sel, start:end, 0] = values.unsqueeze(1)
                p_mask[index, sel, start:end, 0] = 1
            if channels in ("wind", "both"):
                values = _causal_vector(x_w[index, sel, :start, 0])
                wind[index, sel, start:end, 0] = values.unsqueeze(1)
                w_mask[index, sel, start:end, 0] = 1
        self.counts["corrupted_batches"] += 1
        self.counts["corrupted_samples"] += bsz
        self.last_scenarios = scenarios
        return (power, wind, x_hist_nwp, x_target_nwp, target), p_mask, w_mask
