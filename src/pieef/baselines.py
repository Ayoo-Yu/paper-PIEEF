from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from contextlib import contextmanager
import gc
import itertools
import os
import shutil
import tempfile
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

from .core import calc_metrics
from .training_timing import TrainingTimer

UNIFIED_HIST_DIM = 14  # power(1) + scada_ws(1) + hist_nwp(12)
TARGET_NWP_DIM = 12


def _weather_dim(nwp_dim: int) -> int:
    if isinstance(nwp_dim, bool) or not isinstance(nwp_dim, (int, np.integer)) or nwp_dim < 1:
        raise ValueError("nwp_dim must be a positive integer")
    return int(nwp_dim)


def _check_nwp_inputs(x_p, x_n_hist, x_nwp_target, nwp_dim):
    if x_p.ndim != 4 or x_p.shape[-1] != 1:
        raise ValueError("Historical power must have shape [B,N,T,1]")
    if x_n_hist is not None and tuple(x_n_hist.shape) != (*x_p.shape[:3], nwp_dim):
        raise ValueError(f"Historical NWP must have shape [B,N,T,{nwp_dim}]")
    if x_nwp_target is not None and tuple(x_nwp_target.shape) != (*x_p.shape[:2], nwp_dim):
        raise ValueError(f"Target NWP must have shape [B,N,{nwp_dim}]")


def _unified_hist(
    x_p: torch.Tensor,
    x_w: torch.Tensor | None = None,
    x_n_hist: torch.Tensor | None = None,
    nwp_dim: int = TARGET_NWP_DIM,
) -> torch.Tensor:
    _check_nwp_inputs(x_p, x_n_hist, None, _weather_dim(nwp_dim))
    bsz, n_turb, seq_len, _ = x_p.shape
    if x_w is None:
        x_w = torch.zeros(bsz, n_turb, seq_len, 1, device=x_p.device, dtype=x_p.dtype)
    if x_n_hist is None:
        x_n_hist = torch.zeros(bsz, n_turb, seq_len, nwp_dim, device=x_p.device, dtype=x_p.dtype)
    return torch.cat([x_p, x_w, x_n_hist], dim=-1)


def _ct_curve(ws_ms: torch.Tensor) -> torch.Tensor:
    # Smooth proxy of thrust coefficient curve in [0, 1].
    ws = ws_ms.clamp(min=0.0)
    ct = torch.zeros_like(ws)
    mid = (ws >= 3.0) & (ws < 12.0)
    high = (ws >= 12.0) & (ws < 25.0)
    ct[mid] = 0.86 - 0.02 * (ws[mid] - 8.0).abs()
    ct[high] = 0.22
    return ct.clamp(0.0, 0.9)


def _power_curve_norm(ws_ms: torch.Tensor, cut_in: float = 3.0, rated: float = 12.0, cut_out: float = 25.0) -> torch.Tensor:
    p = torch.zeros_like(ws_ms)
    ramp = (ws_ms >= cut_in) & (ws_ms < rated)
    flat = (ws_ms >= rated) & (ws_ms < cut_out)
    p[ramp] = ((ws_ms[ramp] - cut_in) / max(1e-6, rated - cut_in)) ** 3
    p[flat] = 1.0
    return p.clamp(0.0, 1.0)


def _denorm_nwp_target(
    x_nwp_target: torch.Tensor,
    nwp_mean: torch.Tensor | None = None,
    nwp_std: torch.Tensor | None = None,
    nwp_target_speed_index: int = -3,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Return (wind_speed_mps, wind_vec_xy)
    x = x_nwp_target
    if nwp_mean is not None and nwp_std is not None and nwp_mean.numel() == x.shape[-1]:
        x = x * nwp_std.to(x.device).view(1, 1, -1) + nwp_mean.to(x.device).view(1, 1, -1)
    if not -x.shape[-1] <= nwp_target_speed_index < x.shape[-1]:
        raise ValueError("NWP target speed feature index is out of range")
    ws = x[:, :, nwp_target_speed_index].clamp(min=0.0)
    wd_sin = x[:, :, -2]
    wd_cos = x[:, :, -1]
    wind_vec = torch.stack([wd_sin, wd_cos], dim=-1)
    wind_vec = wind_vec / (torch.norm(wind_vec, dim=-1, keepdim=True) + 1e-6)
    return ws, wind_vec


def _jensen_wake(
    ws_ms: torch.Tensor,
    wind_vec: torch.Tensor,
    coords_xy: torch.Tensor,
    rotor_d: torch.Tensor,
    kw: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    # ws_ms: [B, N], wind_vec: [B, N, 2], coords: [N, 2], rotor_d: [N]
    bsz, n_turb = ws_ms.shape
    device = ws_ms.device
    coords = coords_xy.to(device)
    d = rotor_d.to(device).clamp(min=1.0)

    # Use farm-level prevailing direction (mean over turbines).
    w = wind_vec.mean(dim=1)
    w = w / (torch.norm(w, dim=-1, keepdim=True) + 1e-6)
    perp = torch.stack([-w[:, 1], w[:, 0]], dim=-1)

    ci = coords.view(1, n_turb, 1, 2)
    cj = coords.view(1, 1, n_turb, 2)
    delta = cj - ci  # i -> j
    along = (delta * w.view(bsz, 1, 1, 2)).sum(dim=-1)
    cross = torch.abs((delta * perp.view(bsz, 1, 1, 2)).sum(dim=-1))

    ct_i = _ct_curve(ws_ms).unsqueeze(-1)
    d_i = d.view(1, n_turb, 1)
    denom = (1.0 + 2.0 * kw * along / (d_i + 1e-6)).clamp(min=1e-4)
    dw = d_i + 2.0 * kw * along
    in_wake = (along > 0.0) & (cross <= 0.5 * dw)

    deficit = (1.0 - torch.sqrt((1.0 - ct_i).clamp(min=1e-6))) / (denom**2)
    deficit = torch.where(in_wake, deficit, torch.zeros_like(deficit))
    eye = torch.eye(n_turb, device=device, dtype=torch.bool).unsqueeze(0)
    deficit = torch.where(eye, torch.zeros_like(deficit), deficit)

    total_deficit = torch.sqrt((deficit**2).sum(dim=1).clamp(min=0.0, max=0.9999))
    ws_eff = ws_ms * (1.0 - total_deficit)
    p_phys = _power_curve_norm(ws_eff)
    return deficit, p_phys


class PersistenceModel:
    def predict_batch(self, x_p: torch.Tensor) -> torch.Tensor:
        return x_p[:, :, -1, 0]


class DLinearModel(nn.Module):
    def __init__(self, seq_len: int, nwp_dim: int = TARGET_NWP_DIM) -> None:
        super().__init__()
        self.nwp_dim = _weather_dim(nwp_dim)
        self.hist_dim = 2 + self.nwp_dim
        self.hist_proj = nn.Linear(self.hist_dim, 1)
        self.target_proj = nn.Linear(self.nwp_dim, 1)
        self.linear = nn.Linear(seq_len, 1)

    def forward(
        self,
        x_p: torch.Tensor,
        x_w: torch.Tensor | None = None,
        x_n_hist: torch.Tensor | None = None,
        x_nwp_target: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _check_nwp_inputs(x_p, x_n_hist, x_nwp_target, self.nwp_dim)
        bsz, n_turb, seq_len, _ = x_p.shape
        x_hist = _unified_hist(x_p, x_w=x_w, x_n_hist=x_n_hist, nwp_dim=self.nwp_dim)
        x = self.hist_proj(x_hist).squeeze(-1).reshape(bsz * n_turb, seq_len)
        y = self.linear(x).reshape(bsz, n_turb)
        if x_nwp_target is not None:
            y = y + self.target_proj(x_nwp_target).squeeze(-1)
        return y


class GRUModel(nn.Module):
    def __init__(self, hidden_dim: int = 64, nwp_dim: int = TARGET_NWP_DIM) -> None:
        super().__init__()
        self.nwp_dim = _weather_dim(nwp_dim)
        self.hist_dim = 2 + self.nwp_dim
        self.gru = nn.GRU(input_size=self.hist_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.target_proj = nn.Linear(self.nwp_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        x_p: torch.Tensor,
        x_w: torch.Tensor | None = None,
        x_n_hist: torch.Tensor | None = None,
        x_nwp_target: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _check_nwp_inputs(x_p, x_n_hist, x_nwp_target, self.nwp_dim)
        bsz, n_turb, seq_len, _ = x_p.shape
        x_hist = _unified_hist(x_p, x_w=x_w, x_n_hist=x_n_hist, nwp_dim=self.nwp_dim)
        x = x_hist.reshape(bsz * n_turb, seq_len, self.hist_dim)
        out, _ = self.gru(x)
        h = out[:, -1, :]
        if x_nwp_target is not None:
            h = h + self.target_proj(x_nwp_target.reshape(bsz * n_turb, self.nwp_dim))
        y = self.fc(h).reshape(bsz, n_turb)
        return y


class TCNModel(nn.Module):
    def __init__(self, channels: int = 32, nwp_dim: int = TARGET_NWP_DIM) -> None:
        super().__init__()
        self.nwp_dim = _weather_dim(nwp_dim)
        self.hist_dim = 2 + self.nwp_dim
        self.net = nn.Sequential(
            nn.Conv1d(self.hist_dim, channels, kernel_size=3, padding=2, dilation=1),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=4, dilation=2),
            nn.ReLU(),
        )
        self.target_proj = nn.Linear(self.nwp_dim, channels)
        self.head = nn.Linear(channels, 1)

    def forward(
        self,
        x_p: torch.Tensor,
        x_w: torch.Tensor | None = None,
        x_n_hist: torch.Tensor | None = None,
        x_nwp_target: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _check_nwp_inputs(x_p, x_n_hist, x_nwp_target, self.nwp_dim)
        bsz, n_turb, seq_len, _ = x_p.shape
        x_hist = _unified_hist(x_p, x_w=x_w, x_n_hist=x_n_hist, nwp_dim=self.nwp_dim)
        x = x_hist.reshape(bsz * n_turb, seq_len, self.hist_dim).transpose(1, 2)
        feat = self.net(x)[:, :, -1]
        if x_nwp_target is not None:
            feat = feat + self.target_proj(x_nwp_target.reshape(bsz * n_turb, self.nwp_dim))
        y = self.head(feat).reshape(bsz, n_turb)
        return y


class ITransformerModel(nn.Module):
    def __init__(self, seq_len: int, d_model: int = 128, n_heads: int = 4, nwp_dim: int = TARGET_NWP_DIM) -> None:
        super().__init__()
        self.nwp_dim = _weather_dim(nwp_dim)
        self.hist_dim = 2 + self.nwp_dim
        self.embed = nn.Linear(self.hist_dim, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.target_proj = nn.Linear(self.nwp_dim, d_model)
        self.head = nn.Linear(d_model, 1)
        self.seq_len = seq_len

    def forward(
        self,
        x_p: torch.Tensor,
        x_w: torch.Tensor | None = None,
        x_n_hist: torch.Tensor | None = None,
        x_nwp_target: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _check_nwp_inputs(x_p, x_n_hist, x_nwp_target, self.nwp_dim)
        bsz, n_turb, _, _ = x_p.shape
        x_hist = _unified_hist(x_p, x_w=x_w, x_n_hist=x_n_hist, nwp_dim=self.nwp_dim)
        x = x_hist.reshape(bsz * n_turb, self.seq_len, self.hist_dim)
        z = self.embed(x)
        z = self.encoder(z)
        h = z[:, -1, :]
        if x_nwp_target is not None:
            h = h + self.target_proj(x_nwp_target.reshape(bsz * n_turb, self.nwp_dim))
        y = self.head(h).reshape(bsz, n_turb)
        return y


class PatchTSTModel(nn.Module):
    def __init__(self, seq_len: int, patch_len: int = 16, stride: int = 8, d_model: int = 64, nwp_dim: int = TARGET_NWP_DIM) -> None:
        super().__init__()
        self.nwp_dim = _weather_dim(nwp_dim)
        self.hist_dim = 2 + self.nwp_dim
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.hist_proj = nn.Linear(self.hist_dim, 1)
        self.proj = nn.Linear(patch_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=d_model * 2, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.target_proj = nn.Linear(self.nwp_dim, d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(
        self,
        x_p: torch.Tensor,
        x_w: torch.Tensor | None = None,
        x_n_hist: torch.Tensor | None = None,
        x_nwp_target: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _check_nwp_inputs(x_p, x_n_hist, x_nwp_target, self.nwp_dim)
        bsz, n_turb, _, _ = x_p.shape
        x_hist = _unified_hist(x_p, x_w=x_w, x_n_hist=x_n_hist, nwp_dim=self.nwp_dim)
        x = self.hist_proj(x_hist).squeeze(-1).reshape(bsz * n_turb, self.seq_len)
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        z = self.proj(patches)
        z = self.encoder(z)
        h = z[:, -1, :]
        if x_nwp_target is not None:
            h = h + self.target_proj(x_nwp_target.reshape(bsz * n_turb, self.nwp_dim))
        y = self.head(h).reshape(bsz, n_turb)
        return y


class ACFormerModel(nn.Module):
    def __init__(
        self,
        seq_len: int,
        d_model: int = 96,
        patch_kernel: int = 10,
        stride: int = 2,
        n_heads: int = 8,
        n_layers: int = 1,
        nwp_dim: int = TARGET_NWP_DIM,
    ) -> None:
        super().__init__()
        self.nwp_dim = _weather_dim(nwp_dim)
        self.hist_dim = 2 + self.nwp_dim
        self.seq_len = seq_len
        self.patch_kernel = patch_kernel
        self.stride = stride
        self.n_heads = n_heads
        self.n_layers = max(1, n_layers)
        self.d_model = d_model

        self.hist_proj = nn.Linear(self.hist_dim, 1)
        self.target_proj = nn.Linear(self.nwp_dim, 1)
        # Shared patch compression: shared kernels across channels.
        self.shared_patch = nn.Conv1d(1, n_heads, kernel_size=patch_kernel, stride=stride)

        l_patch = (seq_len - patch_kernel) // stride + 1
        self.l_patch = l_patch
        attn_dim = max(8, min(64, l_patch))
        self.q_proj = nn.ModuleList([nn.Linear(l_patch, attn_dim) for _ in range(n_heads)])
        self.k_proj = nn.ModuleList([nn.Linear(l_patch, attn_dim) for _ in range(n_heads)])
        self.v_proj = nn.ModuleList([nn.Linear(l_patch, l_patch) for _ in range(n_heads)])

        # Isomorphic temporal gate + GLU.
        self.gate_conv = nn.ModuleList([nn.Conv1d(1, 2, kernel_size=3, padding=1) for _ in range(n_heads)])

        # Channel-independent transposed conv (decoder), created lazily by channel count.
        self.decoder_by_channels: dict[int, nn.ConvTranspose1d] = {}

        self.residual_linear = nn.Linear(seq_len, 1)
        self.out_linear = nn.Linear(seq_len, 1)
        self.loss_name = "l1"

        self._cached_mean: torch.Tensor | None = None
        self._cached_std: torch.Tensor | None = None

    def _revin_norm(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, S]
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False) + 1e-5
        self._cached_mean = mean
        self._cached_std = std
        return (x - mean) / std

    def _revin_denorm(self, x: torch.Tensor) -> torch.Tensor:
        if self._cached_mean is None or self._cached_std is None:
            return x
        return x * self._cached_std + self._cached_mean

    def _decoder(self, n_channels: int, device: torch.device) -> nn.ConvTranspose1d:
        dec = self.decoder_by_channels.get(n_channels)
        if dec is not None:
            return dec
        dec = nn.ConvTranspose1d(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=self.patch_kernel,
            stride=self.stride,
            groups=n_channels,
        ).to(device)
        self.decoder_by_channels[n_channels] = dec
        self.add_module(f"acformer_decoder_c{n_channels}", dec)
        return dec

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Existing checkpoints include the decoder created at the first forward.
        # Register that same module before strict loading into a fresh model.
        decoder_prefix = prefix + "acformer_decoder_c"
        for key in state_dict:
            if key.startswith(decoder_prefix) and key.endswith(".weight"):
                channel_text = key[len(decoder_prefix):-len(".weight")]
                if channel_text.isdigit():
                    self._decoder(int(channel_text), self.hist_proj.weight.device)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    def _temporal_gated_attention(self, z_head: torch.Tensor, head_id: int) -> torch.Tensor:
        # z_head: [B, C, L]
        q = self.q_proj[head_id](z_head)
        k = self.k_proj[head_id](z_head)
        v = self.v_proj[head_id](z_head)
        scale = float(q.shape[-1]) ** 0.5
        a = torch.softmax(torch.einsum("bcd,bnd->bcn", q, k) / (scale + 1e-6), dim=-1)
        h_a = torch.einsum("bcn,bnl->bcl", a, v)

        g = self.gate_conv[head_id](z_head.reshape(-1, 1, z_head.shape[-1]))
        g = F.glu(g, dim=1).reshape_as(h_a)
        return h_a * g

    def forward(
        self,
        x_p: torch.Tensor,
        x_w: torch.Tensor | None = None,
        x_n_hist: torch.Tensor | None = None,
        x_nwp_target: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _check_nwp_inputs(x_p, x_n_hist, x_nwp_target, self.nwp_dim)
        # x_p: [B, C, S, 1]
        bsz, n_turb, _, _ = x_p.shape
        x_hist = _unified_hist(x_p, x_w=x_w, x_n_hist=x_n_hist, nwp_dim=self.nwp_dim)
        x = self.hist_proj(x_hist).squeeze(-1)
        x_norm = self._revin_norm(x)

        # Shared patch compression.
        z = self.shared_patch(x_norm.reshape(bsz * n_turb, 1, self.seq_len))  # [B*C, M, L]
        z = z.reshape(bsz, n_turb, self.n_heads, self.l_patch).permute(0, 2, 1, 3)  # [B, M, C, L]

        h = z
        for _ in range(self.n_layers):
            hs = []
            for m in range(self.n_heads):
                hs.append(self._temporal_gated_attention(h[:, m], m))
            h = torch.stack(hs, dim=1)

        h_merge = h.mean(dim=1)  # [B, C, L]

        # Independent patch expansion (channel-wise deconvolution).
        dec = self._decoder(n_turb, x_p.device)
        x_amp = dec(h_merge)
        x_amp = x_amp[:, :, : self.seq_len]

        # Residual + output projection.
        x_res = x_norm + x_amp
        y_norm = self.out_linear(x_res).squeeze(-1) + self.residual_linear(x_norm).squeeze(-1)
        if x_nwp_target is not None:
            y_norm = y_norm + self.target_proj(x_nwp_target).squeeze(-1)
        y = self._revin_denorm(y_norm.unsqueeze(-1)).squeeze(-1)
        return y


class TimesNetModel(nn.Module):
    def __init__(self, seq_len: int, d_model: int = 64, nwp_dim: int = TARGET_NWP_DIM) -> None:
        super().__init__()
        self.nwp_dim = _weather_dim(nwp_dim)
        self.hist_dim = 2 + self.nwp_dim
        self.seq_len = seq_len
        self.freq_conv = nn.Conv1d(self.hist_dim, d_model, kernel_size=3, padding=1)
        self.season_conv = nn.Conv1d(d_model, d_model, kernel_size=7, padding=3)
        self.trend_conv = nn.Conv1d(d_model, d_model, kernel_size=15, padding=7)
        self.target_proj = nn.Linear(self.nwp_dim, d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(
        self,
        x_p: torch.Tensor,
        x_w: torch.Tensor | None = None,
        x_n_hist: torch.Tensor | None = None,
        x_nwp_target: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _check_nwp_inputs(x_p, x_n_hist, x_nwp_target, self.nwp_dim)
        bsz, n_turb, _, _ = x_p.shape
        x_hist = _unified_hist(x_p, x_w=x_w, x_n_hist=x_n_hist, nwp_dim=self.nwp_dim)
        x = x_hist.reshape(bsz * n_turb, self.seq_len, self.hist_dim).transpose(1, 2)
        z = F.gelu(self.freq_conv(x))
        z = F.gelu(self.season_conv(z) + self.trend_conv(z))
        h = z[:, :, -1]
        if x_nwp_target is not None:
            h = h + self.target_proj(x_nwp_target.reshape(bsz * n_turb, self.nwp_dim))
        y = self.head(h).reshape(bsz, n_turb)
        return y


class TimeMixerModel(nn.Module):
    def __init__(self, seq_len: int, hidden: int = 64, nwp_dim: int = TARGET_NWP_DIM) -> None:
        super().__init__()
        self.nwp_dim = _weather_dim(nwp_dim)
        self.hist_dim = 2 + self.nwp_dim
        self.seq_len = seq_len
        self.hist_proj = nn.Linear(self.hist_dim, 1)
        self.short = nn.Linear(seq_len, hidden)
        self.mid = nn.Linear(seq_len // 2, hidden)
        self.long = nn.Linear(seq_len // 4, hidden)
        self.target_proj = nn.Linear(self.nwp_dim, hidden)
        self.head = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(
        self,
        x_p: torch.Tensor,
        x_w: torch.Tensor | None = None,
        x_n_hist: torch.Tensor | None = None,
        x_nwp_target: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _check_nwp_inputs(x_p, x_n_hist, x_nwp_target, self.nwp_dim)
        bsz, n_turb, _, _ = x_p.shape
        x_hist = _unified_hist(x_p, x_w=x_w, x_n_hist=x_n_hist, nwp_dim=self.nwp_dim)
        x = self.hist_proj(x_hist).squeeze(-1).reshape(bsz * n_turb, self.seq_len)
        short = self.short(x)
        mid = self.mid(F.avg_pool1d(x.unsqueeze(1), kernel_size=2, stride=2).squeeze(1))
        long = self.long(F.avg_pool1d(x.unsqueeze(1), kernel_size=4, stride=4).squeeze(1))
        if x_nwp_target is not None:
            short = short + self.target_proj(x_nwp_target.reshape(bsz * n_turb, self.nwp_dim))
        z = torch.cat([short, mid, long], dim=-1)
        y = self.head(z).reshape(bsz, n_turb)
        return y


class STANModel(nn.Module):
    def __init__(
        self,
        seq_len: int,
        hidden: int = 64,
        gru_hidden: int = 200,
        gat_hidden: int = 64,
        static_feat_dim: int = 4,
        dist_matrix: torch.Tensor | None = None,
        node_static: torch.Tensor | None = None,
        gravity_alpha: float = 0.35,
        gravity_beta: float = 0.37,
        gravity_r: float = 30.0,
        nwp_dim: int = TARGET_NWP_DIM,
    ) -> None:
        super().__init__()
        self.nwp_dim = _weather_dim(nwp_dim)
        self.hist_dim = 2 + self.nwp_dim
        self.seq_len = seq_len
        self.static_feat_dim = static_feat_dim
        self.gat_hidden = gat_hidden
        self.gravity_alpha = gravity_alpha
        self.gravity_beta = gravity_beta
        self.gravity_r = gravity_r

        self.in_proj = nn.Linear(1 + self.nwp_dim + static_feat_dim, gat_hidden)
        self.gat1 = nn.Linear(gat_hidden, gat_hidden)
        self.gat2 = nn.Linear(gat_hidden, gat_hidden)
        self.attn1 = nn.Linear(gat_hidden * 2, 1)
        self.attn2 = nn.Linear(gat_hidden * 2, 1)
        self.gru = nn.GRU(input_size=gat_hidden, hidden_size=gru_hidden, batch_first=True)
        self.ctx_proj = nn.Linear(gru_hidden, gat_hidden)
        self.head = nn.Sequential(nn.Linear(gat_hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

        self.register_buffer("dist_matrix", dist_matrix if dist_matrix is not None else torch.empty(0), persistent=False)
        self.register_buffer("node_static", node_static if node_static is not None else torch.empty(0), persistent=False)

    def _gravity_adj(self, n_turb: int, device: torch.device) -> torch.Tensor:
        if self.dist_matrix.numel() > 0 and self.dist_matrix.shape[0] == n_turb:
            dist = self.dist_matrix.to(device)
        else:
            idx = torch.arange(n_turb, device=device, dtype=torch.float32)
            dist = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
            dist = dist / (dist.max() + 1e-6)

        if self.node_static.numel() > 0 and self.node_static.shape[0] == n_turb:
            pop = self.node_static[:, 2].to(device).clamp_min(1e-6)
        else:
            pop = torch.ones(n_turb, device=device)

        pop_i = pop.unsqueeze(1).pow(self.gravity_alpha)
        pop_j = pop.unsqueeze(0).pow(self.gravity_beta)
        gravity = pop_i * pop_j * torch.exp(-dist / self.gravity_r)
        gravity = gravity / (gravity.sum(dim=-1, keepdim=True) + 1e-6)
        return gravity

    @staticmethod
    def _gat_layer(h: torch.Tensor, gravity: torch.Tensor, lin: nn.Linear, attn: nn.Linear) -> torch.Tensor:
        # h: [B, N, D]
        z = lin(h)
        bsz, n_turb, d = z.shape
        zi = z.unsqueeze(2).expand(bsz, n_turb, n_turb, d)
        zj = z.unsqueeze(1).expand(bsz, n_turb, n_turb, d)
        e = F.leaky_relu(attn(torch.cat([zi, zj], dim=-1)).squeeze(-1), negative_slope=0.2)
        e = e + torch.log(gravity.unsqueeze(0) + 1e-8)
        a = torch.softmax(e, dim=-1)
        out = torch.einsum("bij,bjd->bid", a, z)
        return F.elu(out)

    def forward(
        self,
        x_p: torch.Tensor,
        x_w: torch.Tensor | None = None,
        x_n_hist: torch.Tensor | None = None,
        x_nwp_target: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _check_nwp_inputs(x_p, x_n_hist, x_nwp_target, self.nwp_dim)
        _ = x_w
        _ = x_nwp_target
        bsz, n_turb, seq_len, _ = x_p.shape
        if x_n_hist is None:
            x_n_hist = torch.zeros(bsz, n_turb, seq_len, self.nwp_dim, device=x_p.device, dtype=x_p.dtype)

        if self.node_static.numel() > 0 and self.node_static.shape[0] == n_turb:
            s = self.node_static.to(x_p.device).unsqueeze(0).unsqueeze(2).expand(bsz, n_turb, seq_len, -1)
        else:
            s = torch.zeros(bsz, n_turb, seq_len, self.static_feat_dim, device=x_p.device, dtype=x_p.dtype)

        dyn = torch.cat([x_p.squeeze(-1).unsqueeze(-1), x_n_hist], dim=-1)
        x = torch.cat([dyn, s], dim=-1)
        gravity = self._gravity_adj(n_turb, x_p.device)

        seq_global = []
        node_last = None
        for t in range(seq_len):
            h = self.in_proj(x[:, :, t, :])
            h = self._gat_layer(h, gravity, self.gat1, self.attn1)
            h = self._gat_layer(h, gravity, self.gat2, self.attn2)
            node_last = h
            seq_global.append(torch.max(h, dim=1).values)

        g = torch.stack(seq_global, dim=1)
        _, h_last = self.gru(g)
        ctx = self.ctx_proj(h_last[-1]).unsqueeze(1)
        out = self.head(node_last + ctx).squeeze(-1)
        return out


class WakeGNNModel(nn.Module):
    def __init__(
        self,
        seq_len: int,
        hidden: int = 64,
        coords_xy: torch.Tensor | None = None,
        rotor_d: torch.Tensor | None = None,
        nwp_mean: torch.Tensor | None = None,
        nwp_std: torch.Tensor | None = None,
        kw: float = 0.05,
        nwp_target_speed_index: int = -3,
        nwp_dim: int = TARGET_NWP_DIM,
    ) -> None:
        super().__init__()
        self.nwp_dim = _weather_dim(nwp_dim)
        self.hist_dim = 2 + self.nwp_dim
        self.seq_len = seq_len
        self.encoder = nn.GRU(input_size=1, hidden_size=hidden, num_layers=1, batch_first=True)
        self.msg = nn.Linear(hidden, hidden)
        self.upd = nn.GRUCell(hidden, hidden)
        self.head = nn.Linear(hidden, 1)
        self.kw = kw
        self.nwp_target_speed_index = int(nwp_target_speed_index)

        self.register_buffer("coords_xy", coords_xy if coords_xy is not None else torch.empty(0), persistent=False)
        self.register_buffer("rotor_d", rotor_d if rotor_d is not None else torch.empty(0), persistent=False)
        self.register_buffer("nwp_mean", nwp_mean if nwp_mean is not None else torch.empty(0), persistent=False)
        self.register_buffer("nwp_std", nwp_std if nwp_std is not None else torch.empty(0), persistent=False)

    def forward(
        self,
        x_p: torch.Tensor,
        x_w: torch.Tensor | None = None,
        x_n_hist: torch.Tensor | None = None,
        x_nwp_target: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _check_nwp_inputs(x_p, x_n_hist, x_nwp_target, self.nwp_dim)
        _ = x_w
        _ = x_n_hist
        bsz, n_turb, _, _ = x_p.shape
        x = x_p.reshape(bsz * n_turb, self.seq_len, 1)
        z, _ = self.encoder(x)
        h = z[:, -1, :].reshape(bsz, n_turb, -1)

        if (
            x_nwp_target is not None
            and self.coords_xy.numel() > 0
            and self.rotor_d.numel() > 0
            and x_nwp_target.shape[1] == n_turb
            and x_nwp_target.shape[-1] >= 3
        ):
            ws, wind_vec = _denorm_nwp_target(
                x_nwp_target,
                self.nwp_mean if self.nwp_mean.numel() > 0 else None,
                self.nwp_std if self.nwp_std.numel() > 0 else None,
                self.nwp_target_speed_index,
            )
            deficit, _ = _jensen_wake(ws, wind_vec, self.coords_xy, self.rotor_d, kw=self.kw)
            adj = deficit / (deficit.sum(dim=-1, keepdim=True) + 1e-6)
        else:
            idx = torch.arange(n_turb, device=h.device, dtype=torch.float32)
            dist = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs() + 1.0
            adj0 = torch.exp(-dist / max(1.0, n_turb / 6.0))
            adj = (adj0 / (adj0.sum(dim=-1, keepdim=True) + 1e-6)).unsqueeze(0).expand(bsz, -1, -1)

        m = torch.einsum("bij,bjd->bid", adj, self.msg(h))
        h2 = self.upd(m.reshape(-1, m.shape[-1]), h.reshape(-1, h.shape[-1])).reshape(bsz, n_turb, -1)
        y = self.head(h2).squeeze(-1)
        return y


class JensenNetModel(nn.Module):
    def __init__(
        self,
        seq_len: int,
        hidden: int = 64,
        coords_xy: torch.Tensor | None = None,
        rotor_d: torch.Tensor | None = None,
        nwp_mean: torch.Tensor | None = None,
        nwp_std: torch.Tensor | None = None,
        kw: float = 0.05,
        nwp_target_speed_index: int = -3,
        nwp_dim: int = TARGET_NWP_DIM,
    ) -> None:
        super().__init__()
        self.nwp_dim = _weather_dim(nwp_dim)
        self.hist_dim = 2 + self.nwp_dim
        self.seq_len = seq_len
        self.feature = nn.Sequential(
            nn.Linear(seq_len, hidden),
            nn.Sigmoid(),
            nn.Linear(hidden, hidden),
            nn.Sigmoid(),
        )
        self.phys_proj = nn.Sequential(nn.Linear(1, hidden), nn.Sigmoid())
        self.wake_gate = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Sigmoid())
        self.head = nn.Linear(hidden, 1)
        self.kw = kw
        self.nwp_target_speed_index = int(nwp_target_speed_index)
        self.register_buffer("coords_xy", coords_xy if coords_xy is not None else torch.empty(0), persistent=False)
        self.register_buffer("rotor_d", rotor_d if rotor_d is not None else torch.empty(0), persistent=False)
        self.register_buffer("nwp_mean", nwp_mean if nwp_mean is not None else torch.empty(0), persistent=False)
        self.register_buffer("nwp_std", nwp_std if nwp_std is not None else torch.empty(0), persistent=False)

    def forward(
        self,
        x_p: torch.Tensor,
        x_w: torch.Tensor | None = None,
        x_n_hist: torch.Tensor | None = None,
        x_nwp_target: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _check_nwp_inputs(x_p, x_n_hist, x_nwp_target, self.nwp_dim)
        _ = x_w
        _ = x_n_hist
        bsz, n_turb, _, _ = x_p.shape
        x = x_p.squeeze(-1).reshape(bsz * n_turb, self.seq_len)
        feat = self.feature(x).reshape(bsz, n_turb, -1)

        if (
            x_nwp_target is not None
            and self.coords_xy.numel() > 0
            and self.rotor_d.numel() > 0
            and x_nwp_target.shape[1] == n_turb
            and x_nwp_target.shape[-1] >= 3
        ):
            ws, wind_vec = _denorm_nwp_target(
                x_nwp_target,
                self.nwp_mean if self.nwp_mean.numel() > 0 else None,
                self.nwp_std if self.nwp_std.numel() > 0 else None,
                self.nwp_target_speed_index,
            )
            _, p_phys = _jensen_wake(ws, wind_vec, self.coords_xy, self.rotor_d, kw=self.kw)
        else:
            p_phys = _power_curve_norm(x_p[:, :, -1, 0] * 25.0)

        p_phys = (p_phys - p_phys.amin(dim=1, keepdim=True)) / (p_phys.amax(dim=1, keepdim=True) - p_phys.amin(dim=1, keepdim=True) + 1e-6)
        phys = self.phys_proj(p_phys.unsqueeze(-1))
        gate = self.wake_gate(torch.cat([feat, phys], dim=-1))
        y = self.head(feat * gate).squeeze(-1)
        return y


@dataclass
class DLTrainConfig:
    epochs: int = 5
    lr: float = 1e-3
    max_batches_per_epoch: int | None = None
    max_test_batches: int | None = None
    patience: int | None = None
    min_delta: float = 0.0
    loss_name: str | None = None
    huber_delta: float = 0.1


def _predict_dl(model: nn.Module, x_p, x_w, x_n_hist, x_nwp_t) -> torch.Tensor:
    try:
        return model(x_p, x_w=x_w, x_n_hist=x_n_hist, x_nwp_target=x_nwp_t)
    except TypeError:
        try:
            return model(x_p, x_nwp_t)
        except TypeError:
            return model(x_p)


def baseline_loss_spec(model: nn.Module | None, cfg: DLTrainConfig | None = None):
    """Explicit loss when configured; otherwise preserve each legacy model's loss."""
    name = cfg.loss_name if cfg is not None and cfg.loss_name is not None else getattr(model, "loss_name", "mse")
    delta = cfg.huber_delta if cfg is not None else 0.1
    if name not in {"mse", "l1", "huber"}:
        raise ValueError("Baseline loss_name must be mse, l1, huber or None")
    if isinstance(delta, bool) or not np.isfinite(delta) or delta <= 0:
        raise ValueError("Baseline huber_delta must be positive and finite")
    return name, float(delta)


def _observed_loss(model: nn.Module, pred: torch.Tensor, target: torch.Tensor, cfg: DLTrainConfig | None = None):
    if pred.shape != target.shape:
        raise ValueError(f"Prediction shape {tuple(pred.shape)} differs from target {tuple(target.shape)}")
    valid = torch.isfinite(target)
    count = int(valid.sum().item())
    if count == 0:
        return None, 0
    if not torch.isfinite(pred[valid]).all():
        raise ValueError("Baseline produced a non-finite prediction for an observed target")
    name, delta = baseline_loss_spec(model, cfg)
    if name == "huber":
        return F.huber_loss(pred[valid], target[valid], delta=delta), count
    if name == "l1":
        return F.l1_loss(pred[valid], target[valid]), count
    return F.mse_loss(pred[valid], target[valid]), count


def _validation_loss(model: nn.Module, loader, device: torch.device, cfg: DLTrainConfig | None = None) -> float:
    model.eval()
    total_loss, total_count = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            x_p, x_w, x_n_hist, x_nwp_t, y = (value.to(device, non_blocking=True) for value in batch)
            pred = _predict_dl(model, x_p, x_w, x_n_hist, x_nwp_t)
            loss, count = _observed_loss(model, pred, y, cfg)
            if count:
                total_loss += float(loss.item()) * count
                total_count += count
    if total_count == 0:
        raise ValueError("Validation split has no finite observed targets")
    return total_loss / total_count


def _baseline_metrics(y_true: np.ndarray, y_pred: np.ndarray, capacity_mw: float | np.ndarray) -> dict[str, float]:
    if y_true.shape != y_pred.shape:
        raise ValueError("Baseline prediction and target arrays must have the same shape")
    valid = np.isfinite(y_true)
    if not valid.any():
        raise ValueError("Evaluation split has no finite observed targets")
    if not np.isfinite(y_pred[valid]).all():
        raise ValueError("Baseline produced non-finite predictions for observed targets")
    nmae, nrmse, r2 = calc_metrics(y_true, y_pred, capacity_mw)
    return {"nmae": nmae, "nrmse": nrmse, "r2": r2}


def train_dl_model(
    model: nn.Module,
    train_loader,
    device: torch.device,
    cfg: DLTrainConfig,
    progress_label: str | None = None,
    val_loader=None,
    checkpoint_dir: Path | str | None = None,
) -> nn.Module:
    if cfg.epochs < 1 or cfg.lr <= 0:
        raise ValueError("epochs and learning rate must be positive")
    if cfg.patience is not None and cfg.patience < 1:
        raise ValueError("patience must be positive or None")
    if cfg.min_delta < 0:
        raise ValueError("min_delta cannot be negative")
    effective_loss, effective_delta = baseline_loss_spec(model, cfg)
    model = model.to(device)
    timing = TrainingTimer(device)
    # ACFormer creates a turbine-count-specific decoder. Register it before Adam.
    if isinstance(model, ACFormerModel):
        sample = next(iter(train_loader))
        model._decoder(sample[0].shape[1], device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    output_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    best_state = None
    best_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0

    for epoch_idx in range(cfg.epochs):
        model.train()
        batch_count = 0
        loss_sum = 0.0
        target_count = 0
        t0 = time.perf_counter()
        timing.start("train", "baseline", epoch_idx + 1)
        for batch_idx, (x_p, x_w, x_n_hist, x_nwp_t, y) in enumerate(train_loader):
            if cfg.max_batches_per_epoch and batch_idx >= cfg.max_batches_per_epoch:
                break
            x_p, x_w, x_n_hist, x_nwp_t, y = (
                x_p.to(device, non_blocking=True),
                x_w.to(device, non_blocking=True),
                x_n_hist.to(device, non_blocking=True),
                x_nwp_t.to(device, non_blocking=True),
                y.to(device, non_blocking=True),
            )
            opt.zero_grad()
            if not torch.isfinite(y).any():
                continue
            pred = _predict_dl(model, x_p, x_w, x_n_hist, x_nwp_t)
            loss, count = _observed_loss(model, pred, y, cfg)
            loss.backward()
            opt.step()

            batch_count += 1
            loss_sum += float(loss.detach().item()) * count
            target_count += count
        timing.stop(batch_count)
        if target_count == 0:
            raise ValueError("Training epoch has no finite observed targets")
        if val_loader is not None:
            timing.start("validation", "baseline", epoch_idx + 1)
        val_loss = _validation_loss(model, val_loader, device, cfg) if val_loader is not None else None
        if val_loader is not None:
            timing.stop(len(val_loader))
        state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        if output_dir is not None:
            torch.save(state, output_dir / "weights_last.pt")
        if val_loss is None or val_loss < best_loss - cfg.min_delta:
            best_state = state
            best_loss = val_loss if val_loss is not None else loss_sum / target_count
            best_epoch = epoch_idx + 1
            stale_epochs = 0
            if output_dir is not None:
                torch.save(best_state, output_dir / "weights_best.pt")
        else:
            stale_epochs += 1
        if progress_label is not None:
            avg_loss = loss_sum / target_count
            dt = time.perf_counter() - t0
            print(
                f"[train] {progress_label} epoch={epoch_idx + 1}/{cfg.epochs} "
                f"batches={batch_count} avg_loss={avg_loss:.6f} "
                f"val_loss={val_loss} best_epoch={best_epoch} elapsed={dt:.1f}s",
                flush=True,
            )
        if val_loader is not None and cfg.patience is not None and stale_epochs >= cfg.patience:
            break

    model.load_state_dict(best_state)
    model.training_timing = timing.result()
    model.training_summary = {
        "loss_name": effective_loss,
        "huber_delta": effective_delta if effective_loss == "huber" else None,
        "best_epoch": best_epoch,
        "epochs_run": epoch_idx + 1,
        "best_validation_loss": best_loss if val_loader is not None else None,
    }
    return model


def eval_dl_model(model: nn.Module, test_loader, device: torch.device, capacity_mw: float | np.ndarray, max_test_batches: int | None = None):
    model.eval()
    ys = []
    preds = []
    with torch.no_grad():
        for i, (x_p, x_w, x_n_hist, x_nwp_t, y) in enumerate(test_loader):
            if max_test_batches and i >= max_test_batches:
                break
            x_p = x_p.to(device, non_blocking=True)
            x_w = x_w.to(device, non_blocking=True)
            x_n_hist = x_n_hist.to(device, non_blocking=True)
            x_nwp_t = x_nwp_t.to(device, non_blocking=True)
            try:
                pred = model(x_p, x_w=x_w, x_n_hist=x_n_hist, x_nwp_target=x_nwp_t).cpu().numpy()
            except TypeError:
                try:
                    pred = model(x_p, x_nwp_t).cpu().numpy()
                except TypeError:
                    pred = model(x_p).cpu().numpy()
            ys.append(y.numpy())
            preds.append(pred)

    y_true = np.concatenate(ys, axis=0) * capacity_mw
    y_pred = np.maximum(np.concatenate(preds, axis=0) * capacity_mw, 0)
    return _baseline_metrics(y_true, y_pred, capacity_mw), y_true, y_pred


def eval_persistence(test_loader, capacity_mw: float | np.ndarray, max_test_batches: int | None = None):
    model = PersistenceModel()
    ys = []
    preds = []
    for i, (x_p, _, _, _, y) in enumerate(test_loader):
        if max_test_batches and i >= max_test_batches:
            break
        pred = model.predict_batch(x_p).numpy()
        ys.append(y.numpy())
        preds.append(pred)

    y_true = np.concatenate(ys, axis=0) * capacity_mw
    y_pred = np.maximum(np.concatenate(preds, axis=0) * capacity_mw, 0)
    return _baseline_metrics(y_true, y_pred, capacity_mw), y_true, y_pred


def _iter_flat_xy(loader, max_batches: int | None = None, drop_invalid_targets: bool = False):
    """Pack exactly the original features, retaining only one loader batch."""
    for i, (x_p, x_w, x_n_hist, x_nwp_t, y) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        bsz, n, seq_len, _ = x_p.shape
        x_hist = torch.cat([x_p, x_w, x_n_hist], dim=-1).reshape(bsz * n, seq_len, -1)
        x_hist_flat = x_hist.reshape(bsz * n, -1)
        x_target = x_nwp_t.reshape(bsz * n, -1)
        x = torch.cat([x_hist_flat, x_target], dim=-1).numpy()
        yy = y.reshape(bsz * n).numpy()
        if drop_invalid_targets:
            valid = np.isfinite(yy)
            x, yy = x[valid], yy[valid]
        yield x, yy, n


def _loader_to_flat_xy(loader, max_batches: int | None = None, drop_invalid_targets: bool = False,
                       cache_dir: Path | str | None = None):
    """Single allocation; formal training supplies an owned disk-cache directory."""
    chunks = iter(_iter_flat_xy(loader, max_batches, drop_invalid_targets))
    try:
        first = next(chunks)
    except StopIteration as exc:
        raise ValueError("Empty tabular data loader") from exc
    n_turb, features = first[2], first[0].shape[1]
    sample_count = len(loader.sampler) if hasattr(loader, "sampler") else len(loader.dataset)
    if max_batches:
        sample_count = min(sample_count, max_batches * int(loader.batch_size))
    rows = sample_count * n_turb
    if cache_dir is None:
        x_all = np.empty((rows, features), dtype=np.float32)
        y_all = np.empty(rows, dtype=np.float32)
    else:
        directory = Path(cache_dir)
        directory.mkdir(parents=True, exist_ok=True)
        required = rows * (features + 1) * np.dtype(np.float32).itemsize
        if shutil.disk_usage(directory).free < required:
            raise RuntimeError(f"Tabular cache needs at least {required / 2**30:.2f} GiB free disk space")
        x_all = np.memmap(directory / "features.float32", dtype=np.float32, mode="w+", shape=(rows, features))
        y_all = np.memmap(directory / "targets.float32", dtype=np.float32, mode="w+", shape=(rows,))
    written = 0
    try:
        for x, y, n in itertools.chain((first,), chunks):
            if n != n_turb or x.shape[1] != features or written + len(y) > rows:
                raise ValueError("Tabular loader shape or sample count changed during packing")
            end = written + len(y)
            x_all[written:end], y_all[written:end] = x, y
            written = end
        if written == 0:
            raise ValueError("Training split has no finite observed targets")
        if cache_dir is not None:
            x_all.flush()
            y_all.flush()
            # Close dirty write mappings before native training scans the fixed rows.
            x_all._mmap.close()
            y_all._mmap.close()
            x_all = np.memmap(directory / "features.float32", dtype=np.float32, mode="r", shape=(rows, features))
            y_all = np.memmap(directory / "targets.float32", dtype=np.float32, mode="r", shape=(rows,))
        return x_all[:written], y_all[:written], n_turb
    except BaseException:
        for values in (x_all, y_all):
            if isinstance(values, np.memmap) and not values._mmap.closed:
                values._mmap.close()
        raise


@contextmanager
def _training_flat_cache(loader, max_batches=None, cache_dir=None):
    base = Path(cache_dir or os.environ.get("PIEEF_TABULAR_CACHE", Path(__file__).resolve().parents[2] / "artifacts/_tabular_cache")).resolve()
    base.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="owned_flat_", dir=base) as temporary:
        directory = Path(temporary).resolve()
        if directory.parent != base or not directory.name.startswith("owned_flat_"):
            raise RuntimeError("Temporary tabular cache escaped its owned directory")
        x = y = None
        try:
            x, y, _ = _loader_to_flat_xy(loader, max_batches, True, cache_dir=directory)
            yield x, y, directory
        finally:
            for values in (x, y):
                if isinstance(values, np.memmap) and not values._mmap.closed:
                    values._mmap.close()
            x = y = None
            gc.collect()


class ExternalMemoryXGBRegressor:
    """Native external-memory hist model with the existing predict/joblib interface."""

    def __init__(self, params: dict, n_estimators: int = 200):
        self.params, self.n_estimators = dict(params), int(n_estimators)
        self.booster_ = None

    def fit(self, x, y, cache_prefix):
        import xgboost as xgb
        rows_per_batch = max(1, min(8192, (16 * 2**20) // (x.shape[1] * 4)))

        class FixedRows(xgb.DataIter):
            def __init__(self):
                self.position = 0
                super().__init__(cache_prefix=str(cache_prefix), release_data=True, on_host=False)

            def reset(self):
                self.position = 0

            def next(self, input_data):
                if self.position == len(y):
                    return False
                end = min(self.position + rows_per_batch, len(y))
                input_data(data=x[self.position:end], label=y[self.position:end])
                self.position = end
                return True

        iterator = matrix = None
        try:
            iterator = FixedRows()
            matrix = xgb.ExtMemQuantileDMatrix(iterator, max_bin=256)
            self.booster_ = xgb.train(self.params, matrix, num_boost_round=self.n_estimators)
            self.n_features_in_ = int(x.shape[1])
            self.booster_.reset()
        finally:
            matrix = iterator = None
            gc.collect()
        return self

    def predict(self, x):
        if self.booster_ is None:
            raise ValueError("XGBoost model is not fitted")
        return self.booster_.inplace_predict(np.asarray(x, dtype=np.float32))

    def get_booster(self):
        return self.booster_

    def get_params(self, deep=True):
        return dict(self.params, n_estimators=self.n_estimators)

    def save_model(self, path):
        self.booster_.save_model(path)


def eval_tabular_model(model, loader, capacity_mw: float | np.ndarray, max_batches: int | None = None):
    """Evaluate an already fitted regressor on a separate chronological split."""
    truths, predictions = [], []
    for x, y, n_turb in _iter_flat_xy(loader, max_batches):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names, but LGBMRegressor was fitted with feature names", category=UserWarning)
            prediction = np.asarray(model.predict(x))
        if prediction.shape != y.shape:
            raise ValueError("Regressor predictions must match the flattened target grid.")
        truths.append(y.reshape(-1, n_turb))
        predictions.append(prediction.reshape(-1, n_turb))
    if not truths:
        raise ValueError("Empty tabular evaluation split")
    truth_mw = np.concatenate(truths, axis=0) * capacity_mw
    prediction_mw = np.maximum(np.concatenate(predictions, axis=0) * capacity_mw, 0)
    return _baseline_metrics(truth_mw, prediction_mw, capacity_mw), truth_mw, prediction_mw


def train_eval_hgbr(train_loader, test_loader, capacity_mw: float | np.ndarray, seed: int, max_batches: int | None = None, cache_dir=None):
    model = HistGradientBoostingRegressor(random_state=seed)
    with _training_flat_cache(train_loader, max_batches, cache_dir) as (x_train, y_train, _):
        model.fit(x_train, y_train)
    metrics, y_true, y_pred = eval_tabular_model(model, test_loader, capacity_mw, max_batches)
    return model, metrics, y_true, y_pred


def lightgbm_loss_options(loss_name: str | None = None, huber_delta: float = 0.1):
    name, delta = baseline_loss_spec(None, DLTrainConfig(loss_name=loss_name, huber_delta=huber_delta))
    if name == "huber":
        return {"objective": "huber", "alpha": delta}
    return {"objective": "regression_l1" if name == "l1" else "regression"}


def train_eval_lightgbm(train_loader, test_loader, capacity_mw: float | np.ndarray, seed: int, max_batches: int | None = None, cache_dir=None,
                        loss_name: str | None = None, huber_delta: float = 0.1):
    loss_options = lightgbm_loss_options(loss_name, huber_delta)
    try:
        import lightgbm as lgb

        model = lgb.LGBMRegressor(
            **loss_options,
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=64,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            verbosity=-1,
            force_col_wise=True,
            histogram_pool_size=64,
        )
    except ImportError as exc:
        raise RuntimeError("LightGBM is required for the LightGBM baseline") from exc

    with _training_flat_cache(train_loader, max_batches, cache_dir) as (x_train, y_train, _):
        model.fit(x_train, y_train)
    model.training_summary = {
        "loss_name": "huber" if loss_options["objective"] == "huber" else "l1" if loss_options["objective"] == "regression_l1" else "mse",
        "huber_delta": loss_options.get("alpha"),
        "actual_objective": model.booster_.params.get("objective"),
        "actual_alpha": model.booster_.params.get("alpha"),
        "lightgbm_parameters": model.get_params(),
    }
    metrics, y_true, y_pred = eval_tabular_model(model, test_loader, capacity_mw, max_batches)
    return model, metrics, y_true, y_pred


def train_eval_xgboost(train_loader, test_loader, capacity_mw: float | np.ndarray, seed: int, max_batches: int | None = None, cache_dir=None):
    try:
        import xgboost as xgb
    except Exception as exc:
        raise RuntimeError(f"xgboost import failed: {exc}")

    params = dict(
        objective="reg:squarederror",
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        tree_method="hist",
    )
    with _training_flat_cache(train_loader, max_batches, cache_dir) as (x_train, y_train, directory):
        if hasattr(xgb, "ExtMemQuantileDMatrix") and hasattr(xgb, "DataIter"):
            model = ExternalMemoryXGBRegressor(params, n_estimators=200)
            model.fit(x_train, y_train, directory / "xgb_cache")
        else:
            # Older test/runtime libraries may use the original path only for small inputs.
            if x_train.nbytes > 256 * 2**20:
                raise RuntimeError("This XGBoost version lacks external-memory quantile support for the complete dataset")
            model = xgb.XGBRegressor(n_estimators=200, **params)
            model.fit(x_train, y_train)
    metrics, y_true, y_pred = eval_tabular_model(model, test_loader, capacity_mw, max_batches)
    return model, metrics, y_true, y_pred


def train_eval_mlp(train_loader, test_loader, capacity_mw: float | np.ndarray, seed: int, max_batches: int | None = None, cache_dir=None):
    model = MLPRegressor(hidden_layer_sizes=(128, 64), activation="relu", random_state=seed, max_iter=200)
    with _training_flat_cache(train_loader, max_batches, cache_dir) as (x_train, y_train, _):
        model.fit(x_train, y_train)
    metrics, y_true, y_pred = eval_tabular_model(model, test_loader, capacity_mw, max_batches)
    return model, metrics, y_true, y_pred
