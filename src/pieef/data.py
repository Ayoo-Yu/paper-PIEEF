from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

NWP_COLS = [
    "Direct_solar_radiation_10",
    "Total_sky_direct_short-wave_(solar)_radiation_at_surface_10",
    "Surface_pressure_10",
    "2_metre_temperature_10",
    "10_metre_U_wind_component_10",
    "10_metre V_wind_component_10",
    "wind_speed200_10",
    "wind_speed100_10",
    "wind_speed10_10",
]


@dataclass
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    handler: "RealDataHandler"
    num_turbines: int
    nwp_dim: int
    capacity_mw: float
    dist_tensor: torch.Tensor
    rel_coords_tensor: torch.Tensor


def _extract_power_cols(df: pd.DataFrame) -> list[str]:
    power_cols = [c for c in df.columns if c.startswith("power") and c[5:].isdigit()]
    if not power_cols:
        raise ValueError("No power columns found (expected power1..powerN).")
    return sorted(power_cols, key=lambda x: int(x[5:]))


def _load_coords(turbines_path: Path, num_turbines: int) -> np.ndarray:
    if not turbines_path.exists():
        raise FileNotFoundError(f"Missing turbines file: {turbines_path}")
    tdf = pd.read_csv(turbines_path)
    required = {"x_coord", "y_coord"}
    if not required.issubset(set(tdf.columns)):
        raise ValueError("turbines.csv must contain x_coord,y_coord")
    coords = tdf[["x_coord", "y_coord"]].values.astype(float)
    if len(coords) >= num_turbines:
        return coords[:num_turbines]

    # Synthetic expansion for scalability runs: tile existing layout with fixed offsets.
    expanded = [coords]
    need = num_turbines - len(coords)
    offset_step = max(float(np.ptp(coords[:, 0])) + 1.0, 1.0)
    copied = 0
    k = 1
    while copied < need:
        take = min(need - copied, len(coords))
        block = coords[:take].copy()
        block[:, 0] = block[:, 0] + k * offset_step
        expanded.append(block)
        copied += take
        k += 1
    return np.concatenate(expanded, axis=0)[:num_turbines]


def clean_outliers_physics(df: pd.DataFrame, power_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_clean = df.copy()
    valid_mask_df = pd.DataFrame(True, index=df.index, columns=power_cols)

    for i, p_col in enumerate(power_cols, start=1):
        w_col = f"turbine_ws{i}"
        has_wind = w_col in df_clean.columns

        outlier_mask = pd.Series(False, index=df_clean.index)

        small_neg_mask = (df_clean[p_col] < 0) & (df_clean[p_col] > -20.0)
        df_clean.loc[small_neg_mask, p_col] = 0.0

        deep_neg_mask = df_clean[p_col] <= -20.0
        outlier_mask |= deep_neg_mask

        if has_wind:
            wind = df_clean[w_col]
            power = df_clean[p_col]

            curtailment_mask = (wind > 5.0) & (power < 10.0)
            outlier_mask |= curtailment_mask

            bins = np.arange(0, 25, 1.0)
            bin_indices = np.digitize(wind, bins)
            for b in range(1, len(bins)):
                bin_mask = bin_indices == b
                if int(bin_mask.sum()) > 10:
                    bin_power = power[bin_mask]
                    mean_p = float(bin_power.mean())
                    std_p = float(bin_power.std())
                    sigma_mask = (bin_power < mean_p - 3 * std_p) | (bin_power > mean_p + 3 * std_p)
                    outlier_mask.loc[df_clean.index[bin_mask][sigma_mask]] = True

        valid_mask_df.loc[outlier_mask, p_col] = False
        df_clean.loc[outlier_mask, p_col] = np.nan
        df_clean[p_col] = df_clean[p_col].interpolate(method="linear", limit_direction="both")

    return df_clean, valid_mask_df


class RealDataHandler:
    def __init__(
        self,
        power_mat: np.ndarray,
        wind_mat: np.ndarray,
        nwp_mat: np.ndarray,
        mask_mat: np.ndarray,
        coords: np.ndarray,
        capacity_mw: float = 2.0,
        seq_len: int = 96,
        horizon: int = 1,
        train_ratio: float = 0.7,
        val_ratio: float = 0.1,
    ) -> None:
        self.seq_len = seq_len
        self.horizon = horizon
        self.capacity_mw = capacity_mw

        u_comp = nwp_mat[:, :, 4]
        v_comp = nwp_mat[:, :, 5]
        nwp_ws = np.sqrt(u_comp**2 + v_comp**2)
        wd_rad = np.arctan2(u_comp, v_comp)
        wd_sin = np.sin(wd_rad)
        wd_cos = np.cos(wd_rad)
        new_feats = np.stack([nwp_ws, wd_sin, wd_cos], axis=-1)
        nwp_mat_enhanced = np.concatenate([nwp_mat, new_feats], axis=-1)

        total_len = len(power_mat)
        train_end = int(total_len * train_ratio)
        val_end = int(total_len * (train_ratio + val_ratio))

        self.mask_full = mask_mat

        train_p, val_p, test_p = power_mat[:train_end], power_mat[train_end:val_end], power_mat[val_end:]
        train_w, val_w, test_w = wind_mat[:train_end], wind_mat[train_end:val_end], wind_mat[val_end:]
        train_n, val_n, test_n = (
            nwp_mat_enhanced[:train_end],
            nwp_mat_enhanced[train_end:val_end],
            nwp_mat_enhanced[val_end:],
        )

        self.capacity_kw = self.capacity_mw * 1000.0 if power_mat.max() > 100 else self.capacity_mw
        self.train_p_norm = train_p / self.capacity_kw
        self.val_p_norm = val_p / self.capacity_kw
        self.test_p_norm = test_p / self.capacity_kw

        self.scada_w_mean = float(np.nanmean(train_w))
        self.scada_w_std = float(np.nanstd(train_w) + 1e-5)
        self.train_w_norm = (train_w - self.scada_w_mean) / self.scada_w_std
        self.val_w_norm = (val_w - self.scada_w_mean) / self.scada_w_std
        self.test_w_norm = (test_w - self.scada_w_mean) / self.scada_w_std

        _, _, f_nwp = train_n.shape
        train_n_flat = train_n.reshape(-1, f_nwp)
        self.nwp_mean = np.mean(train_n_flat, axis=0)
        self.nwp_std = np.std(train_n_flat, axis=0) + 1e-5
        self.train_n_norm = (train_n - self.nwp_mean) / self.nwp_std
        self.val_n_norm = (val_n - self.nwp_mean) / self.nwp_std
        self.test_n_norm = (test_n - self.nwp_mean) / self.nwp_std

        n_turbines = len(coords)
        dist_raw = np.zeros((n_turbines, n_turbines))
        rel_coords = np.zeros((n_turbines, n_turbines, 2))
        for i in range(n_turbines):
            for j in range(n_turbines):
                vec = coords[i] - coords[j]
                rel_coords[i, j] = vec
                dist_raw[i, j] = np.linalg.norm(vec)

        max_dist = dist_raw.max() if dist_raw.max() > 0 else 1.0
        self.dist_matrix = dist_raw / max_dist
        self.rel_coords = rel_coords / max_dist
        self.dist_tensor = torch.FloatTensor(self.dist_matrix)
        self.rel_coords_tensor = torch.FloatTensor(self.rel_coords)

        self.train_set = self._create_dataset(self.train_p_norm, self.train_w_norm, self.train_n_norm)
        self.val_set = self._create_dataset(self.val_p_norm, self.val_w_norm, self.val_n_norm)
        self.test_set = self._create_dataset(self.test_p_norm, self.test_w_norm, self.test_n_norm)

        test_start_idx = val_end
        test_samples = len(self.test_set[0])
        target_indices = np.arange(test_samples) + test_start_idx + self.seq_len + self.horizon - 1
        target_indices = target_indices[target_indices < len(self.mask_full)]
        self.test_mask_target = self.mask_full[target_indices]

    def _create_dataset(self, p_norm: np.ndarray, w_norm: np.ndarray, n_norm: np.ndarray):
        vals_p, vals_w, vals_n_hist, vals_n_target, vals_y = [], [], [], [], []
        total_len = len(p_norm)
        end_idx = total_len - self.seq_len - self.horizon + 1

        for i in range(end_idx):
            vals_p.append(p_norm[i : i + self.seq_len])
            vals_w.append(w_norm[i : i + self.seq_len])
            vals_n_hist.append(n_norm[i : i + self.seq_len])
            target_idx = i + self.seq_len + self.horizon - 1
            vals_y.append(p_norm[target_idx])
            vals_n_target.append(n_norm[target_idx])

        return (
            torch.FloatTensor(np.array(vals_p)).unsqueeze(-1).permute(0, 2, 1, 3),
            torch.FloatTensor(np.array(vals_w)).unsqueeze(-1).permute(0, 2, 1, 3),
            torch.FloatTensor(np.array(vals_n_hist)).permute(0, 2, 1, 3),
            torch.FloatTensor(np.array(vals_n_target)),
            torch.FloatTensor(np.array(vals_y)),
        )

    def get_loaders(self, batch_size: int = 64) -> tuple[DataLoader, DataLoader, DataLoader]:
        train_data = TensorDataset(*self.train_set)
        val_data = TensorDataset(*self.val_set)
        test_data = TensorDataset(*self.test_set)
        return (
            DataLoader(train_data, batch_size=batch_size, shuffle=True),
            DataLoader(val_data, batch_size=batch_size, shuffle=False),
            DataLoader(test_data, batch_size=batch_size, shuffle=False),
        )


def build_data_bundle(
    dms_path: Path,
    turbines_path: Path,
    seq_len: int,
    horizon: int,
    batch_size: int,
    train_ratio: float,
    val_ratio: float,
    capacity_mw: float,
    max_turbines: int | None = None,
) -> DataBundle:
    if not dms_path.exists():
        raise FileNotFoundError(f"Missing dataset: {dms_path}")

    df = pd.read_csv(dms_path)

    power_cols = _extract_power_cols(df)
    if max_turbines is not None:
        if max_turbines <= 0:
            raise ValueError("max_turbines must be positive")
        if max_turbines <= len(power_cols):
            power_cols = power_cols[:max_turbines]
        else:
            base_cols = list(power_cols)
            rng = np.random.default_rng(2026)
            sampled = rng.integers(0, len(base_cols), size=max_turbines - len(base_cols))
            for offset, src_idx in enumerate(sampled):
                idx = len(base_cols) + offset
                src_col = base_cols[int(src_idx)]
                new_col = f"power{idx + 1}"
                df[new_col] = df[src_col].values

                src_wind = f"turbine_ws{int(src_idx) + 1}"
                new_wind = f"turbine_ws{idx + 1}"
                if src_wind in df.columns and new_wind not in df.columns:
                    df[new_wind] = df[src_wind].values
                power_cols.append(new_col)
    num_turbines = len(power_cols)
    coords = _load_coords(turbines_path, num_turbines)

    df_clean, valid_mask_df = clean_outliers_physics(df, power_cols)

    power_matrix = df_clean[power_cols].values.astype(float)
    mask_matrix = valid_mask_df.values.astype(bool)

    wind_cols = [f"turbine_ws{i+1}" for i in range(num_turbines)]
    if all(c in df_clean.columns for c in wind_cols):
        wind_matrix = pd.DataFrame(df_clean[wind_cols].values).interpolate(limit_direction="both").fillna(0.0).values
    else:
        wind_matrix = np.zeros_like(power_matrix)

    existing_nwp = [c for c in NWP_COLS if c in df_clean.columns]
    if len(existing_nwp) < 6:
        raise ValueError("NWP columns are incomplete in DMS.csv")

    feature_matrix = df_clean[existing_nwp].values.astype(float)
    nwp_matrix = np.tile(feature_matrix[:, np.newaxis, :], (1, num_turbines, 1))

    handler = RealDataHandler(
        power_matrix,
        wind_matrix,
        nwp_matrix,
        mask_matrix,
        coords,
        capacity_mw=capacity_mw,
        seq_len=seq_len,
        horizon=horizon,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    train_loader, val_loader, test_loader = handler.get_loaders(batch_size=batch_size)

    return DataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        handler=handler,
        num_turbines=num_turbines,
        nwp_dim=int(handler.train_n_norm.shape[-1]),
        capacity_mw=capacity_mw,
        dist_tensor=handler.dist_tensor,
        rel_coords_tensor=handler.rel_coords_tensor,
    )
