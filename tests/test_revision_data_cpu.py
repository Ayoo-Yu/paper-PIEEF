"""Synthetic file-level checks of the measured-target revision data path."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from vf_fixture import install_source
install_source()
import numpy as np
import pandas as pd
import torch

from pieef.artifacts import save_preprocessing
from pieef.data import NWP_COLS, build_data_bundle
from pieef.model_factory import build_main_model
from pieef.power_training_stats import ensure_training_power_statistics, check_model_training_statistics
from pieef.revision_data import ForecastWindows, _causal_fill, require_verified_forecast_input


class RevisionDataCPUChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.data = self.root / "synthetic_scada.csv"
        self.coords = self.root / "synthetic_turbines.csv"
        count = 120
        self.frame = pd.DataFrame({"Timestamp": pd.date_range("2024-01-01", periods=count, freq="15min", tz="UTC")})
        for turbine in (1, 2, 3):
            self.frame[f"power{turbine}"] = 600 + 200 * np.sin(np.arange(count) / 10 + turbine)
            self.frame[f"turbine_ws{turbine}"] = 7 + np.sin(np.arange(count) / 10 + turbine)
        self.frame.loc[98:100, "power1"] = [1000., np.nan, 0.]
        self.frame.to_csv(self.data, index=False)
        pd.DataFrame(dict(turbine_id=["S1", "S2", "S3"], x_coord=[0, 400, 850],
                          y_coord=[0, 100, -80], elevation_m=[100, 130, 115])).to_csv(self.coords, index=False)
        # This is an explicitly artificial forecast archive, available before all
        # synthetic origins. It does not certify any real weather data source.
        leads = np.arange(97, dtype=float)
        values = np.ones((1, len(leads), len(NWP_COLS)), dtype=np.float32)
        values[0] += .1 * np.sin(leads[:, None] / 8 + np.arange(len(NWP_COLS)))
        values[:, :, 2] = 101325 + 100 * np.sin(leads / 8)
        values[:, :, 3] = 288 + np.cos(leads / 8)
        values[:, :, 4] = 7 + np.sin(leads / 8)
        values[:, :, 5] = 2 + .2 * np.cos(leads / 8)
        archive = self.root / "synthetic_weather.npz"
        np.savez(archive, reference_time=np.array(["2023-12-31T00:00:00+00:00"]),
                 available_time=np.array(["2023-12-31T06:00:00+00:00"]), lead_hours=leads,
                 feature_names=np.array(NWP_COLS), values=values)
        self.protocol = dict(power_unit="kW", timezone="UTC", step_minutes=15,
                             nwp_timing_rule="forecast_vintages", nwp_alignment_verified=True,
                             nwp_vintages_path=str(archive), nwp_feature_columns=list(NWP_COLS))

    def tearDown(self):
        self.temporary.cleanup()

    def bundle(self):
        require_verified_forecast_input(self.protocol)
        return build_data_bundle(self.data, self.coords, 8, 4, 2, .7, .1, 2., revision_protocol=self.protocol)

    def test_actual_revision_loader_keeps_targets_raw_and_history_causal(self):
        bundle = self.bundle()
        dataset = bundle.test_loader.dataset
        self.assertIsInstance(dataset, ForecastWindows)
        for target_index, expected in ((99, float("nan")), (100, 0.)):
            row = int(np.flatnonzero(dataset.indices == target_index)[0])
            actual = float(dataset[row][-1][0])
            if np.isnan(expected):
                self.assertTrue(np.isnan(actual))
            else:
                self.assertEqual(actual, expected)
        self.assertEqual(float(dataset.power[99, 0]), .5)  # 1000 kW / 2 MW
        self.assertTrue(torch.isnan(dataset.targets[99, 0]))
        self.frame.loc[100:, "power1"] = 1900
        self.frame.to_csv(self.data, index=False)
        second = self.bundle().test_loader.dataset
        self.assertTrue(torch.equal(dataset.power[:100], second.power[:100]))
        filled = _causal_fill(np.array([[np.nan], [2.], [np.nan], [9.]]))
        np.testing.assert_array_equal(filled[:, 0], [0., 2., 2., 9.])

    def test_held_out_values_cannot_change_training_statistics(self):
        first = self.bundle()
        stats = ensure_training_power_statistics(first)
        self.frame.loc[96:, "power1"] = 1900
        self.frame.loc[96:, "turbine_ws1"] = 99
        self.frame.to_csv(self.data, index=False)
        second = self.bundle()
        updated = ensure_training_power_statistics(second)
        for key in ("mean_pu", "stdev_pu"):
            np.testing.assert_array_equal(stats[key], updated[key])
        np.testing.assert_array_equal(first.handler.nwp_mean, second.handler.nwp_mean)
        self.assertEqual(first.handler.scada_w_mean, second.handler.scada_w_mean)
        self.assertEqual(first.handler.scada_w_std, second.handler.scada_w_std)

    def test_factory_saved_preprocessing_and_weights_rebuild_strictly(self):
        bundle = self.bundle()
        options = dict(d_model=16, encoder_depth=1, use_density=True, use_terrain_bias=True,
                       revin_missing_policy="train_stats_20", use_mask_indicators=True,
                       forecast_head_policy="target_nwp_physics_mlp", graph_score_policy="cosine_temperature",
                       forecast_skip_policy="last_obs_linear", rated_wind_speed=10., cut_out_speed=22.)
        model = build_main_model(bundle, 8, options, torch.device("cpu")).eval()
        save_preprocessing(self.root / "synthetic_run", bundle, revin_missing_policy="train_stats_20")
        path = self.root / "synthetic_run/weights_best.pt"
        torch.save(model.state_dict(), path)
        with np.load(path.parent / "preprocessing.npz", allow_pickle=False) as saved:
            restored_bundle = self.bundle()
            ensure_training_power_statistics(restored_bundle, dict(saved))
        restored = build_main_model(restored_bundle, 8, options, torch.device("cpu")).eval()
        restored.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
        check_model_training_statistics(restored, restored_bundle)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)
        with self.assertRaises(ValueError):
            build_main_model(bundle, 8, dict(options, revin_training_mean=[0, 0, 0]), torch.device("cpu"))
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
