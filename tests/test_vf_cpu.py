"""CPU checks for the published VF mechanisms, using only synthetic arrays.

Optional publication check: set PIEEF_REFERENCE_SRC to the frozen production
``src`` directory. Two separate processes then compare state, intermediate
outputs, loss and gradients exactly, without importing either tree into the other.
"""
from __future__ import annotations

import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from vf_fixture import install_source, model_and_inputs, forward
install_source()
import numpy as np
import torch
import torch.nn.functional as F

from pieef.forecast_corruption import ForecastCorruption
from pieef.models.graph import DynamicDirectionalGraphLayer
from pieef.models.pretrain import RevIN
from pieef.train_main import TrainConfig, _configure_stage3, _forecast_loss


class VFCPUChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        if torch.cuda.is_initialized():
            raise RuntimeError("CPU verification must not initialize CUDA")

    def test_forward_backward_and_frozen_decoder(self):
        model, inputs = model_and_inputs()
        _configure_stage3(model, "joint", "frozen", reconstruction_weight=.05)
        result = forward(model, inputs)
        self.assertEqual(tuple(result["prediction"].shape), (2, 3))
        for name, tensor in result.items():
            self.assertTrue(torch.isfinite(tensor).all(), name)
        loss = _forecast_loss(result["prediction"], inputs["target"], TrainConfig(huber_delta=.1))
        loss.backward()
        parameters = dict(model.named_parameters())
        for group in ("pi_embedding.", "mae_encoder.", "dcn.calib_net.", "dcn.power_curve.", "graph.",
                      "forecast_head.", "forecast_skip.", "revin."):
            gradients = [p.grad for key, p in parameters.items() if key.startswith(group) and p.grad is not None]
            self.assertTrue(gradients, group)
            self.assertTrue(all(torch.isfinite(g).all() for g in gradients), group)
            self.assertTrue(any(torch.count_nonzero(g) > 0 for g in gradients), group)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.mae_head.parameters()))
        self.assertFalse(torch.cuda.is_initialized())

    def test_checkpoint_strict_roundtrip_and_wrong_architecture_rejected(self):
        model, inputs = model_and_inputs()
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        state = torch.load(buffer, map_location="cpu", weights_only=True)
        restored, _ = model_and_inputs()
        restored.load_state_dict(state, strict=True)
        with torch.no_grad():
            a, b = forward(model, inputs), forward(restored, inputs)
        for key in a:
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
        wrong, _ = model_and_inputs(forecast_skip_policy="none")
        with self.assertRaises(RuntimeError):
            wrong.load_state_dict(state, strict=True)

    def test_f1_exact_twenty_percent_boundary_and_hidden_value_invariance(self):
        layer = RevIN(2, missing_policy="train_stats_20",
                      training_mean=[.3, .4], training_stdev=[.1, .2])
        x = torch.arange(20, dtype=torch.float32).reshape(1, 10, 2) / 20
        mask = torch.ones_like(x)
        mask[:, :2, 0] = 0  # exactly 20% -> training statistics
        mask[:, :3, 1] = 0  # 30% -> statistics of visible entries
        layer(x, "norm", mask)
        torch.testing.assert_close(layer.mean[0, 0, 0], torch.tensor(.3), rtol=0, atol=0)
        torch.testing.assert_close(layer.stdev[0, 0, 0], torch.tensor(.1), rtol=0, atol=0)
        torch.testing.assert_close(layer.mean[0, 0, 1], x[0, :3, 1].mean())
        original_mean, original_std = layer.mean.clone(), layer.stdev.clone()
        changed = x.clone()
        changed[mask.bool()] = 1000
        layer(changed, "norm", mask)
        self.assertTrue(torch.equal(layer.mean, original_mean))
        self.assertTrue(torch.equal(layer.stdev, original_std))

    def test_f2_deterministic_corruption_only_changes_scada(self):
        _, inputs = model_and_inputs()
        batch = (inputs["power"], inputs["wind"], inputs["nwp"], inputs["target_nwp"], inputs["target"])
        originals = [x.clone() for x in batch]
        config = dict(policy="f2_block", coords=np.array([[0, 0], [1, 0], [3, 0]]),
                      scada_wind_std=2., seed=13, num_turbines=3, seq_len=10)
        a, b = ForecastCorruption(**config), ForecastCorruption(**config)
        initial_rng = torch.get_rng_state().clone()
        observed_channels = set()
        for _ in range(64):
            output, power_mask, wind_mask = a.corrupt(batch)
            reference, ref_pm, ref_wm = b.corrupt(batch)
            for first, second in zip(output, reference):
                self.assertTrue(torch.equal(first, second))
            for actual, expected in ((power_mask, ref_pm), (wind_mask, ref_wm)):
                self.assertEqual(actual is None, expected is None)
                if actual is not None:
                    self.assertTrue(torch.equal(actual, expected))
            for index in (2, 3, 4):
                self.assertIs(output[index], batch[index])
            if power_mask is not None:
                for sample, scene in enumerate(a.last_scenarios):
                    observed_channels.add(scene["channels"])
                    start, length = scene["start_index"], scene["duration_steps"]
                    self.assertGreaterEqual(length, 4)
                    self.assertLessEqual(start + length, 10)
                    for channel, array, mask in (("power", output[0], power_mask), ("wind", output[1], wind_mask)):
                        if channel == "wind" and scene["channels"] == "power":
                            self.assertFalse(mask[sample].any())
                            continue
                        selected = torch.where(mask[sample, :, start, 0] > 0)[0]
                        self.assertEqual(len(selected), scene["turbine_count"])
                        old = batch[0 if channel == "power" else 1]
                        expected = old[sample, selected, start - 1, 0] if start else torch.zeros(len(selected))
                        torch.testing.assert_close(array[sample, selected, start:start+length, 0],
                                                   expected[:, None].expand(-1, length), rtol=0, atol=0)
        self.assertEqual(observed_channels, {"power", "both"})
        self.assertTrue(torch.equal(initial_rng, torch.get_rng_state()))
        for actual, original in zip(batch, originals):
            self.assertTrue(torch.equal(actual, original))

    def test_f3_attention_matches_cosine_temperature_and_additive_sensitivity(self):
        torch.manual_seed(42)
        layer = DynamicDirectionalGraphLayer(8, use_direction_bias=False,
                                             use_sensitivity_gate=True, score_policy="cosine_temperature")
        x = torch.randn(2, 3, 8)
        power = torch.tensor([.15, .4, .8]).reshape(1, 3, 1).repeat(2, 1, 1)
        layer(x, torch.ones_like(power), power, torch.zeros(3, 3), torch.zeros(3, 3, 2), torch.ones(2, 2))
        scores = layer.score_scale() * (F.normalize(layer.query(x), dim=-1, eps=1e-6) @
                                        F.normalize(layer.key(x), dim=-1, eps=1e-6).transpose(-1, -2))
        expected = (scores + layer.sensitivity_weight * (power * (1-power)).transpose(-1, -2)).softmax(-1)
        torch.testing.assert_close(layer.last_attn_weights, expected, rtol=0, atol=0)
        with torch.no_grad():
            layer.query.weight.zero_(); layer.query.bias.zero_()
            layer.key.weight.zero_(); layer.key.bias.zero_()
        output = layer(x, power, power, torch.zeros(3, 3), torch.zeros(3, 3, 2), torch.ones(2, 2))
        self.assertTrue(torch.isfinite(output).all())
        torch.testing.assert_close(layer.last_attn_weights.sum(-1), torch.ones(2, 3))

    def test_f4_target_weather_and_physical_power_receive_gradients(self):
        model, _ = model_and_inputs()
        head = model.forecast_head
        graph = torch.randn(2, 3, 16)
        wind = torch.full((2, 3, 1), 8.)
        weather = torch.randn(2, 3, 12, requires_grad=True)
        physical = torch.full((2, 3, 1), .4, requires_grad=True)
        head(graph, wind, x_nwp_target=weather, p_phys=physical).sum().backward()
        for value in (weather, physical):
            self.assertTrue(torch.isfinite(value.grad).all())
            self.assertGreater(torch.count_nonzero(value.grad), 0)

    def test_f6_fully_missing_forecast_ignores_hidden_history(self):
        model, inputs = model_and_inputs()
        inputs["mask"] = torch.ones_like(inputs["power"])
        with torch.no_grad():
            a = forward(model, inputs)["prediction"]
            self.assertFalse(model.forecast_skip.last_terms["observed"].any())
            torch.testing.assert_close(model.forecast_skip.last_terms["last_obs"],
                                       model.revin.affine_bias[None].expand(2, -1), rtol=0, atol=0)
            changed = dict(inputs, power=inputs["power"] + 1000.)
            b = forward(model, changed)["prediction"]
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.isfinite(a).all())

    @unittest.skipUnless(os.environ.get("PIEEF_REFERENCE_SRC"), "set PIEEF_REFERENCE_SRC for the release source-equivalence check")
    def test_frozen_source_forward_loss_and_gradient_exact_equivalence(self):
        reference = Path(os.environ["PIEEF_REFERENCE_SRC"])
        self.assertTrue((reference / "pieef/models/e2e.py").is_file())
        helper = Path(__file__).with_name("vf_fixture.py")
        public = Path(__file__).resolve().parents[1] / "src"
        with tempfile.TemporaryDirectory() as folder:
            saved = []
            for index, source in enumerate((reference, public)):
                output = Path(folder) / f"snapshot{index}.pt"
                subprocess.run([sys.executable, str(helper), "--source", str(source), "--snapshot", str(output)],
                               check=True, capture_output=True, text=True, timeout=90)
                saved.append(torch.load(output, map_location="cpu", weights_only=True))
            def same(a, b, path="snapshot"):
                if isinstance(a, dict):
                    self.assertEqual(a.keys(), b.keys(), path)
                    for key in a:
                        same(a[key], b[key], path + "/" + key)
                elif isinstance(a, torch.Tensor):
                    torch.testing.assert_close(a, b, rtol=0, atol=0, msg=path)
                else:
                    self.assertEqual(a, b, path)
            same(*saved)
            self.assertFalse(saved[0]["cuda_initialized"])


if __name__ == "__main__":
    unittest.main()
