# PI-EEF

Code for **A physics-guided framework for robust wind power forecasting under non-ideal observations**.

Version: `vf-revision-20261028`.

PI-EEF combines masked reconstruction of power histories, differentiable weather calibration and turbine power curves, and graph attention across turbines. This release provides the revised **VF** implementation used by the main model: **V1 + F2 + F3 + F6**. See [Method and data protocol](docs/METHOD_AND_PROTOCOL.md) and [CHANGELOG.md](CHANGELOG.md).

## Quick start: CPU demonstration

The CPU demonstration was tested with Python 3.10.19 and the dependencies in `requirements.txt`. Use a new environment and run from the repository root:

```bash
python -m pip install -r requirements.txt
python scripts/run_public_demo.py
```

The demonstration generates a small synthetic dataset and runs three training stages on CPU using the revision data loader and VF model modules. Outputs are saved under `artifacts/public_demo/vf/`. No confidential farm records or pretrained weights are required.

**This is a functional demonstration, not a reproduction of the paper's scores.** It uses synthetic weather and power, fewer turbines and a reduced training workload. Its metrics must not be cited as paper results. Generated demo metrics and weights are not distributed as real-data experimental results.

## Source layout

| Component | Source |
|---|---|
| Revision data processing and indexed windows | `src/pieef/revision_data.py` |
| Forecast-vintage selection | `src/pieef/forecast_vintages.py` |
| Model assembly | `src/pieef/model_factory.py`, `src/pieef/models/e2e.py` |
| Masking, reconstruction and RevIN | `src/pieef/models/pretrain.py` |
| Calibration, density correction and power curves | `src/pieef/models/physics.py` |
| Normalised graph attention | `src/pieef/models/graph.py` |
| Forecast head and history anchor | `src/pieef/models/forecast.py`, `src/pieef/models/forecast_skip.py` |
| Stage-3 structured corruption | `src/pieef/forecast_corruption.py` |
| Three-stage optimisation | `src/pieef/train_main.py` |
| Production training entry point | `scripts/train.py` |

Legacy policies remain available for controlled comparisons. Omitting the VF switches does **not** automatically select the revised model. Use a VF configuration rather than an older smoke or demo configuration. This is a curated release of the PI-EEF core; it does not include the complete baseline-training suite or private benchmark orchestration.

## Paper configurations

`configs/vf_main_h16.yaml` is the 4-h template for five main-benchmark seeds. `configs/paper/` contains the 25 per-run templates for horizons 1, 4, 16, 48 and 96 steps and seeds 13, 21, 42, 87 and 3407. One step is 15 min. The templates preserve scientific settings and the existing relative `data/raw/` paths; only operational tracking references are sanitised. Original configuration hashes are recorded in release provenance. Private data are not bundled, so supply authorised inputs at those paths or set your own paths in a configuration copy.

First supply an authorised dataset matching the declared schema, then set its paths and weather protocol in a copy of the template:

```bash
python scripts/train.py --config configs/vf_main_h16.yaml --dry-run
python scripts/train.py --config configs/vf_main_h16.yaml
```

The production revision training path requires CUDA. The separate CPU-demo launcher does not relax that requirement or replace `scripts/train.py`. A dry run checks data loading and window shapes; it does not construct the model or establish scientific reproducibility.

Wind Farms A and B contain confidential operational data and are not distributed. Kelmarsh source data are public, but reproducing a score also requires the matching preprocessing, weather archive, split and configuration. Synthetic data cannot substitute for these inputs.

## Data and evaluation semantics

Historical SCADA inputs are filled causally; forecasting targets remain the original valid measurements. Normalisation statistics use the training period. Model-specific missingness indicators and physical branches are separate from the shared numerical inputs used by baselines.

Two weather protocols are supported:

- `author_prepared_ecmwf` uses the supplied weather series by target valid time. Author confirmation records the declared source; it is **not** independent proof of forecast-origin availability. The released main configurations retain `nwp_alignment_verified: false`.
- `forecast_vintages` selects a run with recorded availability no later than the origin, then interpolates leads within that run. Actual provenance metadata are required; the code does not manufacture issue or availability times.

**ECMWF timing and the 24-h benchmark.** The current manuscript describes the following supply schedule for Wind Farms A and B: a run initialised at 18:00 UTC on Day 1 becomes available at 08:00 Beijing Time (UTC+8) on Day 2 and supplies the target-time inputs for Day 3. Under this declared schedule, Wind Farm A's rolling 24-h forecasts issued from 00:00 to 07:45 on Day 2 use a batch that arrives after their origin, by up to 8 h. These are 32 of 96 daily 15-min origin slots, about one third. The full-day 24-h results therefore represent a partly non-causal offline comparison, not a verified operational 24-h evaluation. Giving the learning-based models the same NWP inputs does not remove this availability limitation.

The 15-min, 1-h, 4-h and 12-h horizons do not have this specific early-origin exception under the declared schedule. That timing deduction is not an independent audit of the archived forecast vintages. The [protocol document](docs/METHOD_AND_PROTOCOL.md#4-weather-timing-contracts) gives the cycle, interpolation and origin-time examples. This documentation clarification does not rerun experiments or change the archived predictions and scores.

See [release provenance](docs/PROVENANCE.md) for source/configuration identities and the recorded training and demonstration environments. The following commands inspect the environment, verify bundled source/configuration identities, and run the CPU checks:

```bash
python scripts/check_env.py
python scripts/verify_release.py
python -m unittest discover -s tests -v
```

## Outputs

Production runs save effective configuration and metrics, preprocessing statistics, indexed predictions, history/timing and `weights_stage1.pt`, `weights_stage2.pt`, `weights_best.pt` and `weights_last.pt`. Distinguish completed runs from failures or partial outputs; inspect `evaluation_split` in `metrics.json`.

Matching random seeds alone do not guarantee identical floating-point predictions across hardware or library versions. Record the environment and apply the relevant experiment's reproduction checks. This release does not imply new accuracy or significance results.

## License

See [LICENSE](LICENSE). Dataset access conditions are separate from the software license; the repository does not grant access to confidential operational data.
