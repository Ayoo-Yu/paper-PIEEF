# Production source and experiment provenance

This release contains the **VF model used for the revised paper**, identified by `energy-r3-vf-F2F3F6`. It is not the earlier F1/F4-only model and not the initial manuscript implementation.

## Frozen scientific source

The scientific Python files listed in `provenance/production_source_manifest.json` are copied without changing their bytes from the retained production source. Their SHA-256 values match the corresponding members of the frozen deployment archive:

```text
round3_code_opt.tar.gz
e72d4eedd99531def8b69334f690b8c90204f29a49ba3a38a59eb0d3794bab1d
```

All 105 Python files under `src/` and `scripts/` in that archive matched the retained source during the audit. The public manifest lists the 23 files required for this main-model release; operational cloud dispatch and credential-management scripts are excluded.

The development ancestor is commit `d1f2fa338ca7c1e9bbcf66e2c0a7c80a7497cea1`. The archived source also contains subsequent fixes to `models/pretrain.py` and `train_main.py`, and the optimized `forecast_corruption.py`. Checking out the ancestor commit alone would therefore omit part of the source used for the completed batch. The per-file hashes are the authoritative byte-level identifiers for this release.

The historical configurations retained older `tracking.code_git_commit=4844f3e1...` and `tracking.source_manifest_sha256=69ad299e...` fields from the F1/F4 base. **Those inherited fields do not identify the final VF source.** They are recorded explicitly as ancestry metadata, rather than presented as fresh VF execution hashes. Public configuration metadata is updated separately; the original configuration hashes and completed-run identifiers remain in the manifest.

The audit matched the retained deployment bundle, frozen configuration manifest and completed-run metadata. The original metrics files do not contain a newly bound per-run digest of the entire VF source. This release does not claim otherwise or retroactively rewrite those original records.

## Main-table identity

The main table comprises 25 completed test runs: five horizons, `[1, 4, 16, 48, 96]` steps at 15 minutes per step, and five seeds, `[13, 21, 42, 87, 3407]`. Every original YAML matches its stored manifest hash, and its parsed contents exactly match the configuration saved in that run's `metrics.json`. Run identifiers, original YAML hashes and metric-file hashes are included in the public manifest.

The unchanged scientific settings include:

| Setting | Value |
|---|---|
| History | 96 time steps |
| Main-site turbines | 29 |
| Encoder / graph depth | 3 / 1 |
| Hidden dimension | 128 |
| Mask ratio / joint pre-training mask | 0.4 / disabled |
| RevIN sparse-observation policy | `train_stats_20` |
| Target-weather prediction head | `target_nwp_physics_mlp` |
| Graph score | `cosine_temperature` |
| Forecast anchor | `last_obs_linear` |
| Stage 3 corruption | `f2_block` |
| Stage 3 optimization / decoder | joint / frozen |
| Batch size / stage budgets | 16 / 5, 5, 20 epochs |
| Stage learning rates | 0.001, 0.001, 0.0003 |
| Forecast loss | Huber, delta 0.1 in capacity-normalized power |
| Early stopping / checkpoint criterion | patience 5 / validation NRMSE |

`record_gradient_conflicts` is enabled in the five 4-hour main runs and disabled at the other horizons. The original per-run configurations preserve this diagnostic setting. The generic `epochs` and `lr` fields also remain as historical configuration fields; the main-model trainer uses the explicit three-stage budgets and learning rates.

## Mechanisms retained in VF

The F2 sampler corrupts only the Stage 3 forecasting path. Each training batch has a probability of 0.5 of being corrupted; selected samples independently draw the anchor turbine, 1–10 neighboring turbines, a contiguous block of 4–96 time steps, its position and the channel choice. Power-only and simultaneous power/wind corruption are equiprobable. The sampler uses a separate NumPy random stream. Targets and weather inputs are unchanged; validation and final evaluation use their specified inputs without this training augmentation. This continuous training sampler is distinct from the fixed 39-scenario evaluation matrix.

F3 normalizes queries and keys before their dot product. The score adds the directional bias, terrain bias and learned-weight sensitivity term. Its learnable scale is initialized at 10 and capped at 50. It replaces the earlier multiplicative sensitivity gate. The symmetric sensitivity option retained in the main table is based on normalized physical power, not a claim to be the exact derivative of a turbine power curve.

F6 adds a gated last-visible-power anchor and a zero-initialized linear history branch in the forecast head's RevIN-normalized space, before denormalization. The full-window missing case uses the training-statistics fallback. The pre-training decoder is frozen in Stage 3; joint fine-tuning updates the other intended modules.

The model also retains the earlier F1 training-statistics fallback and F4 target-weather/physical-power head, along with density correction, terrain edge features, monotone power curves and the configured hard operating limits. “F2F3F6” names the final additions, not an omission of those base components.

## Data-processing route

Use the supplied revision configurations with `data.revision_protocol`. `scripts/train.py` calls `build_data_bundle`, which dispatches to `revision_data.build_revision_bundle` when that protocol is present. This is the route used by all 25 audited main runs.

That route retains raw valid measured power as the prediction target, marks invalid targets as missing, causally forward-fills historical invalid/missing inputs, and uses training-period statistics for normalization. Candidate event labels describe recorded observations; valid flagged targets are not replaced by interpolated values. The original main-site configuration uses the author-prepared ECMWF fields and its recorded feature semantics. It does not by itself reconstruct forecast issue/availability times. The retained results and manuscript describe the corresponding weather-data scope.

`data.py` also retains an older, backward-compatible preprocessing branch. Omitting `revision_protocol` selects that legacy branch and **does not reproduce the revised main-table experiment**. The public revision examples must retain the protocol, raw-target policy and causal input-filling settings.

## Runtime and reproduction limits

Retained metadata for the five cloud execution nodes records PyTorch `2.5.1+cu124`, with RTX 4090 D and RTX 3080 Ti hardware. The inspected historical requirements file is unpinned; it is not a complete package lock, and the exact Python version was not established from those node records. Current setup instructions should distinguish a supported installation from an exact historical environment.

The main entry point is `scripts/train.py --config <revision-config.yaml>`. Source publication does not include confidential SCADA data, node credentials, or an assertion that a different GPU/software stack will reproduce every floating-point value bit for bit. Matching the original data, configuration and environment remains necessary for reproducing the reported numerical results. This provenance audit did not rerun training.
