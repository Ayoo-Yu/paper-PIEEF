# Method and data protocol

This document describes the revised VF implementation and its main-benchmark configuration. It distinguishes the scientific settings from the reduced synthetic demonstration. Release identities and archived environments are recorded in [PROVENANCE.md](PROVENANCE.md).

## 1. Revised model and explicit policies

The revised main model is **VF = V1 + F2 + F3 + F6**. V1 already includes F1 and F4. These names identify development components, not separate datasets or evaluation splits.

| Component | Configuration | Implemented behaviour |
|---|---|---|
| F1 | `revin_missing_policy: train_stats_20` | Use training-period power moments when at most 20% of the history is visible to RevIN. |
| F4 | `forecast_head_policy: target_nwp_physics_mlp` | Include target-time NWP and the calibrated physical-power estimate in the forecasting head. |
| F2 | `stage3_forecast_corruption: f2_block` | Train the forecasting branch with structured corruption of historical SCADA inputs. |
| F3 | `graph_score_policy: cosine_temperature` | Normalise Q and K, learn attention temperature, and add the operating-point term to attention scores. |
| F6 | `forecast_skip_policy: last_obs_linear` | Add a last-observed-power anchor and a learned linear history branch in normalised power space. |
| Joint fine-tuning | `stage3_mode: joint`, `stage3_decoder_policy: frozen` | Update the encoder, calibration and forecasting modules while keeping reconstruction-decoder parameters fixed. |

Direction and terrain biases, density correction, the calibrated B-spline power curve and explicit removal indicators remain enabled. The main model uses `joint_masking: false`; F2 supplies the separate forecasting-stage mechanism for power-and-wind corruption. Legacy scoring and sequential-training options remain in the code for controlled comparisons. Their presence does not mean that a legacy configuration is equivalent to VF.

**F2 sampling.** A training batch is selected for corruption with probability 0.5. Within a selected batch, samples receive independently drawn blocks involving 1–10 nearby turbines and 4–96 historical steps, bounded by the available turbine count and history length. At 15-min resolution, the duration range is 1–24 h. Power-only and power-and-wind blocks are equally likely. Half of the blocks end at the forecast origin; the others are placed within the history. Removed values use the preceding available value, or zero when there is no preceding value. Explicit masks retain their removal locations. Targets and NWP inputs are unchanged. A dedicated NumPy generator is seeded from the configured corruption seed, defaulting to the training seed. This random training sampler is separate from a fixed evaluation-fault matrix.

**F3 scoring.** Q and K are L2-normalised. The positive temperature starts at 10 and is capped at 50. Direction, terrain and the learned operating-point contribution are additive score terms before softmax. The symmetric operating-point input is based on physical power in per-unit form. Learned attention is a model interaction weight; it is not itself a measurement of a wake or a guarantee of causal attribution.

**F6 anchoring.** The skip branch uses the most recent visible power and a shared linear map of the history in affine RevIN space. The linear map starts at zero and its sigmoid mixing parameter starts at 0.5. Masked steps are excluded from the linear input. Under the VF missing-history policy, an all-missing history uses the corresponding training mean as its anchor.

## 2. Main-benchmark settings

`configs/vf_main_h16.yaml` groups the five 4-h runs. The 25 individual configurations in `configs/paper/` cover five horizons and five seeds. They preserve the scientific settings and existing relative data paths of the archived runs; only operational tracking references are sanitised. Private data are not bundled, and users must supply authorised inputs at the configured paths.

| Setting | Main configuration |
|---|---|
| Sampling and history | 15 min; 96 historical steps (24 h) |
| Forecast horizons | 1, 4, 16, 48, 96 steps: 15 min, 1 h, 4 h, 12 h, 24 h |
| Seeds | 13, 21, 42, 87, 3407 |
| Chronological split | 70% training, 10% validation, 20% test |
| Width / encoder depth / graph depth | 128 / 3 / 1 |
| Reconstruction mask ratio | 0.4 |
| B-spline control points | 15 |
| Batch and arithmetic | 16; FP32 |
| Stage epoch budgets | 5 / 5 / 20 |
| Stage learning rates | 0.001 / 0.001 / 0.0003 |
| Optimiser | Adam |
| Forecast loss | Huber, delta 0.1 |
| Checkpoint / early stopping | Validation NRMSE; patience 5 |
| Stage-3 auxiliary weights | Reconstruction 0.05; calibration 0.05 |
| Auxiliary decay | Exponential factor `exp(-0.1 * epoch)` |

Stage 1 trains the reconstruction branch; Stage 2 trains the weather-to-power calibration branch; Stage 3 performs joint forecasting fine-tuning. A frozen decoder still passes gradients to the encoder. The generic `epochs` and `lr` fields retained for other model paths do not replace the three PI-EEF stage budgets.

The production revision path in `scripts/train.py` requires CUDA. Its `--dry-run` mode checks data loading and window shapes; it does not construct the model. A successful dry run is not a completed training run or a replication of an archived score.

## 3. Input schema and preprocessing

Data access is separate from software access. Wind Farms A and B are confidential and are not bundled. Supply authorised files and preserve consistent turbine ordering, units, time zone, cadence and capacity. The formal template uses these files:

| File | Required content |
|---|---|
| SCADA/weather CSV | `Timestamp`; `power1` … `powerN` in kW; `turbine_ws1` … `turbine_wsN` in m/s. The prepared-weather protocol also requires the nine weather columns below. |
| Turbine CSV | `turbine_id`, `x_coord`, `y_coord`, `elevation_m`; one row per turbine in the same order as SCADA columns. Coordinates and elevations are in metres. |
| Manufacturer curve CSV | `wind_speed_ms` and `power_pu`, describing the supplied turbine curve. |

The prepared-weather column names are preserved exactly for compatibility:

```text
Direct_solar_radiation_10
Total_sky_direct_short-wave_(solar)_radiation_at_surface_10
Surface_pressure_10
2_metre_temperature_10
10_metre_U_wind_component_10
10_metre V_wind_component_10
wind_speed200_10
wind_speed100_10
wind_speed10_10
```

Wind quantities use m/s, pressure uses Pa and temperature uses K. The loader additionally derives wind magnitude and directional sine/cosine from the 10-m U/V components. In the released main configuration, the target physical-speed input uses index 7, corresponding to the 100-m speed, while graph direction comes from the 10-m vector. A change of directional height or a CFD correction defines a different variant and must not silently replace these inputs.

Historical invalid/missing power and wind are forward-filled within each turbine, using zero only when no previous valid observation exists. There is no backward filling of historical SCADA. Forecast targets are the original valid power measurements, not filled or interpolated history. Normalisation and statistical screening parameters use the training period. Valid low-power or other flagged observations remain targets; observational labels do not establish the operational cause of an event.

For target index `i`, horizon `h` and history length `L`, the history spans `i-h-L+1` through `i-h`, inclusive. Weather supplied to the target branch corresponds to `i`. The meaning of its availability is determined by the selected weather protocol below, not by this indexing alone.

Terrain edge features use elevation differences and differences divided by horizontal inter-turbine distance. They do not constitute a resolved terrain-flow simulation. Shared numerical inputs and targets across methods also do not imply identical model interfaces: PI-EEF has its own removal masks and physical branches, while Persistence uses its designated historical-power rule.

## 4. Weather timing contracts

### Prepared ECMWF series

`nwp_timing_rule: author_prepared_ecmwf` consumes the supplied weather series at target valid times. It requires `nwp_preparation_author_confirmed: true`. This records the author's confirmation of the prepared source and does not independently verify availability at every forecast origin. In the archived main template, `nwp_alignment_verified` is explicitly false.

The current manuscript (Sections 3.1 and 3.4, and Supplementary Section S3.4) describes the following provider/author-declared supply schedule for Wind Farms A and B. Times below are Beijing Time (UTC+8), unless marked UTC.

| Event | Declared timing |
|---|---|
| ECMWF initialisation | 18:00 UTC on Day 1, or 02:00 on Day 2 |
| Batch availability after computation and transmission | 08:00 on Day 2 |
| Forecast coverage | 48 hourly steps, from 02:00 on Day 2 through 01:00 on Day 4 |
| Assigned target day | Day 3, from 00:00 through 23:45 on the 15-min grid |
| Hourly leads used for the target day and interpolation | 22–46 h from initialisation |

Each target value is linearly interpolated between the two bracketing hourly leads of the assigned batch; an exact hourly target uses that lead directly. For example, 23:45 on Day 3 uses the hourly leads for 23:00 on Day 3 and 00:00 on Day 4 from the same batch, with weights 0.25 and 0.75. This describes the manuscript's declared preparation rule. The prepared-series loader consumes the supplied values and does not independently reconstruct or validate their issue and availability timestamps.

**Origin-time limitation at 24 h.** A rolling 24-h forecast for a Day 3 target has its origin at the same clock time on Day 2. Origins from 00:00 to 07:45 therefore precede the assigned batch's 08:00 availability. On a complete daily 15-min grid, these are 32 of 96 origin slots, about one third; the earliest uses a batch received 8 h after the origin. A 24-h forecast issued at 12:00 on Day 2 does have that batch available, but this example does not establish availability for the earlier origins.

The full-day Wind Farm A 24-h benchmark, including its 24-h fault comparisons, is consequently a partly non-causal offline evaluation under the declared schedule. Common NWP inputs establish a shared input condition for the learning-based models, not an operational availability guarantee. Persistence uses historical power rather than target-time NWP. Strict operational 24-h evaluation requires eligible forecast vintages at every origin, or a clearly defined restricted set of issue times, followed by matched reevaluation of the compared models.

For the 12-h horizon, the earliest origin for a Day 3 target is 12:00 on Day 2, after batch availability. The 15-min, 1-h and 4-h horizons also avoid this specific exception. This is a deduction from the declared schedule, not independent verification of every archived input. Wind Farm B's reported extension evaluates 15 min, 1 h and 4 h; the rolling 24-h limitation above refers to the Wind Farm A benchmark.

This 2026-10-31 documentation clarification replaces the earlier description of a 00:00 UTC cycle and 8–31 h leads with the current manuscript's declared schedule. It does not re-audit the private archive, establish row-level provenance, alter verification flags, or report new training or evaluation results. The archived source, configurations, predictions and scores retain their existing identities.

### Forecast-vintage archive

`nwp_timing_rule: forecast_vintages` uses an archive containing actual reference times, availability times, lead hours, feature names and forecast values. For each origin and target, the loader selects an eligible run whose recorded availability is no later than the origin, then interpolates the required lead within that run. This is the path used for origin-aware archives such as the Kelmarsh GFS inputs. It cannot repair missing availability evidence by relabelling a target-time series.

Real experiments require real provenance. Do not invent issue/availability times or turn on a verification flag merely to pass an input check. Keep the recorded weather mode with the predictions and configuration when comparing experiments.

### Synthetic demonstration

`scripts/run_public_demo.py` constructs fictitious SCADA and forecast-vintage arrays and labels them as synthetic. The toy forecast timestamps make the vintage interface testable; they are not claims about ECMWF or GFS availability. The demo runs on CPU with 10 turbines, 800 time steps, width 32, one encoder layer, batch size 4, stage budgets 1/1/2 and at most two batches per epoch. Those reductions are confined to the demonstration. Its validation output checks that the software path works, not that the model reproduces paper accuracy.

## 5. Outputs and interpretation

Production runs save stage-1, stage-2, best and last checkpoints; effective settings and metrics; preprocessing statistics; indexed predictions; loss history; and timing information. Inspect `status` and `evaluation_split` rather than inferring completion from a directory name. Prediction time/turbine indices and target-validity masks are required for aligned comparisons. NRMSE and NMAE are stored as fractions of capacity; multiply by 100 to express them as percentages. A difference in percentage points is not the same quantity as a relative percentage improvement.

Complete data, preprocessing, configuration, checkpoint and environment identities are needed to reproduce a result. Hardware and library changes may alter floating-point outputs even with matching seeds. Apply the reproduction criterion specified for the relevant experiment; the public release does not redefine those tolerances. A completed synthetic smoke test, a model checkpoint alone or a partial seed set is not a reproduction of a paper table.

For an actual comparison, retain all prescribed seeds and report the prescribed evaluation split. Baseline training targets and evaluation targets must be checked explicitly; input sharing alone does not establish fairness. Fault-performance changes should use the same task's clean rerun as their reference under the relevant fault protocol. Training corruption, normal accuracy and controlled-fault evaluation answer different questions and should remain separately identifiable.
