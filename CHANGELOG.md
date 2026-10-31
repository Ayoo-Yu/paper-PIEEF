# Changelog

## 2026-10-31 — Clarify NWP availability and 24-h evaluation scope

- Align the README and weather timing contract with the current manuscript's declared 18:00 UTC ECMWF cycle, next-day 08:00 Beijing Time availability and following-day target inputs.
- Identify the rolling 24-h origins from 00:00 to 07:45 as preceding their assigned batch's availability, making the full-day Wind Farm A 24-h benchmark a partly non-causal offline comparison.
- Distinguish this exception from the 15-min, 1-h, 4-h and 12-h horizons under the declared schedule, while retaining the lack of independently verified row-level provenance for prepared inputs.
- Update documentation only. Scientific code, configuration flags, archived predictions and reported scores are unchanged; no training or evaluation was rerun.

## 2026-10-28 — Revised VF implementation and public demonstration

This release synchronises the public implementation with the revised VF method. The earlier public snapshot represented an earlier model and should not be used as an implementation reference for the revised paper.

### Method and training

- Expose **VF = V1 + F2 + F3 + F6** through explicit policies. V1 retains F1 training-statistic RevIN fallback and the F4 target-NWP/physical-power head.
- Include Stage-3 structured history corruption (`f2_block`), normalised Q/K attention with learned temperature and additive operating-point weighting (`cosine_temperature`), and the observation-anchored forecast skip (`last_obs_linear`).
- Include direction and terrain biases, density-aware physical branches, constrained turbine-specific power curves and explicit removal indicators.
- Provide joint Stage-3 fine-tuning with a frozen reconstruction decoder and decaying reconstruction/calibration losses. Legacy sequential and scoring policies remain available for explicit comparisons.

### Reproducibility and data

- Provide the revised production modules and main-benchmark templates. Operational tracking references are sanitised; existing relative data paths and scientific settings are preserved. Private data are not bundled. Original configuration identity is recorded in release provenance.
- Add `scripts/run_public_demo.py`, a separate CPU launcher for a small synthetic three-stage demonstration using the revision data path and VF modules. Production `scripts/train.py` retains its CUDA requirement for revision training.
- Document raw measured targets separately from causally filled historical inputs.
- Document the different `author_prepared_ecmwf` and `forecast_vintages` semantics, including the retrospective scope of the prepared Wind Farm A long-horizon benchmark. Author confirmation is not independent verification of availability at the origin.
- Replace machine-specific instructions with repository-relative commands and a method/protocol reference.

### Scope

This is a code and documentation release, not a new experimental result. It does not alter archived paper scores, retrain completed models, distribute confidential farm data or present synthetic demo metrics as real-data evidence. Baseline reproduction tolerances and statistical procedures remain experiment-specific rather than being changed by this release.
