# Method and protocol

Individual configurations identify the model, horizon, seed and experiment
family. The final main model uses reconstruction, weather calibration and
forecasting, structured forecasting corruption, normalised graph attention
and an observation anchor. The legacy namespace supports the earlier controlled
comparison. CFD and graph-ablation switches describe additional experiments.

Main A has five horizons and five seeds; the extension sites have three horizons
and three seeds. Splits are chronological 70%/10%/20%. Normalisation uses only
the training period. Invalid historical inputs are filled causally. Targets
remain valid measured power and retain validity masks. Candidate events describe
observations and do not establish their operational cause. A uses 100-m target
wind speed and 10-m graph direction; Kelmarsh/B use the configured 100-m vector.
SCADA power is kW; reported normalised scores are fractions of capacity.
The fixed 39 evaluation faults and training-time corruption are distinct.
Time-series DM/HAC tests and seed-paired tests retain their respective families.

Kelmarsh contains daily GFS runs initialised at 00 UTC: 367 runs spanning
2023-12-31 to 2024-12-31, leads 3–33 h. Availability times come from the original
S3 LastModified. At each origin the latest run available at the origin is used.
Frozen point extracts and the final vintage cube are provided; the complete
global GRIB archive is not included.

Wind Farms A/B use an author-prepared ECMWF archive aligned to target valid
time. The files do not record issue or availability times. Refer to manuscript
Section 3.1 and Supplementary Section S3.4 for the operational schedule. The
loader preserves the archive values without inferring missing timestamps.

Results export consumes frozen sources without training. The public CPU demo
checks synthetic inference only. Frozen-model checks use the released data and
weights. Software changes may cause small floating-point differences; build
validation records the observed error. Table 7 summary values are supplied,
but raw per-run timing logs were not retained. No replacement timings are made.
