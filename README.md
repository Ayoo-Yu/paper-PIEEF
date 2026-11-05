# PI-EEF: R1 submission release

Code, anonymised operational data and reported results for *A physics-guided
framework for robust wind power forecasting under non-ideal observations*,
Energy manuscript **EGY-D-26-13020 R1**, submitted 2026-11-03.

The release includes Wind Farms A and B, Kelmarsh SCADA, the GFS forecast
extracts used by the model, the complete PI-EEF and experiment code, frozen
main-model weights and predictions, and the reported tables and figures.
Wind Farm A terrain input is withheld to keep the site anonymous. The CFD
lookup tables, intersector, wake, roughness and quality-check results and
complete CFD code are included; rebuilding A terrain-flow simulations requires
the withheld terrain input.

Four asset archives accompany the repository: `pieef_wind_farm_A.zip`,
`pieef_kelmarsh.zip`, `pieef_wind_farm_B.zip` and `kelmarsh_auxiliary_scada.zip`.
Exact bytes and SHA-256 are in [release-assets.json](release-assets.json).
Each archive includes `SHA256SUMS.txt` for all other members. The main-model
archives contain 25 A runs and 9 runs for each extension site, including
checkpoints, preprocessing, indexed test predictions, metrics and protocols.
Baseline code, configurations and scores are supplied; baseline weights are
not included. The archives are attached to
[GitHub Release `r1-submission-20261103`](https://github.com/Ayoo-Yu/paper-PIEEF/releases/tag/r1-submission-20261103),
and [release-assets.json](release-assets.json) gives each archive's download
URL, size in bytes and SHA-256.

Use an existing compatible Python environment. Local checks use Python 3.10.19,
NumPy 2.2.6, pandas 2.3.3, PyYAML 6.0.3 and PyTorch 2.5.1+cu121 on CPU.
These versions describe local verification, not a complete lock of historical
training environments. Historical model metadata records PyTorch 2.5.1.
Core dependencies are listed in `requirements.txt`; experiment-specific
baselines and plotting may require their optional dependencies.

Run the CPU checks from the repository root:

```sh
python scripts/check_env.py
python -m unittest discover -s tests -v
python scripts/verify_release.py
python scripts/run_public_demo.py
python scripts/reproduce.py --all --from-results
```

The public demo performs synthetic inference only. Results regeneration reads
frozen sources and performs no training. Individual paper items and output
paths are listed in `configs/paper_items.json` and the [item command table](docs/REPRODUCING.md); use `--item ITEM --from-results`
for one item. The numerical tables and plot source data are in `results/`;
figures and editable schematic sources are in `figures/`. Results export
preserves the reported values. The Table 7 summary is supplied; raw per-run
timing logs were not retained. No new timing measurements replace those logs.

`src/pieef/` contains the main model, baselines, preprocessing, fault protocols,
statistical helpers, CFD integration and the retained legacy comparison.
`scripts/` contains training and analysis entry points. Configurations identify
each experiment family, horizon and seed. Training is separate from the
commands above and is not performed during release verification.

[data/SCHEMA.md](data/SCHEMA.md) documents columns, units and turbine ordering.
SCADA power is in kW; weather wind, pressure and temperature use m/s, Pa and K.
Wind Farm A/B layouts use metres relative to their coordinate and elevation minima; Kelmarsh retains public coordinates.
Wind Farm A uses T01–T29, B uses T01–T72, and Kelmarsh uses 1–6. Timestamps,
measured targets, prediction values and frozen training statistics are retained.
The chronological split is 70%/10%/20%; historical filling is causal and
normalisation uses the training period. Candidate-event labels describe
observations, rather than confirmed operational causes. See
[METHOD_AND_PROTOCOL.md](docs/METHOD_AND_PROTOCOL.md) for the data contract.

Kelmarsh contains daily GFS runs initialised at 00 UTC: **367 runs,
2023-12-31 to 2024-12-31, leads 3–33 h**. Availability times are taken from the
original S3 LastModified; at each origin the latest run available at the origin
is used. The point extracts and final forecast-vintage cube are included.
The complete global GFS GRIB archive is not included.

Wind Farms A and B contain an author-prepared ECMWF archive aligned to target
valid time. The files do not record issue or availability times. Refer to
manuscript Section 3.1 and Supplementary Section S3.4 for the operational
schedule. The supplied fields are the forecast archive used in the study,
provided for reproducing the reported results.

Figure 6 is regenerated using relative elevation; its colour scale differs
from the paper's by a constant offset. Geographic coordinates and absolute
elevation are removed from its source data. No A DEM-derived grid, mesh,
STL, surface or geometry is distributed. See
[ANONYMISATION.md](docs/ANONYMISATION.md) and
[PROVENANCE.md](docs/PROVENANCE.md). The public manifests identify the transformed
release bytes; internal source inventories and original site mappings are
not distributed.

PI-EEF code is MIT licensed ([LICENSE](LICENSE)). Wind Farm A/B operational
data and all author-produced published results in `data/`, `results/`,
`figures/` and model asset archives are CC BY 4.0 ([LICENSE-DATA](LICENSE-DATA)).
Third-party material retains its source terms. Kelmarsh SCADA is CC BY 4.0:
Plumley, C. and Takeuchi, R., *Kelmarsh wind farm data*, version 4, Zenodo (2025),
[DOI 10.5281/zenodo.16807551](https://doi.org/10.5281/zenodo.16807551), manuscript
[33]. GFS is NOAA public data, cited in manuscript [34]:
[NOAA NCEI Global Forecast System](https://www.ncei.noaa.gov/products/weather-climate-models/global-forecast).
ECMWF archive and CFD component notices are in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). roughFoam retains GPLv3-or-later
and cJSON retains MIT, with their own licence notices in the CFD source tree.

Please cite the manuscript and release using [CITATION.cff](CITATION.cff).
Contact: liuzhenqing@hust.edu.cn.
