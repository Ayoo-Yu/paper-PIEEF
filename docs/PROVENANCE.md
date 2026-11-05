# Release provenance

This release accompanies Energy manuscript EGY-D-26-13020 R1, *A physics-guided
framework for robust wind power forecasting under non-ideal observations*.
Internal inventories, original source paths and machine tracking are excluded.

provenance/release_manifest.json records output bytes/SHA-256 of repository
files, excluding itself, ignored runtime outputs and Git storage. The source
and configuration manifests identify transformed public bytes; they do not
claim that renamed files are byte-identical to private originals. Model assets
retain the frozen scientific tensors, predictions, normalisation statistics
and timestamps; string metadata and preprocessing coordinates are anonymised.

release-assets.json records each zip's bytes and SHA-256. Each zip contains
SHA256SUMS.txt for its other members. Model assets include checkpoints,
preprocessing, indexed predictions, metrics and protocol metadata. Baseline
scores and configurations are included; baseline weights are not included.
No training or new timing measurements are performed for this release.

Historical execution metadata records PyTorch 2.5.1. The local validation
environment is stated in README and is not a full historical environment lock.
Table 7 raw per-run timing logs were not retained. The complete global GFS
GRIB archive is not included, but the final point extracts and vintage cube
used by the model are supplied. A terrain input is withheld to keep the site
anonymous; see ANONYMISATION.md.
