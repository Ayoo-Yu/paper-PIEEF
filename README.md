# PI-EEF Open-Source Release Package

This folder is a curated release bundle for public code sharing.

## Included

- Core PI-EEF model code (`src/pieef/`)
- Public demo data generator (`scripts/generate_public_demo_data.py`)
- Training entrypoint (`scripts/train.py`)
- Environment checker (`scripts/check_env.py`)
- Public demo configs (`configs/main_h16_public_demo.yaml`, `configs/main_public_demo_quick.yaml`)
- Synthetic demo dataset (`data/public_demo/`)

## Not Included

- Private/raw wind farm data
- Large local experiment artifacts and logs
- Baseline training scripts and benchmark artifacts

## Quick Start

1. Install dependencies

```bash
pip install -r requirements.txt
```

2. Regenerate synthetic demo data (optional)

```bash
python scripts/generate_public_demo_data.py \
  --output-dir data/public_demo \
  --turbines 3 \
  --days 60 \
  --seed 2026
```

3. Check environment

```bash
python scripts/check_env.py
```

4. Dry-run pipeline

```bash
python scripts/train.py --config configs/main_public_demo_quick.yaml --dry-run
```

5. Run quick training

```bash
python scripts/train.py --config configs/main_public_demo_quick.yaml
```
