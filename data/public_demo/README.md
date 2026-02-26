# Public Demo Data

This folder stores synthetic demonstration data for public reproducibility.

Generate files with:

```bash
python scripts/generate_public_demo_data.py \
  --output-dir data/public_demo \
  --turbines 3 \
  --days 60 \
  --seed 2026
```

Expected outputs:

- `DMS.csv`
- `turbines.csv`
