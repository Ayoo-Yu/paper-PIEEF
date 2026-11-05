#!/usr/bin/env python3
"""Validate a user-supplied anonymous terrain grid for independent CFD work.
The terrain input used for Wind Farm A is withheld to preserve site anonymity.
Published lookup tables reproduce the reported downstream results without it.
"""
from pathlib import Path
import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Your terrain NPZ with x_m, y_m, z_m arrays')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.input, allow_pickle=False) as grid:
        values = {key: grid[key] for key in ('x_m', 'y_m', 'z_m')}
    if not all(np.isfinite(value).all() for value in values.values()):
        raise ValueError('Terrain values must be finite')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **values)
    print('Validated user-supplied terrain grid')

if __name__ == '__main__':
    main()
