#!/usr/bin/env python3
"""Report only the dependencies used by this curated public release."""
from importlib import import_module
import json
import platform
import sys

def main():
    report = {'python': sys.version, 'platform': platform.platform(), 'dependencies': {}}
    missing = []
    for name in ('numpy', 'pandas', 'yaml', 'torch'):
        try:
            module = import_module(name)
            report['dependencies'][name] = str(module.__version__)
        except ImportError:
            report['dependencies'][name] = 'MISSING'
            missing.append(name)
    if 'torch' not in missing:
        import torch
        report['cuda_available'] = torch.cuda.is_available()
    print(json.dumps(report, indent=2))
    if missing:
        raise SystemExit(1)

if __name__ == '__main__':
    main()
