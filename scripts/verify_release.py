#!/usr/bin/env python3
"""Verify all public release files using the anonymous output manifest."""
from pathlib import Path
import hashlib
import json
ROOT = Path(__file__).resolve().parents[1]
def main():
    manifest = json.loads((ROOT/'provenance/release_manifest.json').read_text(encoding='utf-8'))
    records = manifest['files']
    total = 0
    for row in records:
        relative = Path(row['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Manifest path must stay inside the repository')
        raw = (ROOT/relative).read_bytes()
        if len(raw) != row['bytes'] or hashlib.sha256(raw).hexdigest() != row['sha256']:
            raise ValueError(f'Release file differs: {relative}')
        if len(raw) >= 50_000_000:
            raise ValueError(f'Tracked file exceeds the size gate: {relative}')
        total += len(raw)
    if total >= 300_000_000:
        raise ValueError('Tracked release tree exceeds 300 MB')
    print(json.dumps({'verified':True,'files':len(records),'bytes':total},indent=2))
if __name__ == '__main__':
    main()
