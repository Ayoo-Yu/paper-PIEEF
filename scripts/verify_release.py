#!/usr/bin/env python3
"""Verify published scientific files/configurations against their release manifests."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def check(row, hash_key):
    relative = Path(row['path'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Manifest path must stay within this repository')
    path = ROOT / relative
    content = path.read_bytes()
    if 'bytes' in row and len(content) != row['bytes']:
        raise ValueError(f'Size mismatch: {relative}')
    if hashlib.sha256(content).hexdigest() != row[hash_key]:
        raise ValueError(f'SHA-256 mismatch: {relative}')


def main():
    source = json.loads((ROOT / 'provenance/production_source_manifest.json').read_text(encoding='utf-8'))
    configs = json.loads((ROOT / 'provenance/config_manifest.json').read_text(encoding='utf-8'))
    for row in source['scientific_core_files']:
        check(row, 'sha256')
    for row in configs['records']:
        check(row, 'published_sha256')
    if len(configs['records']) != 25 or not all(r['scientific_fields_unchanged'] for r in configs['records']):
        raise ValueError('Expected the 25 matched main-table configuration templates')
    print(json.dumps({'verified': True, 'scientific_files': len(source['scientific_core_files']),
        'matched_configuration_files': len(configs['records']),
        'implementation_id': source['implementation_id']}, indent=2))


if __name__ == '__main__':
    main()
