#!/usr/bin/env python3
"""Download the declared release assets and validate their SHA-256 digests."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
ROOT=Path(__file__).resolve().parents[1]
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', required=True, help='URL containing the four reviewed asset ZIPs')
    parser.add_argument('--output', type=Path, default=ROOT/'assets')
    args=parser.parse_args()
    manifest=json.loads((ROOT/'release-assets.json').read_text(encoding='utf-8'))
    records=manifest.get('assets',manifest.get('files',[])) if isinstance(manifest,dict) else manifest
    args.output.mkdir(parents=True,exist_ok=True)
    for row in records:
        name=row.get('name',row.get('filename',row.get('path')))
        if Path(name).name != name:raise ValueError('Asset name must be a filename')
        target=args.output/name
        urllib.request.urlretrieve(args.base_url.rstrip('/')+'/'+name,target)
        digest=hashlib.sha256(target.read_bytes()).hexdigest()
        if target.stat().st_size != row['bytes'] or digest != row['sha256']:
            raise ValueError(f'Asset checksum mismatch: {name}')
        print(name)
if __name__=='__main__':main()
