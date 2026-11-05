#!/usr/bin/env python3
"""Check public tree/member names and text for reserved private identifiers.
The full internal release audit additionally inspects numerical and binary data.
"""
from pathlib import Path
import json
import re
import zipfile
ROOT=Path(__file__).resolve().parents[1]
# Encoded to avoid introducing prohibited original names into the release.
WORDS=tuple(bytes.fromhex(value).decode('utf-8') for value in (
'44616d75','444d53','646d73','5a68757975616e7869','5a5958','7a7978',
'e5a4a7e5b995e5b1b1','e7abb9e59bade8a5bf','32342e305f3130332e33','4e323945313134'))
PATTERN=re.compile('|'.join(re.escape(word) for word in WORDS))
def main():
    hits=[]
    for path in ROOT.rglob('*'):
        if not path.is_file() or '.git' in path.relative_to(ROOT).parts:
            continue
        relative=path.relative_to(ROOT).as_posix()
        if PATTERN.search(relative): hits.append(relative)
        raw=path.read_bytes()
        if PATTERN.search(raw.decode('utf-8',errors='ignore')):hits.append(relative)
    print(json.dumps({'passed':not hits,'hits':hits},indent=2))
    if hits:raise SystemExit(1)
if __name__=='__main__':main()
