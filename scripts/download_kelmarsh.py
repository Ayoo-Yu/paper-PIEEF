#!/usr/bin/env python3
"""Download only the selected year's public SCADA and its two static tables."""
from __future__ import annotations

import argparse
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://zenodo.org/records/16807551/files/"
FILES = (
    "Kelmarsh_WT_static.csv",
    "Kelmarsh_WT_dataSignalMapping.csv",
    "Kelmarsh_SCADA_2024_5962.zip",
)
# Explicitly bypass both environment and system proxies for every request/redirect.
DIRECT_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def download(name: str, destination: Path):
    target = destination / name
    if target.exists():
        if target.suffix == ".zip":
            with zipfile.ZipFile(target) as archive:
                print(f"Already downloaded: {name}; {len(archive.infolist())} members", flush=True)
        else:
            print(f"Already downloaded: {name}", flush=True)
        return target
    partial = target.with_suffix(target.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "PI-EEF-research-data/1.0"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(BASE + name + "?download=1", headers=headers)
    with DIRECT_OPENER.open(request, timeout=60) as response:
        content_range = response.headers.get("Content-Range", "")
        if offset and response.status == 206:
            if not content_range.startswith(f"bytes {offset}-"):
                raise ValueError("The server returned an inconsistent resume range.")
        else:
            offset = 0
        size = response.headers.get("Content-Length")
        expected = offset + int(size) if size else None
        written, last_update = offset, time.monotonic()
        print(f"Downloading {name}; total bytes={expected}", flush=True)
        with partial.open("ab" if offset else "wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                written += len(chunk)
                if time.monotonic() - last_update >= 30:
                    print(f"  {name}: {written / 1e6:.1f} MB downloaded", flush=True)
                    last_update = time.monotonic()
        if expected is not None and written != expected:
            raise ValueError(f"Incomplete download: {written} of {expected} bytes.")
    if target.suffix == ".zip":
        with zipfile.ZipFile(partial) as archive:
            if not archive.infolist():
                raise ValueError("The SCADA archive is empty.")
    partial.replace(target)
    print(f"Saved {target}", flush=True)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=ROOT / "data/kelmarsh")
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        download(name, args.destination)


if __name__ == "__main__":
    main()
