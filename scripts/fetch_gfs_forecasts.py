#!/usr/bin/env python3
"""Direct-download required NOAA GFS vintage fields, never whole global files.

Example: --site kelmarsh --start-date 2024-01-01 --end-date 2024-01-01
--latitude 52.401461 --longitude -0.943105 --leads 3 6
Default daily 00 UTC runs and 3-hour leads 3..33 cover the initial <=4-hour
forecast experiments, subject to actual source availability and store lookup.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pieef.gfs_forecasts import StationRequest, fetch_station_archives


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", required=True, choices=["kelmarsh", "wind_farm_B"])
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--latitude", required=True, type=float)
    parser.add_argument("--longitude", required=True, type=float)
    parser.add_argument("--cycles", nargs="+", type=int, default=[0])
    parser.add_argument("--leads", nargs="+", type=int, default=list(range(3, 34, 3)))
    parser.add_argument("--cache-directory", type=Path, default=ROOT / "data/raw/gfs_fields")
    parser.add_argument("--point-cache", type=Path, default=ROOT / "data/raw/nwp/gfs_points")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--extra-site", action="append", nargs=6,
                        metavar=("NAME", "LAT", "LON", "START", "END", "OUTPUT"),
                        help="Another point/date range to decode from the same downloaded fields")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    start = datetime.strptime(args.start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(args.end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    if end < start:
        parser.error("end-date precedes start-date")
    if not -90 <= args.latitude <= 90 or not -180 <= args.longitude <= 180:
        parser.error("Coordinates must be latitude [-90,90] and longitude [-180,180]")
    cycles = sorted(set(args.cycles))
    if any(cycle not in [0, 6, 12, 18] for cycle in cycles):
        parser.error("GFS cycles must be 00/06/12/18 UTC")
    references = [start + timedelta(days=day, hours=cycle)
                  for day in range((end - start).days + 1) for cycle in cycles]
    leads = sorted(set(args.leads))
    output = args.output or ROOT / "data/raw" / args.site / "gfs_forecast_vintages.npz"
    stations = [StationRequest(args.site, args.latitude, args.longitude, tuple(references), output)]
    for name, latitude, longitude, start_text, end_text, output_text in args.extra_site or []:
        extra_start = datetime.strptime(start_text, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        extra_end = datetime.strptime(end_text, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        if extra_end < extra_start:
            parser.error("Extra site's end-date precedes start-date")
        extra_refs = tuple(extra_start + timedelta(days=day, hours=cycle)
                           for day in range((extra_end - extra_start).days + 1) for cycle in cycles)
        stations.append(StationRequest(name, float(latitude), float(longitude), extra_refs, Path(output_text)))
    fetch_station_archives(stations, leads, args.point_cache, args.cache_directory, workers=args.workers)


if __name__ == "__main__":
    main()
