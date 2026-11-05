"""Fetch only required GRIB2 fields from NOAA's operational GFS archive.

Network access is explicitly direct. Cached global fields can be reused across
sites; only the requested fields are downloaded. Point values use bilinear
interpolation on the real regular latitude/longitude grid. Forecast availability
is the latest original S3 object Last-Modified among all selected leads.
"""
from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed, CancelledError
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
import json
from pathlib import Path
from threading import Lock, Event, local
import time
import urllib.request

import numpy as np

BUCKET = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"


@dataclass(frozen=True)
class FieldSpec:
    name: str
    variable: str
    index_level: str
    level_type: str
    level: float
    units: str


FIELD_SPECS = (
    FieldSpec("u10", "UGRD", "10 m above ground", "heightAboveGround", 10, "m s**-1"),
    FieldSpec("v10", "VGRD", "10 m above ground", "heightAboveGround", 10, "m s**-1"),
    FieldSpec("surface_pressure", "PRES", "surface", "surface", 0, "Pa"),
    FieldSpec("t2m", "TMP", "2 m above ground", "heightAboveGround", 2, "K"),
    FieldSpec("u100", "UGRD", "100 m above ground", "heightAboveGround", 100, "m s**-1"),
    FieldSpec("v100", "VGRD", "100 m above ground", "heightAboveGround", 100, "m s**-1"),
    FieldSpec("dewpoint_2m", "DPT", "2 m above ground", "heightAboveGround", 2, "K"),
)
FEATURE_NAMES = ("u10_ms", "v10_ms", "surface_pressure_pa", "temperature_2m_k",
                 "u100_ms", "v100_ms", "dewpoint_2m_k", "wind_speed100_ms", "wind_speed10_ms")
FEATURE_UNITS = tuple(spec.units for spec in FIELD_SPECS) + ("m s**-1", "m s**-1")
_DECODE_LOCK = Lock()
_SESSION_LOCAL = local()


@dataclass(frozen=True)
class IndexEntry:
    spec: FieldSpec
    start: int
    end: int


@dataclass(frozen=True)
class DecodedPoint:
    value: float
    reference_time: datetime
    valid_time: datetime
    lead_hours: float
    short_name: str
    units: str
    grid_points: tuple[tuple[float, float, float], ...]


@dataclass(frozen=True)
class StationRequest:
    name: str
    latitude: float
    longitude: float
    references: tuple[datetime, ...]
    output_path: Path


def direct_opener():
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    # Optional, already installed in the acquisition runtime. Reusing direct
    # connections avoids tens of thousands of separate TLS handshakes. Both
    # the urllib fallback and requests path explicitly disable all proxies.
    try:
        import requests
    except ImportError:
        return opener
    if not hasattr(_SESSION_LOCAL, "session"):
        session = requests.Session()
        session.trust_env = False
        session.proxies = {}
        _SESSION_LOCAL.session = session
    opener.gfs_session = _SESSION_LOCAL.session
    return opener


class _PooledResponse:
    def __init__(self, response):
        self.response = response
        self.status, self.headers = response.status_code, response.headers

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.response.close()

    def read(self, size=None):
        try:
            return self.response.raw.read(size, decode_content=False)
        except Exception as error:
            raise OSError(f"Direct forecast response failed while reading: {error}") from error


def _open_direct(opener, request, timeout):
    session = getattr(opener, "gfs_session", None)
    if session is None:
        return opener.open(request, timeout=timeout)
    import requests
    url = request.full_url if isinstance(request, urllib.request.Request) else request
    headers = dict(request.header_items()) if isinstance(request, urllib.request.Request) else {"Accept-Encoding": "identity"}
    try:
        response = session.get(url, headers=headers, proxies={}, timeout=(15, timeout), stream=True)
    except requests.RequestException as error:
        raise urllib.error.URLError(str(error)) from error
    if response.status_code not in {200, 206}:
        status, reason, response_headers = response.status_code, response.reason, response.headers
        response.close()
        raise urllib.error.HTTPError(url, status, reason, response_headers, None)
    return _PooledResponse(response)


def gfs_url(reference: datetime, lead: int) -> str:
    if reference.tzinfo is None or reference.utcoffset() != timedelta(0):
        raise ValueError("GFS initialization must be explicit UTC")
    if reference.hour not in {0, 6, 12, 18} or reference.minute or reference.second:
        raise ValueError("GFS runs initialize at 00, 06, 12, and 18 UTC")
    if not isinstance(lead, (int, np.integer)) or isinstance(lead, bool) or lead < 1 or lead > 120:
        raise ValueError("Use positive integer forecast leads through 120 h")
    return (f"{BUCKET}/gfs.{reference:%Y%m%d}/{reference:%H}/atmos/"
            f"gfs.t{reference:%H}z.pgrb2.0p25.f{lead:03d}")


def parse_gfs_index(text: str, reference: datetime, lead: int, specs=FIELD_SPECS) -> list[IndexEntry]:
    entries = []
    rows = [line.split(":") for line in text.splitlines() if line.strip()]
    offsets = [int(row[1]) for row in rows]
    if any(second < first for first, second in zip(offsets, offsets[1:])):
        raise ValueError("GFS index offsets are not sorted")
    for spec in specs:
        candidates = [(index, row) for index, row in enumerate(rows)
                      if len(row) >= 6 and row[3] == spec.variable and row[4] == spec.index_level]
        if len(candidates) != 1:
            raise ValueError(f"Expected one {spec.name} field, found {len(candidates)}")
        index, row = candidates[0]
        if row[2] != f"d={reference:%Y%m%d%H}" or row[5] != f"{lead} hour fcst":
            raise ValueError(f"{spec.name}: index is not the requested instantaneous forecast vintage")
        start = offsets[index]
        following = next((offset for offset in offsets[index + 1:] if offset > start), None)
        if following is None:
            raise ValueError("Selected final field has no following index offset")
        entries.append(IndexEntry(spec, start, following - 1))
    return entries


def split_grib2_messages(blob: bytes) -> list[bytes]:
    messages, offset = [], 0
    while offset < len(blob):
        if len(blob) - offset < 16 or blob[offset:offset + 4] != b"GRIB" or blob[offset + 7] != 2:
            raise ValueError("Expected a complete GRIB2 message")
        length = int.from_bytes(blob[offset + 8:offset + 16], "big")
        if length < 20 or offset + length > len(blob):
            raise ValueError("GRIB2 message length is invalid/truncated")
        message = blob[offset:offset + length]
        if message[-4:] != b"7777":
            raise ValueError("GRIB2 end marker is missing")
        messages.append(message)
        offset += length
    return messages


def _request(opener, request, *, timeout: float = 60, retries: int = 3):
    for attempt in range(retries):
        try:
            return opener.open(request, timeout=timeout)
        except (OSError, urllib.error.URLError):
            if attempt + 1 == retries:
                raise
            time.sleep(min(4, 2**attempt))


def _read_request(opener, request, *, expected_bytes=None, timeout=45, retries=4):
    """Retry both connection and body reading; never accept a partial range."""
    for attempt in range(retries):
        try:
            with _open_direct(opener, request, timeout) as response:
                if expected_bytes is not None:
                    if response.status != 206:
                        raise RuntimeError("Server did not honor Range; refusing a whole global GRIB download")
                    requested = request.headers["Range"].replace("=", " ")
                    if response.headers.get("Content-Range", "").split("/")[0] != requested:
                        raise RuntimeError("Server returned a different GRIB byte range")
                    data = response.read(expected_bytes + 1)
                    if len(data) != expected_bytes:
                        raise OSError("Selected forecast byte range was truncated or oversized")
                else:
                    data = response.read()
                return data, {key.title(): value for key, value in response.headers.items()}
        except (OSError, urllib.error.URLError) as error:
            if isinstance(error, urllib.error.HTTPError) and error.code in {400, 401, 403, 404}:
                raise
            if attempt + 1 == retries:
                raise
            print(f"[retry] {type(error).__name__}: {error}; attempt={attempt + 1}/{retries}", flush=True)
            time.sleep(min(4, 2**attempt))


def fetch_run_lead(reference: datetime, lead: int, cache_directory: Path | None, opener=None,
                   *, specs=FIELD_SPECS, retain_global=True):
    """Return only requested original messages and authentic S3 availability.

    ``retain_global=False`` keeps newly fetched global messages only in memory.
    Existing full-field sample caches may still satisfy a subset request; they
    are never overwritten or deleted by this mode.
    """
    specs = tuple(specs)
    if not specs or len({spec.name for spec in specs}) != len(specs):
        raise ValueError("Requested GFS fields must be non-empty and unique")
    opener = opener or direct_opener()
    url = gfs_url(reference, lead)
    directory = Path(cache_directory) / reference.strftime("%Y%m%d%H") if cache_directory is not None else None
    suffix = "selected" if specs == FIELD_SPECS else "_".join(spec.name for spec in specs)
    path = directory / f"f{lead:03d}.{suffix}.grib2" if directory is not None else None
    metadata_path = directory / f"f{lead:03d}.{suffix.replace('selected', 'source')}.json" if directory is not None else None
    candidates = [(path, metadata_path)] if path is not None else []
    if directory is not None and specs != FIELD_SPECS:
        candidates.append((directory / f"f{lead:03d}.selected.grib2", directory / f"f{lead:03d}.source.json"))
    for cached_path, cached_metadata in candidates:
        if not cached_path.exists() or not cached_metadata.exists():
            continue
        metadata = json.loads(cached_metadata.read_text(encoding="utf-8"))
        messages = split_grib2_messages(cached_path.read_bytes())
        names = metadata["field_names"]
        if metadata["source_url"] != url or len(messages) != len(names) or not all(spec.name in names for spec in specs):
            raise ValueError("Cached GFS fields do not match the requested schema/vintage")
        subset_metadata = dict(metadata)
        subset_metadata["field_names"] = [spec.name for spec in specs]
        subset_metadata["field_bytes"] = {spec.name: metadata["field_bytes"][spec.name] for spec in specs}
        subset_metadata["byte_ranges"] = {spec.name: metadata["byte_ranges"][spec.name] for spec in specs}
        subset_metadata["downloaded_bytes"] = 0
        return [messages[names.index(spec.name)] for spec in specs], datetime.fromisoformat(metadata["source_last_modified"]), subset_metadata

    index_bytes, _ = _read_request(opener, url + ".idx")
    index = index_bytes.decode("utf-8")
    entries = parse_gfs_index(index, reference, lead, specs)
    # Join only exactly adjacent requested fields, e.g. U and V, without extra fields.
    groups = []
    for entry in sorted(entries, key=lambda entry: entry.start):
        if groups and entry.start == groups[-1][-1].end + 1:
            groups[-1].append(entry)
        else:
            groups.append([entry])
    messages_by_name, timestamps = {}, []
    for group in groups:
        start, end = group[0].start, group[-1].end
        request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"})
        data, headers = _read_request(opener, request, expected_bytes=end - start + 1)
        modified = headers.get("Last-Modified")
        if modified is None:
            raise RuntimeError("Source forecast has no Last-Modified availability timestamp")
        timestamps.append(parsedate_to_datetime(modified).astimezone(timezone.utc))
        for entry in group:
            chunk = data[entry.start - start:entry.end - start + 1]
            if len(split_grib2_messages(chunk)) != 1:
                raise ValueError(f"{entry.spec.name}: index range did not contain exactly one GRIB2 message")
            messages_by_name[entry.spec.name] = chunk
    available = max(timestamps)
    if available < reference:
        raise ValueError("Source availability precedes forecast initialization")
    messages = [messages_by_name[spec.name] for spec in specs]
    metadata = {
        "source_url": url,
        "reference_time": reference.isoformat(),
        "lead_hours": lead,
        "source_last_modified": available.isoformat(),
        "field_names": [spec.name for spec in specs],
        "field_bytes": {entry.spec.name: entry.end - entry.start + 1 for entry in entries},
        "byte_ranges": {entry.spec.name: [entry.start, entry.end] for entry in entries},
        "downloaded_bytes": sum(entry.end - entry.start + 1 for entry in entries),
    }
    if retain_global:
        if directory is None:
            raise ValueError("A cache directory is required to retain global fields")
        directory.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"".join(messages))
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return messages, available, metadata


def bilinear_indices(ni: int, nj: int, lat0: float, lon0: float, di: float, dj: float,
                     latitude: float, longitude: float):
    """Four neighbours/weights for north-to-south global regular-ll storage."""
    if di <= 0 or dj <= 0 or not np.isclose(ni * di, 360):
        raise ValueError("Expected a cyclic global longitude grid with positive increments")
    y = (lat0 - latitude) / dj
    if not 0 <= y <= nj - 1:
        raise ValueError("Requested latitude lies outside the source grid")
    x = ((longitude - lon0) % 360) / di
    north = min(int(np.floor(y)), nj - 2)
    west = int(np.floor(x)) % ni
    east = (west + 1) % ni
    fx, fy = x - np.floor(x), y - north
    indices = [north * ni + west, north * ni + east, (north + 1) * ni + west, (north + 1) * ni + east]
    weights = np.array([(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy])
    coordinates = tuple((lat0 - row * dj, ((lon0 + col * di + 180) % 360) - 180, float(weight))
                        for row, col, weight in zip([north, north, north + 1, north + 1], [west, east, west, east], weights))
    return indices, weights, coordinates


def decode_point(message: bytes, latitude: float, longitude: float, spec: FieldSpec) -> DecodedPoint:
    """Decode a real GFS field and bilinearly sample its four native neighbours."""
    # Individual handles are independent, but the Windows library's build-time
    # thread-safety option is unknown. Serialize decode, while HTTP stays parallel.
    with _DECODE_LOCK:
        return _decode_point(message, latitude, longitude, spec)


def _decode_point(message: bytes, latitude: float, longitude: float, spec: FieldSpec) -> DecodedPoint:
    import eccodes

    handle = eccodes.codes_new_from_message(message)
    try:
        get = lambda key: eccodes.codes_get(handle, key)
        if get("edition") != 2 or get("centre") != "kwbc" or get("typeOfGeneratingProcess") != 2:
            raise ValueError("Field is not a NCEP GRIB2 forecast")
        identities = {"UGRD": (2, 2), "VGRD": (2, 3), "PRES": (3, 0), "TMP": (0, 0), "DPT": (0, 6)}
        if get("discipline") != 0 or (get("parameterCategory"), get("parameterNumber")) != identities[spec.variable]:
            raise ValueError(f"{spec.name}: decoded parameter differs from the selected index field")
        if get("typeOfLevel") != spec.level_type or float(get("level")) != spec.level or get("units") != spec.units:
            raise ValueError(f"{spec.name}: unexpected level or units")
        if get("stepType") != "instant" or get("gridType") != "regular_ll":
            raise ValueError("Expected instantaneous regular-ll GFS fields")
        if get("iScansNegatively") or get("jScansPositively") or get("jPointsAreConsecutive") or get("alternativeRowScanning"):
            raise NotImplementedError("This GFS grid scanning mode is unsupported")
        reference = datetime.strptime(f"{get('dataDate'):08d}{get('dataTime'):04d}", "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
        valid = datetime.strptime(f"{get('validityDate'):08d}{get('validityTime'):04d}", "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
        lead = (valid - reference).total_seconds() / 3600
        if lead <= 0:
            raise ValueError("Forecast field has no positive lead")
        indices, weights, points = bilinear_indices(
            get("Ni"), get("Nj"), get("latitudeOfFirstGridPointInDegrees"), get("longitudeOfFirstGridPointInDegrees"),
            get("iDirectionIncrementInDegrees"), get("jDirectionIncrementInDegrees"), latitude, longitude,
        )
        values = np.asarray(eccodes.codes_get_double_elements(handle, "values", indices), dtype=float)
        if get("numberOfMissing"):
            values[values == float(get("missingValue"))] = np.nan
        relevant = weights > 0
        value = float(values[relevant] @ weights[relevant]) if np.isfinite(values[relevant]).all() else float("nan")
        return DecodedPoint(value, reference, valid, lead, str(get("shortName")), str(get("units")), points)
    finally:
        eccodes.codes_release(handle)


def fetch_point_archive(references, leads, latitude: float, longitude: float,
                        cache_directory: Path, output_path: Path):
    """Write the ForecastVintageStore NPZ schema plus units and point metadata."""
    references, leads = list(references), list(leads)
    if not references or any(second <= first for first, second in zip(references, references[1:])):
        raise ValueError("Forecast references must be non-empty and strictly increasing")
    if not leads or any(second <= first for first, second in zip(leads, leads[1:])):
        raise ValueError("Forecast leads must be non-empty and strictly increasing")
    values = np.full((len(references), len(leads), len(FEATURE_NAMES)), np.nan)
    availability = []
    opener = direct_opener()
    for run, reference in enumerate(references):
        last_available = reference
        for column, lead in enumerate(leads):
            messages, available, metadata = fetch_run_lead(reference, lead, cache_directory, opener)
            fields = [decode_point(message, latitude, longitude, spec) for message, spec in zip(messages, FIELD_SPECS)]
            if any(field.reference_time != reference or field.lead_hours != lead or field.valid_time != reference + timedelta(hours=lead) for field in fields):
                raise ValueError("Decoded forecast dates/leads disagree with the requested source vintage")
            raw = np.array([field.value for field in fields])
            values[run, column] = np.r_[raw, np.hypot(raw[4], raw[5]), np.hypot(raw[0], raw[1])]
            last_available = max(last_available, available)
            print(f"[gfs] init={reference.isoformat()} lead_h={lead} "
                  f"selected_mb={sum(metadata['field_bytes'].values()) / 1e6:.2f} "
                  f"available={available.isoformat()} values={values[run, column].round(6).tolist()}", flush=True)
        availability.append(last_available.isoformat())
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path, reference_time=np.array([value.isoformat() for value in references]),
        available_time=np.array(availability), lead_hours=np.array(leads, dtype=float),
        feature_names=np.array(FEATURE_NAMES), values=values,
        feature_units=np.array(FEATURE_UNITS), latitude=float(latitude), longitude=float(longitude),
        source_model="NOAA GFS operational 0.25 degree", spatial_interpolation="bilinear four native grid neighbours",
        availability_basis="latest original S3 Last-Modified over all selected forecast fields and leads",
    )
    print(f"[saved] {output_path} shape={values.shape}", flush=True)
    return output_path


def _save_point_cache(path, *, reference, leads, values, availability, station):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".writing.npz")
    np.savez_compressed(temporary, reference_time=reference.isoformat(), lead_hours=np.array(leads, dtype=float),
                        values=values, lead_available_time=availability, feature_names=np.array(FEATURE_NAMES),
                        latitude=station.latitude, longitude=station.longitude)
    temporary.replace(path)


def _fetch_station_run(reference, stations, leads, point_cache, global_cache, stop_event):
    """One global field request serves every requested station on this date."""
    records = {}
    for station in stations:
        path = Path(point_cache) / station.name / f"{reference:%Y%m%d%H}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as cached:
                if (str(cached["reference_time"]) != reference.isoformat() or
                        tuple(cached["feature_names"].tolist()) != FEATURE_NAMES or
                        not np.isclose(float(cached["latitude"]), station.latitude, rtol=0, atol=1e-10) or
                        not np.isclose(float(cached["longitude"]), station.longitude, rtol=0, atol=1e-10)):
                    raise ValueError(f"Station point cache schema/coordinates changed: {path}")
                old_leads = cached["lead_hours"].copy()
                old_values, old_available = cached["values"].copy(), cached["lead_available_time"].copy()
            if (old_leads.ndim != 1 or not np.isfinite(old_leads).all() or
                    not np.all(np.diff(old_leads) > 0) or old_values.shape != (len(old_leads), len(FEATURE_NAMES))
                    or old_available.shape != (len(old_leads),)):
                raise ValueError("Station point cache dimensions/leads are invalid")
            cache_leads = np.union1d(old_leads, leads)
            values = np.full((len(cache_leads), len(FEATURE_NAMES)), np.nan)
            available = np.full(len(cache_leads), "", dtype="U32")
            old_rows = np.searchsorted(cache_leads, old_leads)
            values[old_rows], available[old_rows] = old_values, old_available
        else:
            cache_leads = np.array(leads, dtype=float)
            values = np.full((len(leads), len(FEATURE_NAMES)), np.nan)
            available = np.full(len(leads), "", dtype="U32")
        records[station.name] = [station, path, values, available, cache_leads]
    downloaded = 0
    opener = direct_opener()
    for lead in leads:
        if stop_event.is_set():
            raise CancelledError("Sibling forecast download failed; point caches retained")
        needed = [record for record in records.values()
                  if not np.isfinite(record[2][np.searchsorted(record[4], lead)]).all()]
        if not needed:
            continue
        messages, available, metadata = fetch_run_lead(reference, lead, global_cache, opener, retain_global=False)
        for station, path, values, availability, cache_leads in needed:
            row = int(np.searchsorted(cache_leads, lead))
            fields = [decode_point(message, station.latitude, station.longitude, spec)
                      for message, spec in zip(messages, FIELD_SPECS)]
            if any(field.reference_time != reference or field.lead_hours != lead or
                   field.valid_time != reference + timedelta(hours=lead) or not np.isfinite(field.value)
                   for field in fields):
                raise ValueError(f"Missing/mismatched operational forecast: {station.name}, {reference}, +{lead}h")
            raw = np.array([field.value for field in fields])
            values[row] = np.r_[raw, np.hypot(raw[4], raw[5]), np.hypot(raw[0], raw[1])]
            availability[row] = available.isoformat()
            _save_point_cache(path, reference=reference, leads=cache_leads, values=values,
                              availability=availability, station=station)
        downloaded += metadata.get("downloaded_bytes", sum(metadata["field_bytes"].values()))
    result = {}
    for name, (station, path, values, availability, cache_leads) in records.items():
        requested_rows = np.searchsorted(cache_leads, leads)
        values, availability = values[requested_rows], availability[requested_rows]
        if not np.isfinite(values).all() or any(not value for value in availability):
            raise ValueError(f"Incomplete station run: {path}")
        timestamps = [datetime.fromisoformat(str(value)) for value in availability]
        if any(value < reference for value in timestamps):
            raise ValueError("Source availability precedes forecast initialization")
        result[name] = (values, max(timestamps).isoformat())
    return result, downloaded


def fetch_station_archives(stations, leads, point_cache, global_cache=None, *, workers=4):
    """Concurrent daily downloads, resumable small point caches, shared ranges.

    A failure cancels pending dates and leaves per-lead point caches. Complete
    output archives are written only after every requested real field is decoded.
    No newly fetched global GRIB messages are retained on disk.
    """
    stations, leads = list(stations), list(leads)
    if not stations or len({station.name for station in stations}) != len(stations):
        raise ValueError("Station names must be non-empty and unique")
    if not leads or any(second <= first for first, second in zip(leads, leads[1:])):
        raise ValueError("Forecast leads must be non-empty and strictly increasing")
    if not 1 <= workers <= 4:
        raise ValueError("workers must be 1..4")
    runs = {}
    for station in stations:
        if not station.references or any(second <= first for first, second in zip(station.references, station.references[1:])):
            raise ValueError("Station references must be non-empty and strictly increasing")
        if not -90 <= station.latitude <= 90 or not -180 <= station.longitude <= 180:
            raise ValueError("Station coordinates are invalid")
        for reference in station.references:
            gfs_url(reference, leads[0])
            runs.setdefault(reference, []).append(station)
    results, total_bytes, stopped = {}, 0, Event()
    started = time.monotonic()
    print(f"[start] stations={[station.name for station in stations]} daily_runs={len(runs)} "
          f"leads={leads} workers={workers} direct/no proxy; globals discarded", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_fetch_station_run, reference, requested, leads, point_cache, global_cache, stopped): reference
                   for reference, requested in sorted(runs.items())}
        try:
            for count, future in enumerate(as_completed(futures), 1):
                reference = futures[future]
                record, downloaded = future.result()
                results[reference] = record
                total_bytes += downloaded
                available = {name: value[1] for name, value in record.items()}
                print(f"[gfs] {count}/{len(runs)} init={reference.date()} stations={list(record)} "
                      f"available={available} downloaded_MB={downloaded/1e6:.2f} "
                      f"total_GB={total_bytes/1e9:.3f} elapsed_min={(time.monotonic()-started)/60:.1f}", flush=True)
        except BaseException as error:
            stopped.set()
            for future in futures:
                future.cancel()
            print(f"[stopped] {type(error).__name__}: {error}; point caches retained, no complete archive written", flush=True)
            raise
    for station in stations:
        station.output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = station.output_path.with_name(station.output_path.stem + ".writing.npz")
        values = np.stack([results[reference][station.name][0] for reference in station.references])
        available = [results[reference][station.name][1] for reference in station.references]
        np.savez_compressed(
            temporary, reference_time=np.array([value.isoformat() for value in station.references]),
            available_time=np.array(available), lead_hours=np.array(leads, dtype=float),
            feature_names=np.array(FEATURE_NAMES), values=values, feature_units=np.array(FEATURE_UNITS),
            latitude=station.latitude, longitude=station.longitude, source_model="NOAA GFS operational 0.25 degree",
            spatial_interpolation="bilinear four native grid neighbours at stated representative site point",
            temporal_interpolation="three-hour lead anchors; ForecastVintageStore linearly interpolates within selected run",
            availability_basis="latest original S3 LastModified over selected fields and leads", download_complete=True,
        )
        temporary.replace(station.output_path)
        print(f"[saved] {station.output_path} shape={values.shape} all_finite=True", flush=True)
    return [station.output_path for station in stations]
