"""Read regular latitude/longitude GRIB1 fields with simple grid-point packing.

This deliberately small decoder implements the ECMWF/WMO section definitions:
https://codes.ecmwf.int/grib/format/grib1/sections/
It rejects unsupported packing and grids instead of silently returning values.
Parameter identity retains both the table version and parameter number.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

import numpy as np


def _signed_magnitude(data: bytes) -> int:
    value = int.from_bytes(data, "big")
    sign = 1 << (8 * len(data) - 1)
    return -(value & (sign - 1)) if value & sign else value


def _ibm_float(data: bytes) -> float:
    if len(data) != 4:
        raise ValueError("GRIB1 reference value requires four bytes")
    mantissa = int.from_bytes(data[1:], "big")
    sign = -1.0 if data[0] & 128 else 1.0
    return sign * mantissa / 2**24 * 16.0 ** ((data[0] & 127) - 64)


@dataclass(frozen=True)
class RegularLatLonGrid:
    ni: int
    nj: int
    latitude_first: float
    longitude_first: float
    latitude_last: float
    longitude_last: float
    i_increment: float
    j_increment: float
    scanning_mode: int

    @property
    def size(self) -> int:
        return self.ni * self.nj

    def coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        """Return latitude and longitude in the field's stored point order."""
        if self.scanning_mode & 0x1F:
            raise NotImplementedError("Reserved/alternate scanning mode is unsupported")
        di = -self.i_increment if self.scanning_mode & 0x80 else self.i_increment
        dj = self.j_increment if self.scanning_mode & 0x40 else -self.j_increment
        lons = self.longitude_first + np.arange(self.ni) * di
        lats = self.latitude_first + np.arange(self.nj) * dj
        if self.scanning_mode & 0x20:
            return np.tile(lats, self.ni), np.repeat(lons, self.nj)
        return np.repeat(lats, self.ni), np.tile(lons, self.nj)


@dataclass(frozen=True)
class Grib1Message:
    table_version: int
    parameter: int
    centre: int
    generating_process: int
    level_type: int
    level: int
    reference_time: datetime
    step_unit: int
    start_step: int
    end_step: int
    time_range_indicator: int
    grid: RegularLatLonGrid
    values: np.ndarray
    mars_class: int | None = None
    mars_type: int | None = None
    mars_stream: int | None = None
    offset: int = 0

    @property
    def mars_parameter(self) -> str:
        return f"{self.parameter}.{self.table_version}"

    @property
    def lead_hours(self) -> float:
        seconds = {0: 60, 1: 3600, 2: 86400, 10: 10800, 11: 21600, 12: 43200, 13: 1}
        if self.step_unit not in seconds:
            raise NotImplementedError(f"Time unit {self.step_unit} is unsupported")
        return self.end_step * seconds[self.step_unit] / 3600.0

    @property
    def valid_time(self) -> datetime:
        return self.reference_time + timedelta(hours=self.lead_hours)


def _section(message: bytes, position: int, minimum: int) -> tuple[bytes, int]:
    if position + 3 > len(message) - 4:
        raise ValueError("Truncated GRIB1 section header")
    length = int.from_bytes(message[position:position + 3], "big")
    if length < minimum or position + length > len(message) - 4:
        raise ValueError("Invalid or truncated GRIB1 section length")
    return message[position:position + length], position + length


def decode_grib1(message: bytes, *, offset: int = 0) -> Grib1Message:
    """Decode one complete GRIB1 message; times are timezone-aware UTC."""
    if len(message) < 12 or message[:4] != b"GRIB" or message[-4:] != b"7777":
        raise ValueError("Missing GRIB1 start/end marker")
    if message[7] != 1:
        raise NotImplementedError("Only GRIB edition 1 is supported")
    if int.from_bytes(message[4:7], "big") != len(message):
        raise ValueError("GRIB1 message length does not match the indicator")
    pds, position = _section(message, 8, 28)
    if not pds[7] & 0x80:
        raise NotImplementedError("A grid description section is required")
    gds, position = _section(message, position, 32)
    if gds[5] != 0 or gds[3] != 0 or gds[4] != 255:
        raise NotImplementedError("Only regular latitude/longitude grids without vertical lists are supported")
    ni, nj = int.from_bytes(gds[6:8], "big"), int.from_bytes(gds[8:10], "big")
    if ni == 0 or nj == 0 or ni == 65535 or nj == 65535:
        raise NotImplementedError("Reduced or empty grids are unsupported")
    di, dj = int.from_bytes(gds[23:25], "big"), int.from_bytes(gds[25:27], "big")
    if di == 65535 or dj == 65535:
        raise NotImplementedError("Grid increments must be explicitly defined")
    grid = RegularLatLonGrid(
        ni=ni, nj=nj, latitude_first=_signed_magnitude(gds[10:13]) / 1000.0,
        longitude_first=_signed_magnitude(gds[13:16]) / 1000.0,
        latitude_last=_signed_magnitude(gds[17:20]) / 1000.0,
        longitude_last=_signed_magnitude(gds[20:23]) / 1000.0,
        i_increment=di / 1000.0, j_increment=dj / 1000.0,
        scanning_mode=gds[27],
    )
    grid.coordinates()  # Validate the scanning mode before exposing values.
    bitmap = np.ones(grid.size, dtype=bool)
    if pds[7] & 0x40:
        bms, position = _section(message, position, 6)
        if int.from_bytes(bms[4:6], "big") != 0:
            raise NotImplementedError("Predefined GRIB1 bitmaps are unsupported")
        bits = np.unpackbits(np.frombuffer(bms[6:], dtype=np.uint8))
        if len(bits) - bms[3] != grid.size:
            raise ValueError("Bitmap size does not match the grid")
        bitmap = bits[:grid.size].astype(bool)
    bds, position = _section(message, position, 11)
    if position != len(message) - 4:
        raise ValueError("Unexpected bytes after binary data section")
    if bds[3] & 0xD0:
        raise NotImplementedError("Only simple grid-point packing without additional flags is supported")
    width, unused = bds[10], bds[3] & 0x0F
    if width > 32:
        raise NotImplementedError("Packed values wider than 32 bits are unsupported")
    count = int(bitmap.sum())
    reference = _ibm_float(bds[6:10])
    binary_scale, decimal_scale = _signed_magnitude(bds[4:6]), _signed_magnitude(pds[26:28])
    if width == 0:
        packed = np.zeros(count, dtype=np.float64)
    else:
        bits = np.unpackbits(np.frombuffer(bds[11:], dtype=np.uint8))
        if len(bits) - unused != count * width:
            raise ValueError("Packed data size does not match the grid/bitmap")
        bits = bits[:count * width].reshape(count, width).astype(np.uint64)
        weights = np.left_shift(np.uint64(1), np.arange(width - 1, -1, -1, dtype=np.uint64))
        packed = (bits @ weights).astype(np.float64)
    decoded = (reference + packed * 2.0**binary_scale) * 10.0**(-decimal_scale)
    values = np.full(grid.size, np.nan, dtype=np.float64)
    values[bitmap] = decoded
    tri, p1, p2 = pds[20], pds[18], pds[19]
    if tri in (0, 1):
        start_step, end_step = p1, p1
    elif tri in (2, 3, 4, 5):
        start_step, end_step = p1, p2
    elif tri == 10:
        start_step = end_step = int.from_bytes(pds[18:20], "big")
    else:
        raise NotImplementedError(f"Time range indicator {tri} is unsupported")
    year = (pds[24] - 1) * 100 + pds[12]
    reference_time = datetime(year, pds[13], pds[14], pds[15], pds[16], tzinfo=timezone.utc)
    local_one = len(pds) >= 52 and pds[40] == 1
    result = Grib1Message(
        table_version=pds[3], parameter=pds[8], centre=pds[4], generating_process=pds[5],
        level_type=pds[9], level=int.from_bytes(pds[10:12], "big"),
        reference_time=reference_time, step_unit=pds[17], start_step=start_step, end_step=end_step,
        time_range_indicator=tri, grid=grid, values=values,
        mars_class=pds[41] if local_one else None,
        mars_type=pds[42] if local_one else None,
        mars_stream=int.from_bytes(pds[43:45], "big") if local_one else None,
        offset=offset,
    )
    result.valid_time  # Reject unsupported time units now, rather than later.
    return result


def iter_grib1(path: Path | str) -> Iterator[Grib1Message]:
    """Stream a file of consecutive GRIB1 messages, retaining byte offsets."""
    with Path(path).open("rb") as stream:
        while True:
            offset = stream.tell()
            indicator = stream.read(8)
            if not indicator:
                return
            if len(indicator) != 8 or indicator[:4] != b"GRIB":
                raise ValueError(f"Invalid GRIB1 indicator at byte {offset}")
            if indicator[7] != 1:
                raise NotImplementedError("Only GRIB edition 1 is supported")
            length = int.from_bytes(indicator[4:7], "big")
            if length < 12:
                raise ValueError(f"Invalid GRIB1 message length at byte {offset}")
            payload = stream.read(length - 8)
            if len(payload) != length - 8:
                raise ValueError(f"Truncated GRIB1 message at byte {offset}")
            yield decode_grib1(indicator + payload, offset=offset)
