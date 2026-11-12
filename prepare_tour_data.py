#!/usr/bin/env python3
"""Normalize the 2026-09-05 smartwatch/MOX source bundle without changing it.

The outputs deliberately retain exact raw values alongside interpreted columns.
High-rate BDF data is written at native cadence; no interpolation is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import plistlib
import re
import shutil
import sqlite3
import sys
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


SQLITE_MAGIC = b"SQLite format 3\x00"
TIMEZONE = "Europe/Belgrade"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ascii(value: bytes) -> str:
    return value.decode("latin-1").strip()


def decode_int24_le(raw: np.ndarray) -> np.ndarray:
    packed = np.asarray(raw, dtype=np.uint8).reshape(-1, 3)
    value = (
        packed[:, 0].astype(np.int32)
        | (packed[:, 1].astype(np.int32) << 8)
        | (packed[:, 2].astype(np.int32) << 16)
    )
    return np.where(value & 0x800000, value - (1 << 24), value).astype(np.int32)


@dataclass(frozen=True)
class Signal:
    label: str
    dimension: str
    physical_min: float
    physical_max: float
    digital_min: float
    digital_max: float
    samples_per_record: int
    byte_offset: int


class BDF:
    def __init__(self, path: Path, timezone: str):
        self.path = path
        self.timezone = ZoneInfo(timezone)
        with path.open("rb") as stream:
            fixed = stream.read(256)
            if len(fixed) != 256 or not fixed[:8].startswith(b"\xffBIOSEMI"):
                raise ValueError(f"Nem támogatott BDF: {path}")
            self.header_bytes = int(_ascii(fixed[184:192]))
            header = fixed + stream.read(self.header_bytes - 256)
        self.patient = _ascii(header[8:88])
        self.recording = _ascii(header[88:168])
        self.reserved = _ascii(header[192:236])
        naive = datetime.strptime(
            f"{_ascii(header[168:176])} {_ascii(header[176:184])}",
            "%d.%m.%y %H.%M.%S",
        )
        self.start = naive.replace(tzinfo=self.timezone)
        declared_records = int(_ascii(header[236:244]))
        self.record_duration = float(_ascii(header[244:252]))
        signal_count = int(_ascii(header[252:256]))
        offset = 256
        columns: list[list[str]] = []
        for width in (16, 80, 8, 8, 8, 8, 8, 80, 8, 32):
            columns.append(
                [_ascii(header[offset + i * width : offset + (i + 1) * width]) for i in range(signal_count)]
            )
            offset += signal_count * width
        samples = [int(value) for value in columns[8]]
        byte_offsets = np.cumsum([0] + samples[:-1]) * 3
        self.record_bytes = 3 * sum(samples)
        payload = path.stat().st_size - self.header_bytes
        if payload < 0 or payload % self.record_bytes:
            raise ValueError(f"Csonka vagy hibás BDF payload: {path}")
        self.record_count = payload // self.record_bytes
        if declared_records >= 0 and declared_records != self.record_count:
            raise ValueError(f"BDF rekordszám-eltérés: {path}")
        self.signals = [
            Signal(
                columns[0][i], columns[2][i], float(columns[3][i]), float(columns[4][i]),
                float(columns[5][i]), float(columns[6][i]), samples[i], int(byte_offsets[i])
            )
            for i in range(signal_count)
        ]

    @property
    def duration_seconds(self) -> float:
        return self.record_count * self.record_duration

    def index(self, label: str) -> int:
        matches = [i for i, signal in enumerate(self.signals) if signal.label.casefold() == label.casefold()]
        if len(matches) != 1:
            raise ValueError(f"Hiányzó vagy nem egyértelmű BDF-csatorna: {label}")
        return matches[0]

    def rate(self, index: int) -> float:
        return self.signals[index].samples_per_record / self.record_duration

    def batches(self, index: int, records_per_batch: int = 1800):
        signal = self.signals[index]
        memory = np.memmap(
            self.path, dtype=np.uint8, mode="r", offset=self.header_bytes,
            shape=(self.record_count, self.record_bytes),
        )
        for first in range(0, self.record_count, records_per_batch):
            last = min(first + records_per_batch, self.record_count)
            begin = signal.byte_offset
            end = begin + 3 * signal.samples_per_record
            packed = np.array(memory[first:last, begin:end], copy=True)
            yield first * signal.samples_per_record, decode_int24_le(packed)
        del memory


def _timestamp_array(start_ns: int, first: int, count: int, rate: float) -> pa.Array:
    step_ns = round(1_000_000_000 / rate)
    if not np.isclose(step_ns * rate, 1_000_000_000):
        raise ValueError(f"A {rate} Hz nem ábrázolható egész nanoszekundummal.")
    values = start_ns + (first + np.arange(count, dtype=np.int64)) * step_ns
    return pa.array(values, type=pa.timestamp("ns", tz="UTC"))


def _writer(path: Path, schema: pa.Schema) -> pq.ParquetWriter:
    return pq.ParquetWriter(path, schema, compression="zstd", use_dictionary=["source_file"])


def export_bdf(files: list[Path], output: Path, timezone: str) -> list[dict]:
    output.mkdir(parents=True, exist_ok=True)
    ecg_schema = pa.schema([
        ("source_file", pa.string()), ("segment_sample_index", pa.int64()),
        ("timestamp_utc", pa.timestamp("ns", tz="UTC")), ("ecg_digital", pa.int32()),
        ("ecg_mv", pa.float32()),
    ])
    accel_schema = pa.schema([
        ("source_file", pa.string()), ("segment_sample_index", pa.int64()),
        ("timestamp_utc", pa.timestamp("ns", tz="UTC")),
        ("x_mg", pa.int32()), ("y_mg", pa.int32()), ("z_mg", pa.int32()),
    ])
    activity_schema = pa.schema([
        ("source_file", pa.string()), ("segment_sample_index", pa.int64()),
        ("timestamp_utc", pa.timestamp("ns", tz="UTC")),
        ("intensity", pa.int32()), ("classification", pa.int32()),
    ])
    metadata: list[dict] = []
    with _writer(output / "ecg.parquet", ecg_schema) as ecg_writer, _writer(
        output / "accelerometer.parquet", accel_schema
    ) as accel_writer, _writer(output / "activity.parquet", activity_schema) as activity_writer:
        for path in files:
            bdf = BDF(path, timezone)
            start_ns = int(bdf.start.timestamp()) * 1_000_000_000
            ecg_index = bdf.index("ECG")
            ecg_signal = bdf.signals[ecg_index]
            scale = (ecg_signal.physical_max - ecg_signal.physical_min) / (
                ecg_signal.digital_max - ecg_signal.digital_min
            )
            for first, digital in bdf.batches(ecg_index):
                physical = ecg_signal.physical_min + (digital.astype(np.float64) - ecg_signal.digital_min) * scale
                ecg_writer.write_table(pa.Table.from_arrays([
                    pa.array([path.name] * len(digital)), pa.array(first + np.arange(len(digital), dtype=np.int64)),
                    _timestamp_array(start_ns, first, len(digital), bdf.rate(ecg_index)),
                    pa.array(digital), pa.array(physical.astype(np.float32)),
                ], schema=ecg_schema))

            axis_indices = [bdf.index(label) for label in ("X", "Y", "Z")]
            axis_batches = [bdf.batches(index) for index in axis_indices]
            for batches in zip(*axis_batches):
                first = batches[0][0]
                values = [item[1] for item in batches]
                if any(item[0] != first or len(item[1]) != len(values[0]) for item in batches):
                    raise ValueError("Eltérő gyorsulásmérő batch-ek.")
                count = len(values[0])
                accel_writer.write_table(pa.Table.from_arrays([
                    pa.array([path.name] * count), pa.array(first + np.arange(count, dtype=np.int64)),
                    _timestamp_array(start_ns, first, count, bdf.rate(axis_indices[0])),
                    *(pa.array(value) for value in values),
                ], schema=accel_schema))

            activity_indices = [bdf.index(label) for label in ("Intensity", "Classification")]
            activity_batches = [bdf.batches(index) for index in activity_indices]
            for batches in zip(*activity_batches):
                first = batches[0][0]
                values = [item[1] for item in batches]
                count = len(values[0])
                activity_writer.write_table(pa.Table.from_arrays([
                    pa.array([path.name] * count), pa.array(first + np.arange(count, dtype=np.int64)),
                    _timestamp_array(start_ns, first, count, bdf.rate(activity_indices[0])),
                    pa.array(values[0]), pa.array(values[1]),
                ], schema=activity_schema))

            metadata.append({
                "source_file": path.name,
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
                "timezone_assumption": timezone,
                "start_local": bdf.start.isoformat(),
                "end_exclusive_local": datetime.fromtimestamp(
                    bdf.start.timestamp() + bdf.duration_seconds, bdf.timezone
                ).isoformat(),
                "duration_seconds": bdf.duration_seconds,
                "record_count": bdf.record_count,
                "record_duration_seconds": bdf.record_duration,
                "header_bytes": bdf.header_bytes,
                "record_bytes": bdf.record_bytes,
                "patient": bdf.patient,
                "recording": bdf.recording,
                "reserved": bdf.reserved,
                "signals": [asdict(signal) | {"sample_rate_hz": bdf.rate(i)} for i, signal in enumerate(bdf.signals)],
            })
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    for info in archive.infolist():
        relative = PurePosixPath(info.filename)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Nem biztonságos ZIP elérési út: {info.filename!r}")
        target = destination.joinpath(*relative.parts)
        if not target.resolve().is_relative_to(root):
            raise ValueError(f"A ZIP member kilépne a célmappából: {info.filename!r}")
        if info.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open("wb") as sink:
                shutil.copyfileobj(source, sink)


def _table_name(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_]+", name):
        raise ValueError(f"Nem biztonságos SQLite táblanév: {name!r}")
    return name


def _decode_hr(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    decoded = numeric.where(numeric != -1)
    decoded = decoded.where(decoded >= 0, decoded + 256)
    return decoded.astype("Float64")


def export_gadgetbridge(zip_path: Path, output: Path, timezone: str) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    extracted = output / "extracted"
    with zipfile.ZipFile(zip_path) as archive:
        safe_extract(archive, extracted)
        candidates = [info for info in archive.infolist() if info.filename == "database/Gadgetbridge"]
        if len(candidates) != 1:
            raise ValueError("A ZIP-ben nincs pontosan egy database/Gadgetbridge.")
        db_bytes = archive.read(candidates[0])
        inventory = []
        tz = ZoneInfo(timezone)
        for info in archive.infolist():
            inferred = None
            match = re.match(r"(\d{13})_", Path(info.filename).name)
            if match:
                inferred = datetime.fromtimestamp(int(match.group(1)) / 1000, tz).isoformat(timespec="milliseconds")
            inventory.append({
                "path": info.filename, "size_bytes": info.file_size,
                "compressed_size_bytes": info.compress_size, "crc32": f"{info.CRC:08x}",
                "zip_modified": "%04d-%02d-%02dT%02d:%02d:%02d" % info.date_time,
                "filename_epoch_local": inferred,
            })
    if not db_bytes.startswith(SQLITE_MAGIC):
        raise ValueError("A kanonikus Gadgetbridge member nem SQLite.")
    db_path = output / "gadgetbridge.sqlite"
    db_path.write_bytes(db_bytes)
    pd.DataFrame(inventory).to_parquet(output / "zip_inventory.parquet", index=False, compression="zstd")

    connection = sqlite3.connect(":memory:")
    connection.deserialize(db_bytes)
    integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
    if integrity != "ok":
        raise ValueError(f"SQLite integrity_check: {integrity}")
    table_dir = output / "tables"
    table_dir.mkdir(exist_ok=True)
    catalog = []
    tables = [row[0] for row in connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )]
    for raw_name in tables:
        name = _table_name(raw_name)
        schema = connection.execute(f'PRAGMA table_info("{name}")').fetchall()
        row_count = connection.execute(f'SELECT count(*) FROM "{name}"').fetchone()[0]
        catalog.append({
            "name": name, "row_count": row_count,
            "columns": [
                {"cid": row[0], "name": row[1], "declared_type": row[2], "not_null": bool(row[3]),
                 "default": row[4], "primary_key_position": row[5]}
                for row in schema
            ],
        })
        if row_count:
            frame = pd.read_sql_query(f'SELECT * FROM "{name}"', connection)
            frame.to_parquet(table_dir / f"{name}.parquet", index=False, compression="zstd")
    (output / "sqlite_catalog.json").write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")

    activity = pd.read_sql_query("SELECT * FROM HUAWEI_ACTIVITY_SAMPLE", connection)
    activity["timestamp_utc"] = pd.to_datetime(activity["TIMESTAMP"], unit="s", utc=True)
    activity["timestamp_local"] = activity["timestamp_utc"].dt.tz_convert(timezone)
    activity["other_timestamp_utc"] = pd.to_datetime(activity["OTHER_TIMESTAMP"], unit="s", utc=True)
    activity["heart_rate_bpm"] = _decode_hr(activity["HEART_RATE"])
    for raw, cooked in (("STEPS", "steps"), ("DISTANCE", "distance_m")):
        activity[cooked] = pd.to_numeric(activity[raw], errors="coerce").where(activity[raw] >= 0).astype("Float64")
    activity["calories_kcal_inferred"] = (
        pd.to_numeric(activity["CALORIES"], errors="coerce").where(activity["CALORIES"] >= 0) / 1000.0
    ).astype("Float64")
    activity.to_parquet(output / "huawei_activity_raw_normalized.parquet", index=False, compression="zstd")

    fused = activity[(activity["SOURCE"] == 11) & (activity["OTHER_TIMESTAMP"] > activity["TIMESTAMP"])].copy()
    if fused["TIMESTAMP"].duplicated().any():
        raise ValueError("A dokumentált SOURCE=11 előre mutató Huawei activity nézet nem egyedi.")
    fused = fused[[
        "TIMESTAMP", "timestamp_utc", "timestamp_local", "DEVICE_ID", "USER_ID", "SOURCE",
        "RAW_KIND", "RAW_INTENSITY", "HEART_RATE", "heart_rate_bpm", "STEPS", "steps",
        "DISTANCE", "distance_m", "CALORIES", "calories_kcal_inferred", "SPO", "RESTING_HEART_RATE",
    ]].sort_values("TIMESTAMP")
    fused.to_parquet(output / "huawei_activity_fused_1min.parquet", index=False, compression="zstd")

    generic_hr = pd.read_sql_query("SELECT * FROM GENERIC_HEART_RATE_SAMPLE", connection)
    generic_hr["timestamp_utc"] = pd.to_datetime(generic_hr["TIMESTAMP"], unit="ms", utc=True)
    generic_hr["timestamp_local"] = generic_hr["timestamp_utc"].dt.tz_convert(timezone)
    generic_hr.to_parquet(output / "generic_hr_normalized.parquet", index=False, compression="zstd")

    workouts = pd.read_sql_query("SELECT * FROM HUAWEI_WORKOUT_SUMMARY_SAMPLE", connection)
    base = pd.read_sql_query("SELECT * FROM BASE_ACTIVITY_SUMMARY", connection)
    workouts["start_utc"] = pd.to_datetime(workouts["START_TIMESTAMP"], unit="s", utc=True)
    workouts["end_utc"] = pd.to_datetime(workouts["END_TIMESTAMP"], unit="s", utc=True)
    workouts["start_local"] = workouts["start_utc"].dt.tz_convert(timezone)
    workouts["end_local"] = workouts["end_utc"].dt.tz_convert(timezone)
    base_names = dict(zip((base["START_TIME"] // 1000).astype(int), base["NAME"]))
    workouts["display_name"] = workouts["START_TIMESTAMP"].map(base_names)
    workouts.to_parquet(output / "workouts_normalized.parquet", index=False, compression="zstd")

    samples = pd.read_sql_query("SELECT * FROM HUAWEI_WORKOUT_DATA_SAMPLE", connection)
    samples["timestamp_utc"] = pd.to_datetime(samples["TIMESTAMP"], unit="s", utc=True)
    samples["timestamp_local"] = samples["timestamp_utc"].dt.tz_convert(timezone)
    samples["heart_rate_bpm"] = _decode_hr(samples["HEART_RATE"])
    samples.to_parquet(output / "workout_samples_normalized.parquet", index=False, compression="zstd")
    connection.close()
    return {
        "source_file": zip_path.name,
        "sha256": sha256(zip_path),
        "size_bytes": zip_path.stat().st_size,
        "canonical_database_member": "database/Gadgetbridge",
        "canonical_database_sha256": hashlib.sha256(db_bytes).hexdigest(),
        "canonical_database_size_bytes": len(db_bytes),
        "sqlite_integrity_check": integrity,
        "table_count": len(tables),
        "nonempty_table_count": sum(item["row_count"] > 0 for item in catalog),
        "zip_member_count": len(inventory),
        "zero_byte_gps_members": sum(item["size_bytes"] == 0 and "/GPS/" in item["path"] for item in inventory),
        "zero_byte_rri_members": sum(item["size_bytes"] == 0 and "/RRI/" in item["path"] for item in inventory),
    }


def export_text_clipping(path: Path, output: Path) -> dict:
    with path.open("rb") as stream:
        plist = plistlib.load(stream)
    value = plist.get("UTI-Data", {}).get("public.utf8-plain-text")
    result = {
        "source_file": path.name, "sha256": sha256(path), "size_bytes": path.stat().st_size,
        "type": "Apple binary property-list text clipping", "referenced_filename": value,
        "contains_fit_payload": False,
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _haversine_steps(latitude: np.ndarray, longitude: np.ndarray) -> np.ndarray:
    lat = np.radians(np.asarray(latitude, dtype=np.float64))
    lon = np.radians(np.asarray(longitude, dtype=np.float64))
    dlat, dlon = np.diff(lat), np.diff(lon)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat[:-1]) * np.cos(lat[1:]) * np.sin(dlon / 2) ** 2
    return np.r_[0.0, 12_742_000.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))]


def export_gpx(path: Path, output: Path, timezone: str) -> dict:
    """Export all GPX track points and derived, explicitly labelled geometry columns."""
    output.mkdir(parents=True, exist_ok=True)
    root = ET.parse(path).getroot()
    local_name = lambda tag: tag.rsplit("}", 1)[-1]
    if local_name(root.tag) != "gpx":
        raise ValueError(f"Nem GPX gyökérelem: {path}")
    rows = []
    track_names, track_types = [], []
    segment_id = 0
    for track in [child for child in root if local_name(child.tag) == "trk"]:
        name = next((child.text or "" for child in track if local_name(child.tag) == "name"), "")
        kind = next((child.text or "" for child in track if local_name(child.tag) == "type"), "")
        track_names.append(name)
        track_types.append(kind)
        for segment in [child for child in track if local_name(child.tag) == "trkseg"]:
            for point_index, point in enumerate(child for child in segment if local_name(child.tag) == "trkpt"):
                children = {local_name(child.tag): child.text for child in point}
                if "time" not in children:
                    raise ValueError("Időbélyeg nélküli GPX trackpoint.")
                rows.append({
                    "source_file": path.name, "track_name": name, "track_type": kind,
                    "segment_id": segment_id, "segment_point_index": point_index,
                    "latitude_deg": float(point.attrib["lat"]), "longitude_deg": float(point.attrib["lon"]),
                    "elevation_m": float(children["ele"]) if children.get("ele") is not None else np.nan,
                    "timestamp_utc": pd.Timestamp(children["time"]),
                })
            segment_id += 1
    if not rows:
        raise ValueError("A GPX nem tartalmaz trackpointot.")
    frame = pd.DataFrame(rows)
    if frame["timestamp_utc"].dt.tz is None:
        raise ValueError("A GPX időbélyegei nem tartalmaznak időzónát.")
    frame["timestamp_utc"] = frame["timestamp_utc"].dt.tz_convert("UTC")
    frame["timestamp_local"] = frame["timestamp_utc"].dt.tz_convert(timezone)
    frame["delta_time_s"] = frame.groupby("segment_id")["timestamp_utc"].diff().dt.total_seconds()
    distances = np.zeros(len(frame), dtype=np.float64)
    for _, index in frame.groupby("segment_id").groups.items():
        positions = np.asarray(list(index), dtype=int)
        distances[positions] = _haversine_steps(
            frame.loc[positions, "latitude_deg"], frame.loc[positions, "longitude_deg"]
        )
    frame["delta_distance_m"] = distances
    frame["distance_cumulative_m"] = frame["delta_distance_m"].cumsum()
    frame["speed_kmh_raw"] = 3.6 * frame["delta_distance_m"] / frame["delta_time_s"]
    rolling_distance = frame["delta_distance_m"].rolling(15, center=True, min_periods=2).sum()
    rolling_time = frame["delta_time_s"].rolling(15, center=True, min_periods=2).sum()
    frame["speed_kmh_15point"] = 3.6 * rolling_distance / rolling_time
    frame.to_parquet(output / "track_points.parquet", index=False, compression="zstd")
    gap_count = int((frame["delta_time_s"] > 10).sum())
    result = {
        "source_file": path.name, "sha256": sha256(path), "size_bytes": path.stat().st_size,
        "creator": root.attrib.get("creator"), "version": root.attrib.get("version"),
        "track_names": track_names, "track_types": track_types, "segment_count": segment_id,
        "point_count": len(frame), "timezone_from_gpx": "UTC (Z)", "local_timezone": timezone,
        "start_utc": frame["timestamp_utc"].min().isoformat(),
        "end_utc": frame["timestamp_utc"].max().isoformat(),
        "start_local": frame["timestamp_local"].min().isoformat(),
        "end_local": frame["timestamp_local"].max().isoformat(),
        "elapsed_seconds": float((frame["timestamp_utc"].max() - frame["timestamp_utc"].min()).total_seconds()),
        "polyline_distance_m": float(frame["delta_distance_m"].sum()),
        "timestamp_gap_count_over_10s": gap_count,
        "max_timestamp_gap_s": float(frame["delta_time_s"].max()),
        "extension_tags": [],
    }
    (output / "metadata.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _fit_frame(messages: list[dict], developer_names: dict[int, str]) -> pd.DataFrame:
    rows = []
    for message in messages:
        row = {}
        for key, value in message.items():
            if key == "developer_fields" and isinstance(value, dict):
                row["developer_fields_json"] = json.dumps(value, ensure_ascii=False, default=str)
                for number, field_value in value.items():
                    row[f"developer_{developer_names.get(int(number), f'field_{number}')}"] = field_value
            elif isinstance(value, (dict, list, tuple)):
                row[f"{key}_json"] = json.dumps(value, ensure_ascii=False, default=str)
            else:
                row[key] = value
        rows.append(row)
    frame = pd.DataFrame(rows)
    for column in frame.columns:
        if column in {"timestamp", "start_time", "time_created", "local_timestamp"} or column.endswith("_timestamp"):
            if not frame[column].dropna().empty:
                try:
                    frame[column] = pd.to_datetime(frame[column], utc=True)
                except (TypeError, ValueError):
                    pass
    return frame


def export_fit(path: Path, output: Path, timezone: str) -> dict:
    """Decode every FIT message group with Garmin's official Python SDK."""
    vendor = Path(__file__).resolve().parent / ".vendor"
    if str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))
    try:
        from garmin_fit_sdk import Decoder, Stream, __version__ as sdk_version
    except ImportError as exc:
        raise RuntimeError("Hiányzik a hivatalos garmin-fit-sdk a projekt .vendor mappájából.") from exc

    if not Decoder(Stream.from_file(str(path))).is_fit():
        raise ValueError(f"Nem érvényes FIT: {path}")
    integrity = Decoder(Stream.from_file(str(path))).check_integrity()
    decoder = Decoder(Stream.from_file(str(path)))
    messages, errors = decoder.read()
    if errors:
        raise ValueError("FIT dekódolási hibák: " + "; ".join(str(error) for error in errors))
    output.mkdir(parents=True, exist_ok=True)
    descriptions = messages.get("field_description_mesgs", [])
    developer_names = {
        int(item["field_definition_number"]): str(item["field_name"])
        for item in descriptions if "field_definition_number" in item and "field_name" in item
    }
    catalog = []
    frames = {}
    for group, items in messages.items():
        frame = _fit_frame(items, developer_names)
        frames[group] = frame
        frame.to_parquet(output / f"{group}.parquet", index=False, compression="zstd")
        field_counts = {}
        for item in items:
            for key in item:
                field_counts[key] = field_counts.get(key, 0) + 1
        catalog.append({"message_group": group, "message_count": len(items), "field_presence": field_counts})

    records_raw = messages.get("record_mesgs", [])
    by_timestamp: dict[datetime, dict] = {}
    for message in records_raw:
        timestamp = message.get("timestamp")
        if timestamp is None:
            continue
        by_timestamp.setdefault(timestamp, {"timestamp_utc": timestamp}).update(message)
    records = pd.DataFrame(by_timestamp.values()).sort_values("timestamp_utc").reset_index(drop=True)
    records["timestamp_utc"] = pd.to_datetime(records["timestamp_utc"], utc=True)
    records["timestamp_local"] = records["timestamp_utc"].dt.tz_convert(timezone)
    for raw, cooked in (("position_lat", "latitude_deg"), ("position_long", "longitude_deg")):
        if raw in records:
            records[cooked] = pd.to_numeric(records[raw], errors="coerce") * 180.0 / (2**31)
    records["delta_time_s"] = records["timestamp_utc"].diff().dt.total_seconds()
    records["delta_distance_m"] = pd.to_numeric(records.get("distance"), errors="coerce").diff()

    events = frames.get("event_mesgs", pd.DataFrame()).copy()
    if not events.empty:
        events = events.sort_values("timestamp").reset_index(drop=True)
        events["timestamp_local"] = events["timestamp"].dt.tz_convert(timezone)
        events["timer_running_after_event"] = events["event_type"].eq("start")
        timer_state = events[["timestamp", "timer_running_after_event"]].rename(columns={"timestamp": "event_timestamp_utc"})
        records = pd.merge_asof(
            records.sort_values("timestamp_utc"), timer_state,
            left_on="timestamp_utc", right_on="event_timestamp_utc", direction="backward",
        )
        # A record- és event-időbélyegek natív felbontása (ms/us/ns) a forrás és a
        # pandas/pyarrow verzió szerint eltérhet; ns-re kényszerítve biztosítjuk, hogy
        # az int64 időbélyegek összemérhetők legyenek.
        record_ns = records["timestamp_utc"].astype("datetime64[ns, UTC]").astype("int64").to_numpy()
        previous_ns = np.r_[record_ns[0], record_ns[:-1]]
        timer_ns = np.zeros(len(records), dtype=np.int64)
        open_start = None
        for event in events.itertuples():
            event_ns = int(pd.Timestamp(event.timestamp).as_unit("ns").value)
            if event.event_type == "start":
                open_start = event_ns
            elif event.event_type == "stop" and open_start is not None:
                timer_ns += np.maximum(
                    0, np.minimum(record_ns, event_ns) - np.maximum(previous_ns, open_start)
                )
                open_start = None
        records["timer_delta_s"] = timer_ns / 1_000_000_000.0
        events.to_parquet(output / "events_normalized.parquet", index=False, compression="zstd")
    records.to_parquet(output / "records_normalized.parquet", index=False, compression="zstd")

    sessions = messages.get("session_mesgs", [])
    session = sessions[0] if len(sessions) == 1 else {}
    result = {
        "source_file": path.name, "sha256": sha256(path), "size_bytes": path.stat().st_size,
        "garmin_fit_sdk_version": sdk_version, "is_fit": True, "crc_and_size_integrity": bool(integrity),
        "message_group_count": len(messages), "message_count": sum(len(items) for items in messages.values()),
        "message_groups": {group: len(items) for group, items in messages.items()},
        "record_field_names": sorted({key for item in records_raw for key in item}),
        "heart_rate_present": any("heart_rate" in item for item in records_raw),
        "cadence_present": any("cadence" in item for item in records_raw),
        "power_present": any("power" in item for item in records_raw),
        "start_utc": records["timestamp_utc"].min().isoformat(),
        "end_utc": records["timestamp_utc"].max().isoformat(),
        "start_local": records["timestamp_local"].min().isoformat(),
        "end_local": records["timestamp_local"].max().isoformat(),
        "unified_record_count": len(records),
        "session_summary": {key: (value.isoformat() if isinstance(value, datetime) else value) for key, value in session.items()},
    }
    (output / "catalog.json").write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "metadata.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def create_synchronized_timeline(root: Path, timezone: str) -> None:
    watch = pd.read_parquet(root / "gadgetbridge" / "huawei_activity_fused_1min.parquet")
    watch["minute_utc"] = watch["timestamp_utc"].dt.floor("min")
    watch_view = watch[[
        "minute_utc", "heart_rate_bpm", "steps", "distance_m", "calories_kcal_inferred",
        "RAW_INTENSITY",
    ]].rename(columns={"RAW_INTENSITY": "watch_raw_intensity"})
    bdf = pd.read_parquet(root / "bdf" / "activity.parquet")
    bdf["minute_utc"] = bdf["timestamp_utc"].dt.floor("min")
    bdf_view = bdf.groupby("minute_utc", as_index=False).agg(
        bdf_samples=("intensity", "size"), bdf_intensity_mean=("intensity", "mean"),
        bdf_intensity_max=("intensity", "max"), bdf_classification=("classification", "median"),
    )
    timeline = pd.merge(watch_view, bdf_view, on="minute_utc", how="outer").sort_values("minute_utc")
    fit_path = root / "fit" / "records_normalized.parquet"
    if fit_path.is_file():
        fit = pd.read_parquet(fit_path)
        fit["minute_utc"] = fit["timestamp_utc"].dt.floor("min")
        fit_view = fit.groupby("minute_utc", as_index=False).agg(
            fit_records=("timestamp_utc", "size"), fit_distance_m=("distance", "max"),
            fit_speed_mean_mps=("enhanced_speed", "mean"), fit_altitude_mean_m=("enhanced_altitude", "mean"),
            fit_gps_accuracy_median_m=("gps_accuracy", "median"),
            fit_latitude_mean_deg=("latitude_deg", "mean"), fit_longitude_mean_deg=("longitude_deg", "mean"),
        )
        timeline = pd.merge(timeline, fit_view, on="minute_utc", how="outer").sort_values("minute_utc")
    gpx_path = root / "gpx" / "track_points.parquet"
    if gpx_path.is_file():
        gpx = pd.read_parquet(gpx_path)
        gpx["minute_utc"] = gpx["timestamp_utc"].dt.floor("min")
        gpx_view = gpx.groupby("minute_utc", as_index=False).agg(
            gpx_points=("segment_point_index", "size"),
            gpx_distance_m=("delta_distance_m", "sum"),
            gpx_speed_mean_kmh=("speed_kmh_15point", "mean"),
            gpx_elevation_mean_m=("elevation_m", "mean"),
            gpx_latitude_mean_deg=("latitude_deg", "mean"),
            gpx_longitude_mean_deg=("longitude_deg", "mean"),
        )
        timeline = pd.merge(timeline, gpx_view, on="minute_utc", how="outer").sort_values("minute_utc")
    timeline["minute_local"] = timeline["minute_utc"].dt.tz_convert(timezone)
    timeline["watch_available"] = timeline[["heart_rate_bpm", "steps", "distance_m", "calories_kcal_inferred"]].notna().any(axis=1)
    timeline["bdf_available"] = timeline["bdf_samples"].notna()
    timeline["gpx_available"] = timeline.get("gpx_points", pd.Series(index=timeline.index, dtype=float)).notna()
    timeline["fit_available"] = timeline.get("fit_records", pd.Series(index=timeline.index, dtype=float)).notna()
    timeline.to_parquet(root / "synchronized_1min.parquet", index=False, compression="zstd")


def run(input_dir: Path, output: Path, timezone: str) -> None:
    sources = sorted(path for path in input_dir.iterdir() if path.is_file())
    bdf_files = sorted(input_dir.glob("*.bdf"))
    zip_files = sorted(input_dir.glob("*.zip"))
    clipping_files = sorted(input_dir.glob("*.textClipping"))
    gpx_files = sorted(input_dir.glob("*.gpx"))
    fit_files = sorted(input_dir.glob("*.fit"))
    if len(zip_files) != 1 or len(clipping_files) > 1 or len(gpx_files) > 1 or len(fit_files) > 1:
        raise ValueError("A várt input: legfeljebb néhány BDF, egy ZIP, valamint legfeljebb egy FIT/GPX/textClipping.")
    output.mkdir(parents=True, exist_ok=True)
    source_before = {path.name: sha256(path) for path in sources}
    bdf_meta = export_bdf(bdf_files, output / "bdf", timezone)
    gadget_meta = export_gadgetbridge(zip_files[0], output / "gadgetbridge", timezone)
    clipping_meta = export_text_clipping(clipping_files[0], output / "fit_reference.json") if clipping_files else None
    if not clipping_files:
        (output / "fit_reference.json").unlink(missing_ok=True)
    fit_meta = export_fit(fit_files[0], output / "fit", timezone) if fit_files else None
    gpx_meta = export_gpx(gpx_files[0], output / "gpx", timezone) if gpx_files else None
    create_synchronized_timeline(output, timezone)
    source_after = {path.name: sha256(path) for path in sources}
    if source_before != source_after:
        raise RuntimeError("Egy nyers forrás checksumja feldolgozás közben megváltozott.")
    manifest = {
        "created_at_utc": datetime.now(ZoneInfo("UTC")).isoformat(),
        "timezone_assumption": timezone,
        "source_directory": str(input_dir.resolve()),
        "sources": [
            {"file": path.name, "size_bytes": path.stat().st_size, "sha256": source_before[path.name]}
            for path in sources
        ],
        "bdf": bdf_meta,
        "gadgetbridge": gadget_meta,
        "text_clipping": clipping_meta,
        "fit": fit_meta,
        "gpx": gpx_meta,
        "source_priority": {
            "route_coordinates": ["FIT record", "GPX cross-check"],
            "distance_elapsed_timer_ascent_descent": ["FIT session summary", "FIT records", "GPX cross-check"],
            "speed": ["FIT enhanced_speed", "FIT distance/time", "GPX coordinate derivative cross-check"],
            "heart_rate": ["MOX ECG where overlapping", "Huawei activity HR", "FIT (field absent)"],
            "absolute_elevation": ["GPX Strava-corrected profile", "FIT phone GPS altitude cross-check"],
        },
        "warnings": [
            "A valódi FIT CRC-ellenőrzött elsődleges aktivitásforrás; a GPX összehasonlító export.",
            "A Gadgetbridge ZIP mindkét GPS és mind a 44 RRI memberét 0 bájttal tartalmazza.",
            "A BDF nem tárol időzónát; Europe/Belgrade értelmezést alkalmaztunk.",
            "A calories_kcal_inferred mező a Huawei nyers CALORIES/1000 értelmezése; az eredeti érték is megmaradt.",
            "A szinkronizált tábla outer join, interpoláció és résáthidalás nélkül.",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Kész nyersadat-réteg: {output.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BDF/Gadgetbridge/FIT-clipping normalizálás Parquet formára.")
    parser.add_argument("--input", type=Path, default=Path("09_05"))
    parser.add_argument("--output", type=Path, default=Path("09_05/processed"))
    parser.add_argument("--timezone", default=TIMEZONE)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.input, args.output, args.timezone)
