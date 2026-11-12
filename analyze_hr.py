#!/usr/bin/env python3
"""Huawei/Gadgetbridge HR comparison against a MOX BDF ECG reference."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import sqlite3
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Sequence
from urllib.parse import quote
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, find_peaks, sosfiltfilt


DEFAULT_BDF = "08_28/15673_20260828_164058.bdf"
DEFAULT_DB = "08_28/Gadgetbridge_20260828-213228"
SQLITE_HEADER = b"SQLite format 3\x00"
MAX_ZIP_SQLITE_BYTES = 512 * 1024 * 1024


def _ascii(field: bytes) -> str:
    return field.decode("latin-1").strip()


def decode_int24_le(raw: np.ndarray) -> np.ndarray:
    """Decode BDF little-endian signed 24-bit samples into int32."""
    data = np.asarray(raw, dtype=np.uint8)
    if data.shape[-1] != 3:
        raise ValueError("A 24 bites minták utolsó dimenziója 3 kell legyen.")
    value = (
        data[..., 0].astype(np.int32)
        | (data[..., 1].astype(np.int32) << 8)
        | (data[..., 2].astype(np.int32) << 16)
    )
    return np.where(value & 0x800000, value - (1 << 24), value).astype(np.int32)


def physical_scale(
    digital: np.ndarray,
    physical_min: float,
    physical_max: float,
    digital_min: float,
    digital_max: float,
) -> np.ndarray:
    if digital_max == digital_min:
        raise ValueError("Érvénytelen BDF skála: digital min == digital max.")
    return physical_min + (np.asarray(digital, dtype=np.float64) - digital_min) * (
        (physical_max - physical_min) / (digital_max - digital_min)
    )


@dataclass(frozen=True)
class BDFSignal:
    label: str
    physical_dimension: str
    physical_min: float
    physical_max: float
    digital_min: float
    digital_max: float
    samples_per_record: int
    byte_offset_in_record: int


class BDFReader:
    """Small, purpose-built BDF reader for signal metadata and 24-bit samples."""

    def __init__(self, path: Path, timezone: str):
        self.path = Path(path)
        try:
            self.timezone = ZoneInfo(timezone)
        except Exception as exc:
            raise ValueError(f"Ismeretlen időzóna: {timezone}") from exc

        with self.path.open("rb") as stream:
            fixed = stream.read(256)
            if len(fixed) != 256:
                raise ValueError("A BDF fejléc túl rövid.")
            self.header_bytes = int(_ascii(fixed[184:192]))
            header = fixed + stream.read(self.header_bytes - 256)
        if len(header) != self.header_bytes:
            raise ValueError("A BDF fejléc csonka.")
        if not header[:8].startswith(b"\xffBIOSEMI"):
            raise ValueError("A fájl nem támogatott 24 bites BioSemi BDF.")

        date_text = _ascii(header[168:176])
        time_text = _ascii(header[176:184])
        naive = datetime.strptime(f"{date_text} {time_text}", "%d.%m.%y %H.%M.%S")
        self.start = naive.replace(tzinfo=self.timezone)
        self.record_duration = float(_ascii(header[244:252]))
        if self.record_duration <= 0:
            raise ValueError("A BDF rekordhossz nem pozitív.")
        declared_records = int(_ascii(header[236:244]))
        signal_count = int(_ascii(header[252:256]))
        if signal_count <= 0:
            raise ValueError("A BDF nem tartalmaz jelcsatornát.")

        offset = 256
        widths = (16, 80, 8, 8, 8, 8, 8, 80, 8, 32)
        columns: list[list[str]] = []
        for width in widths:
            columns.append(
                [
                    _ascii(header[offset + i * width : offset + (i + 1) * width])
                    for i in range(signal_count)
                ]
            )
            offset += signal_count * width
        samples_per_record = [int(value) for value in columns[8]]
        if any(value <= 0 for value in samples_per_record):
            raise ValueError("A BDF egyik csatornájának mintaszáma nem pozitív.")
        self.record_bytes = 3 * sum(samples_per_record)
        payload_bytes = self.path.stat().st_size - self.header_bytes
        if payload_bytes < 0 or payload_bytes % self.record_bytes:
            raise ValueError("A BDF adatblokk mérete nem egész számú rekord.")
        actual_records = payload_bytes // self.record_bytes
        if declared_records >= 0 and declared_records != actual_records:
            raise ValueError(
                f"BDF rekordeltérés: fejléc={declared_records}, fájl={actual_records}."
            )
        self.record_count = int(actual_records)
        byte_offsets = np.cumsum([0] + samples_per_record[:-1]) * 3
        self.signals = tuple(
            BDFSignal(
                label=columns[0][i],
                physical_dimension=columns[2][i],
                physical_min=float(columns[3][i]),
                physical_max=float(columns[4][i]),
                digital_min=float(columns[5][i]),
                digital_max=float(columns[6][i]),
                samples_per_record=samples_per_record[i],
                byte_offset_in_record=int(byte_offsets[i]),
            )
            for i in range(signal_count)
        )

    @property
    def duration_seconds(self) -> float:
        return self.record_count * self.record_duration

    @property
    def end_timestamp(self) -> float:
        return self.start.timestamp() + self.duration_seconds

    def find_signal(self, wanted: str) -> int:
        exact = [i for i, signal in enumerate(self.signals) if signal.label.casefold() == wanted.casefold()]
        if len(exact) == 1:
            return exact[0]
        contains = [i for i, signal in enumerate(self.signals) if wanted.casefold() in signal.label.casefold()]
        if len(contains) == 1:
            return contains[0]
        labels = ", ".join(repr(signal.label) for signal in self.signals)
        if not exact and not contains:
            raise ValueError(f"Nem található {wanted!r} BDF csatorna. Elérhető: {labels}")
        raise ValueError(f"Nem egyértelmű {wanted!r} BDF csatorna. Elérhető: {labels}")

    def sample_rate(self, index: int) -> float:
        return self.signals[index].samples_per_record / self.record_duration

    def read_signal(self, index: int, physical: bool = True) -> np.ndarray:
        signal = self.signals[index]
        memory = np.memmap(
            self.path,
            dtype=np.uint8,
            mode="r",
            offset=self.header_bytes,
            shape=(self.record_count, self.record_bytes),
        )
        begin = signal.byte_offset_in_record
        end = begin + 3 * signal.samples_per_record
        packed = np.array(memory[:, begin:end], copy=True).reshape(-1, 3)
        del memory
        digital = decode_int24_le(packed)
        if not physical:
            return digital
        return physical_scale(
            digital,
            signal.physical_min,
            signal.physical_max,
            signal.digital_min,
            signal.digital_max,
        )


@dataclass(frozen=True)
class WatchData:
    raw_timestamp: np.ndarray
    timestamp: np.ndarray
    heart_rate: np.ndarray
    table: str
    timestamp_unit: str
    source_row_count: int
    archive_member: str | None


def _quoted_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _validate_zip_member(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> None:
    if info.is_dir() or info.file_size < 100:
        raise ValueError(f"A ZIP SQLite member túl rövid vagy könyvtár: {info.filename!r}.")
    if info.file_size > MAX_ZIP_SQLITE_BYTES:
        raise ValueError(
            f"A ZIP SQLite member túl nagy ({info.file_size} byte): {info.filename!r}."
        )
    if info.flag_bits & 0x1:
        raise ValueError(f"A ZIP SQLite member titkosított: {info.filename!r}.")
    try:
        with archive.open(info, "r") as stream:
            header = stream.read(len(SQLITE_HEADER))
    except (RuntimeError, zipfile.BadZipFile) as exc:
        raise ValueError(f"A ZIP member nem olvasható: {info.filename!r}.") from exc
    if header != SQLITE_HEADER:
        raise ValueError(f"A ZIP member nem SQLite adatbázis: {info.filename!r}.")


def choose_gadgetbridge_zip_member(archive: zipfile.ZipFile) -> zipfile.ZipInfo:
    """Prefer the canonical export DB; allow only one header-confirmed fallback."""
    canonical = [
        info
        for info in archive.infolist()
        if info.filename == "database/Gadgetbridge"
    ]
    if len(canonical) > 1:
        raise ValueError("A ZIP többször tartalmazza a kanonikus database/Gadgetbridge membert.")
    if canonical:
        _validate_zip_member(archive, canonical[0])
        return canonical[0]

    candidates = []
    for info in archive.infolist():
        if info.is_dir() or not 100 <= info.file_size <= MAX_ZIP_SQLITE_BYTES:
            continue
        try:
            with archive.open(info, "r") as stream:
                is_sqlite = stream.read(len(SQLITE_HEADER)) == SQLITE_HEADER
        except (RuntimeError, zipfile.BadZipFile):
            is_sqlite = False
        if is_sqlite:
            candidates.append(info)
    if len(candidates) != 1:
        names = ", ".join(repr(info.filename) for info in candidates) or "nincs"
        raise ValueError(
            "A ZIP-ben nincs kanonikus database/Gadgetbridge; az SQLite fallback "
            f"nem egyértelmű ({names})."
        )
    _validate_zip_member(archive, candidates[0])
    return candidates[0]


def _open_watch_connection(path: Path) -> tuple[sqlite3.Connection, str | None]:
    archive_member = None
    if path.suffix.casefold() == ".zip" or zipfile.is_zipfile(path):
        try:
            with zipfile.ZipFile(path, "r") as archive:
                info = choose_gadgetbridge_zip_member(archive)
                database = archive.read(info)
                archive_member = info.filename
        except (zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
            raise ValueError(f"Érvénytelen Gadgetbridge ZIP: {path.name!r}.") from exc
        if len(database) != info.file_size or database[: len(SQLITE_HEADER)] != SQLITE_HEADER:
            raise ValueError(f"A ZIP SQLite member sérült vagy csonka: {archive_member!r}.")
        if not hasattr(sqlite3.Connection, "deserialize"):
            raise RuntimeError("Ez a Python sqlite3 build nem támogatja a memóriabeli deserialize-t.")
        connection = sqlite3.connect(":memory:")
        try:
            connection.deserialize(database)
        except sqlite3.DatabaseError as exc:
            connection.close()
            raise ValueError(f"A ZIP SQLite member nem tölthető be: {archive_member!r}.") from exc
    else:
        with path.open("rb") as stream:
            header = stream.read(len(SQLITE_HEADER))
        if path.stat().st_size < 100 or header != SQLITE_HEADER:
            raise ValueError(f"A watch input nem SQLite fájl vagy Gadgetbridge ZIP: {path.name!r}.")
        uri = f"file:{quote(str(path))}?mode=ro&immutable=1"
        connection = sqlite3.connect(uri, uri=True)
    connection.execute("PRAGMA query_only=ON")
    return connection, archive_member


def load_watch_data(path: Path) -> WatchData:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"A watch input nem található: {path}")
    connection, archive_member = _open_watch_connection(path)
    try:
        tables = [
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
        ]
        candidates: list[tuple[int, str, str, str]] = []
        for table in tables:
            info = connection.execute(
                f"PRAGMA table_info({_quoted_identifier(table)})"
            ).fetchall()
            by_lower = {row[1].casefold(): row[1] for row in info}
            if "timestamp" in by_lower and "heart_rate" in by_lower:
                priority = 0 if table.casefold() == "generic_heart_rate_sample" else 1
                candidates.append(
                    (priority, table, by_lower["timestamp"], by_lower["heart_rate"])
                )
        if not candidates:
            raise ValueError("Nem található TIMESTAMP + HEART_RATE oszlopú SQLite tábla.")
        _, table, timestamp_column, hr_column = sorted(candidates)[0]
        query = (
            f"SELECT {_quoted_identifier(timestamp_column)}, {_quoted_identifier(hr_column)} "
            f"FROM {_quoted_identifier(table)} ORDER BY {_quoted_identifier(timestamp_column)}"
        )
        rows = connection.execute(query).fetchall()
    finally:
        connection.close()
    if not rows:
        raise ValueError(f"A kiválasztott {table} tábla üres.")

    raw_timestamp = np.asarray([row[0] for row in rows], dtype=np.float64)
    heart_rate = np.asarray([row[1] for row in rows], dtype=np.float64)
    magnitude = float(np.nanmedian(np.abs(raw_timestamp)))
    if magnitude >= 1e17:
        divisor, unit = 1e9, "ns"
    elif magnitude >= 1e14:
        divisor, unit = 1e6, "µs"
    elif magnitude >= 1e11:
        divisor, unit = 1e3, "ms"
    else:
        divisor, unit = 1.0, "s"
    timestamp = raw_timestamp / divisor
    finite = np.isfinite(timestamp) & np.isfinite(heart_rate)
    timestamp, raw_timestamp, heart_rate = (
        timestamp[finite],
        raw_timestamp[finite],
        heart_rate[finite],
    )
    order = np.argsort(timestamp, kind="stable")
    return WatchData(
        raw_timestamp=raw_timestamp[order],
        timestamp=timestamp[order],
        heart_rate=heart_rate[order],
        table=table,
        timestamp_unit=unit,
        source_row_count=len(rows),
        archive_member=archive_member,
    )


def contiguous_intervals(
    timestamps: Sequence[float], gap_threshold: float | None = None
) -> tuple[list[tuple[float, float]], float]:
    values = np.asarray(timestamps, dtype=np.float64)
    values = np.unique(values[np.isfinite(values)])
    if not len(values):
        return [], 5.0 if gap_threshold is None else float(gap_threshold)
    positive = np.diff(values)
    positive = positive[positive > 0]
    cadence = float(np.median(positive)) if len(positive) else 1.0
    threshold = max(5.0, 3.0 * cadence) if gap_threshold is None else float(gap_threshold)
    split = np.flatnonzero(np.diff(values) > threshold)
    starts = np.r_[0, split + 1]
    ends = np.r_[split, len(values) - 1]
    return [(float(values[a]), float(values[b])) for a, b in zip(starts, ends)], threshold


def intersect_intervals(
    intervals: Iterable[tuple[float, float]], start: float, end: float
) -> list[tuple[float, float]]:
    result = []
    for left, right in intervals:
        overlap = (max(left, start), min(right, end))
        if overlap[0] <= overlap[1]:
            result.append(overlap)
    return result


@dataclass
class QRSBlock:
    start: float
    end: float
    quality: float
    usable: bool
    reason: str
    peaks: np.ndarray
    agreement: float
    plausible_rr: float
    prominence_snr: float
    morphology_consistency: float = math.nan
    off_qrs_p99_ratio: float = math.nan
    contaminated_fraction: float = math.nan


@dataclass
class ReferenceSegment:
    start: float
    end: float
    rr_midpoint: np.ndarray
    instantaneous_hr: np.ndarray


@dataclass
class ECGReference:
    blocks: list[QRSBlock]
    segments: list[ReferenceSegment]
    window_seconds: float = 5.0

    def evaluate(self, query_time: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
        query = np.asarray(query_time, dtype=np.float64)
        reference = np.full(query.shape, np.nan, dtype=np.float64)
        quality = np.zeros(query.shape, dtype=np.float64)
        segment_index = 0
        for block in self.blocks:
            mask = (query >= block.start) & (query < block.end)
            quality[mask] = block.quality
            if not block.usable or not np.any(mask):
                continue
            while segment_index + 1 < len(self.segments) and self.segments[segment_index].end <= block.start:
                segment_index += 1
            segment = self.segments[segment_index]
            for output_index in np.flatnonzero(mask):
                t = query[output_index]
                lo = np.searchsorted(segment.rr_midpoint, t - self.window_seconds)
                hi = np.searchsorted(segment.rr_midpoint, t + self.window_seconds)
                values = segment.instantaneous_hr[lo:hi]
                values = values[(values >= 30.0) & (values <= 220.0)]
                if len(values) >= 5:
                    reference[output_index] = float(np.median(values))
        return reference, quality


def _detect_qrs_legacy(
    ecg: np.ndarray,
    sample_rate: float,
    start_timestamp: float,
    block_seconds: float = 60.0,
) -> ECGReference:
    if sample_rate < 70:
        raise ValueError("Az ECG mintavétel túl alacsony a QRS-detektáláshoz.")
    signal = np.asarray(ecg, dtype=np.float64)
    if len(signal) < 10 * sample_rate:
        raise ValueError("Az ECG túl rövid a QRS-detektáláshoz.")
    high_cut = min(30.0, 0.45 * sample_rate)
    filtered = sosfiltfilt(
        butter(3, (5.0, high_cut), btype="bandpass", fs=sample_rate, output="sos"),
        signal,
    )
    block_samples = max(int(round(block_seconds * sample_rate)), int(10 * sample_rate))
    blocks: list[QRSBlock] = []
    minimum_distance = max(1, int(round(0.30 * sample_rate)))
    corroboration_distance = int(round(0.12 * sample_rate))
    energy_width = max(3, int(round(0.10 * sample_rate)))

    for first in range(0, len(filtered), block_samples):
        last = min(len(filtered), first + block_samples)
        section = filtered[first:last]
        if len(section) < 10 * sample_rate:
            continue
        median = float(np.median(section))
        sigma = 1.4826 * float(np.median(np.abs(section - median))) + 1e-12
        positive_tail = float(np.percentile(section, 99.5) - median)
        negative_tail = float(median - np.percentile(section, 0.5))
        polarity = 1.0 if positive_tail >= negative_tail else -1.0
        oriented = polarity * section
        primary, properties = find_peaks(
            oriented,
            distance=minimum_distance,
            height=float(np.median(oriented)) + 1.5 * sigma,
            prominence=3.0 * sigma,
        )

        derivative = np.diff(section, prepend=section[0])
        energy = uniform_filter1d(
            derivative * derivative, size=energy_width, mode="nearest"
        )
        energy_median = float(np.median(energy))
        energy_sigma = 1.4826 * float(np.median(np.abs(energy - energy_median))) + 1e-15
        secondary, _ = find_peaks(
            energy,
            distance=minimum_distance,
            height=energy_median + 6.0 * energy_sigma,
            prominence=3.0 * energy_sigma,
        )

        if len(primary) and len(secondary):
            insertion = np.searchsorted(secondary, primary)
            left_distance = np.where(
                insertion > 0,
                np.abs(primary - secondary[np.maximum(insertion - 1, 0)]),
                np.iinfo(np.int32).max,
            )
            right_distance = np.where(
                insertion < len(secondary),
                np.abs(primary - secondary[np.minimum(insertion, len(secondary) - 1)]),
                np.iinfo(np.int32).max,
            )
            corroborated = np.minimum(left_distance, right_distance) <= corroboration_distance
        else:
            corroborated = np.zeros(len(primary), dtype=bool)
        accepted = primary[corroborated]
        rr = np.diff(accepted) / sample_rate
        plausible = (rr >= 0.30) & (rr <= 2.0)
        agreement = float(np.mean(corroborated)) if len(primary) else 0.0
        plausible_ratio = float(np.mean(plausible)) if len(rr) else 0.0
        prominence_snr = (
            float(np.median(properties["prominences"][corroborated]) / sigma)
            if np.any(corroborated)
            else 0.0
        )
        median_prominence = (
            float(np.median(properties["prominences"][corroborated]))
            if np.any(corroborated)
            else 0.0
        )
        rr_coverage = (
            float((accepted[-1] - accepted[0]) / max(1, len(section) - 1))
            if len(accepted) >= 2
            else 0.0
        )
        # MOX stores this channel in mV. These absolute rails prevent an
        # adaptive threshold from interpreting a disconnected flat lead or
        # high-energy motion as a clean, merely lower/higher-amplitude ECG.
        signal_floor_ok = max(positive_tail, negative_tail) >= 0.05
        artifact_floor_ok = sigma <= 0.15
        count_factor = min(1.0, len(accepted) / max(20.0, block_seconds * 0.65))
        quality = float(
            min(1.0, agreement / 0.85)
            * min(1.0, plausible_ratio / 0.95)
            * min(1.0, prominence_snr / 8.0)
            * min(1.0, median_prominence / 0.05)
            * min(1.0, rr_coverage / 0.85)
            * count_factor
        )
        usable = bool(
            len(accepted) >= max(20, int(block_seconds * 0.45))
            and agreement >= 0.75
            and plausible_ratio >= 0.90
            and prominence_snr >= 3.5
            and median_prominence >= 0.05
            and rr_coverage >= 0.85
            and signal_floor_ok
            and artifact_floor_ok
        )
        failed = []
        if len(accepted) < max(20, int(block_seconds * 0.45)):
            failed.append("kevés_QRS")
        if agreement < 0.75:
            failed.append("detektor_eltérés")
        if plausible_ratio < 0.90:
            failed.append("RR_inkonzisztens")
        if prominence_snr < 3.5:
            failed.append("alacsony_SNR")
        if median_prominence < 0.05 or not signal_floor_ok:
            failed.append("abszolút_jelszint")
        if rr_coverage < 0.85:
            failed.append("elégtelen_RR_lefedettség")
        if not artifact_floor_ok:
            failed.append("magas_zajszint")
        absolute_peaks = start_timestamp + (first + accepted) / sample_rate
        blocks.append(
            QRSBlock(
                start=start_timestamp + first / sample_rate,
                end=start_timestamp + last / sample_rate,
                quality=quality,
                usable=usable,
                reason="megfelelő" if usable else "+".join(failed),
                peaks=absolute_peaks,
                agreement=agreement,
                plausible_rr=plausible_ratio,
                prominence_snr=prominence_snr,
            )
        )

    segments: list[ReferenceSegment] = []
    run: list[QRSBlock] = []

    def finish_run(items: list[QRSBlock]) -> None:
        if not items:
            return
        peaks = np.unique(np.concatenate([item.peaks for item in items]))
        rr = np.diff(peaks)
        valid = (rr >= 0.30) & (rr <= 2.0)
        segments.append(
            ReferenceSegment(
                start=items[0].start,
                end=items[-1].end,
                rr_midpoint=((peaks[:-1] + peaks[1:]) / 2.0)[valid],
                instantaneous_hr=(60.0 / rr)[valid],
            )
        )

    for block in blocks:
        if block.usable and (not run or math.isclose(run[-1].end, block.start, abs_tol=1e-6)):
            run.append(block)
        else:
            finish_run(run)
            run = [block] if block.usable else []
    finish_run(run)
    if not segments:
        raise ValueError("Egyetlen EKG blokk sem ment át a QRS minőségkapun.")
    return ECGReference(blocks=blocks, segments=segments)


def _template_correlations(
    signal: np.ndarray,
    peaks: np.ndarray,
    corroborated: np.ndarray,
    sample_rate: float,
) -> np.ndarray:
    """Correlate aligned beats with a robust local median QRS template."""
    correlations = np.zeros(len(peaks), dtype=np.float64)
    before = max(2, int(round(0.08 * sample_rate)))
    after = max(3, int(round(0.12 * sample_rate)))
    candidate_index = []
    clips = []
    for index in np.flatnonzero(corroborated):
        peak = int(peaks[index])
        if peak - before < 0 or peak + after >= len(signal):
            continue
        clip = np.asarray(signal[peak - before : peak + after + 1], dtype=np.float64)
        clip = clip - np.median(clip)
        norm = float(np.linalg.norm(clip))
        if norm <= 1e-12:
            continue
        candidate_index.append(index)
        clips.append(clip / norm)
    if len(clips) < 5:
        return correlations
    matrix = np.vstack(clips)
    template = np.median(matrix, axis=0)
    template_norm = float(np.linalg.norm(template))
    if template_norm <= 1e-12:
        return correlations
    template /= template_norm
    correlations[np.asarray(candidate_index)] = matrix @ template
    return correlations


def _filter_ecg(ecg: np.ndarray, sample_rate: float) -> np.ndarray:
    """Zero-phase baseline removal and QRS band-pass used by all diagnostics."""
    signal = np.asarray(ecg, dtype=np.float64)
    baseline_removed = sosfiltfilt(
        butter(2, 0.7, btype="highpass", fs=sample_rate, output="sos"), signal
    )
    high_cut = min(30.0, 0.45 * sample_rate)
    return sosfiltfilt(
        butter(3, (5.0, high_cut), btype="bandpass", fs=sample_rate, output="sos"),
        baseline_removed,
    )


def detect_qrs(
    ecg: np.ndarray,
    sample_rate: float,
    start_timestamp: float,
    block_seconds: float = 30.0,
) -> ECGReference:
    """Detect QRS complexes in short local windows with ECG-only morphology SQI."""
    if sample_rate < 70:
        raise ValueError("Az ECG mintavétel túl alacsony a QRS-detektáláshoz.")
    signal = np.asarray(ecg, dtype=np.float64)
    if len(signal) < 10 * sample_rate:
        raise ValueError("Az ECG túl rövid a QRS-detektáláshoz.")

    # Both stages are zero-phase, so baseline removal does not shift R peaks.
    filtered = _filter_ecg(signal, sample_rate)

    block_samples = max(int(round(block_seconds * sample_rate)), int(10 * sample_rate))
    padding = int(round(5.0 * sample_rate))
    minimum_distance = max(1, int(round(0.30 * sample_rate)))
    corroboration_distance = int(round(0.12 * sample_rate))
    energy_width = max(3, int(round(0.10 * sample_rate)))
    blocks: list[QRSBlock] = []

    for core_first in range(0, len(filtered), block_samples):
        core_last = min(len(filtered), core_first + block_samples)
        if core_last - core_first < 10 * sample_rate:
            continue
        analysis_first = max(0, core_first - padding)
        analysis_last = min(len(filtered), core_last + padding)
        section = filtered[analysis_first:analysis_last]
        median = float(np.median(section))
        sigma = 1.4826 * float(np.median(np.abs(section - median))) + 1e-12
        positive_tail = float(np.percentile(section, 99.5) - median)
        negative_tail = float(median - np.percentile(section, 0.5))
        polarity = 1.0 if positive_tail >= negative_tail else -1.0
        oriented = polarity * section
        primary, properties = find_peaks(
            oriented,
            distance=minimum_distance,
            height=float(np.median(oriented)) + 1.5 * sigma,
            prominence=3.0 * sigma,
        )

        derivative = np.diff(section, prepend=section[0])
        energy = uniform_filter1d(
            derivative * derivative, size=energy_width, mode="nearest"
        )
        energy_median = float(np.median(energy))
        energy_sigma = 1.4826 * float(np.median(np.abs(energy - energy_median))) + 1e-15
        secondary, _ = find_peaks(
            energy,
            distance=minimum_distance,
            height=energy_median + 6.0 * energy_sigma,
            prominence=3.0 * energy_sigma,
        )
        if len(primary) and len(secondary):
            insertion = np.searchsorted(secondary, primary)
            left_distance = np.where(
                insertion > 0,
                np.abs(primary - secondary[np.maximum(insertion - 1, 0)]),
                np.iinfo(np.int32).max,
            )
            right_distance = np.where(
                insertion < len(secondary),
                np.abs(primary - secondary[np.minimum(insertion, len(secondary) - 1)]),
                np.iinfo(np.int32).max,
            )
            corroborated = np.minimum(left_distance, right_distance) <= corroboration_distance
        else:
            corroborated = np.zeros(len(primary), dtype=bool)

        correlations = _template_correlations(
            oriented, primary, corroborated, sample_rate
        )
        morphology_ok = correlations >= 0.65
        global_primary = analysis_first + primary
        in_core = (global_primary >= core_first) & (global_primary < core_last)
        corroborated_core = corroborated & in_core
        accepted_mask = corroborated & morphology_ok & in_core
        accepted = global_primary[accepted_mask]
        primary_core_count = int(np.count_nonzero(in_core))
        agreement = (
            float(np.count_nonzero(corroborated_core) / primary_core_count)
            if primary_core_count
            else 0.0
        )
        morphology_consistency = (
            float(np.mean(morphology_ok[corroborated_core]))
            if np.any(corroborated_core)
            else 0.0
        )
        median_template_correlation = (
            float(np.median(correlations[accepted_mask]))
            if np.any(accepted_mask)
            else 0.0
        )
        rr = np.diff(accepted) / sample_rate
        plausible = (rr >= 0.30) & (rr <= 2.0)
        plausible_ratio = float(np.mean(plausible)) if len(rr) else 0.0
        core_prominences = properties["prominences"][accepted_mask]
        prominence_snr = (
            float(np.median(core_prominences) / sigma) if len(core_prominences) else 0.0
        )
        median_prominence = (
            float(np.median(core_prominences)) if len(core_prominences) else 0.0
        )
        rr_coverage = (
            float((accepted[-1] - accepted[0]) / max(1, core_last - core_first - 1))
            if len(accepted) >= 2
            else 0.0
        )
        signal_floor_ok = (
            max(positive_tail, negative_tail) >= 0.05 and median_prominence >= 0.05
        )
        core_signal = filtered[core_first:core_last]
        off_qrs_mask = np.ones(len(core_signal), dtype=bool)
        exclusion_width = int(round(0.12 * sample_rate))
        for peak in accepted - core_first:
            peak = int(peak)
            off_qrs_mask[
                max(0, peak - exclusion_width) : min(
                    len(core_signal), peak + exclusion_width + 1
                )
            ] = False
        off_qrs = np.abs(core_signal[off_qrs_mask] - np.median(core_signal))
        if len(off_qrs) and median_prominence > 0:
            off_qrs_p99_ratio = float(np.percentile(off_qrs, 99.0) / median_prominence)
            off_qrs_rms_ratio = float(
                np.sqrt(np.mean(off_qrs**2)) / median_prominence
            )
            contaminated_fraction = float(
                np.mean(off_qrs > 0.5 * median_prominence)
            )
        else:
            off_qrs_p99_ratio = math.inf
            off_qrs_rms_ratio = math.inf
            contaminated_fraction = 1.0
        transient_artifact_ok = (
            off_qrs_p99_ratio <= 0.55
            and off_qrs_rms_ratio <= 0.16
            and contaminated_fraction <= 0.02
        )
        count_floor = max(10, int(block_seconds * 0.45))
        count_factor = min(1.0, len(accepted) / max(10.0, block_seconds * 0.65))
        contamination_factor = min(
            1.0,
            0.25 / max(off_qrs_p99_ratio, 1e-12),
        ) * min(1.0, 0.08 / max(off_qrs_rms_ratio, 1e-12))
        quality = float(
            min(1.0, agreement / 0.85)
            * min(1.0, plausible_ratio / 0.95)
            * min(1.0, prominence_snr / 8.0)
            * min(1.0, median_prominence / 0.05)
            * min(1.0, rr_coverage / 0.85)
            * min(1.0, morphology_consistency / 0.90)
            * min(1.0, median_template_correlation / 0.80)
            * count_factor
            * contamination_factor
        )
        usable = bool(
            len(accepted) >= count_floor
            and agreement >= 0.75
            and morphology_consistency >= 0.80
            and median_template_correlation >= 0.75
            and plausible_ratio >= 0.90
            and prominence_snr >= 3.5
            and signal_floor_ok
            and rr_coverage >= 0.85
            and transient_artifact_ok
        )
        failed = []
        if len(accepted) < count_floor:
            failed.append("kevés_QRS")
        if agreement < 0.75:
            failed.append("detektor_eltérés")
        if morphology_consistency < 0.80 or median_template_correlation < 0.75:
            failed.append("morfológia_inkonzisztens")
        if plausible_ratio < 0.90:
            failed.append("RR_inkonzisztens")
        if prominence_snr < 3.5:
            failed.append("alacsony_SNR")
        if not signal_floor_ok:
            failed.append("abszolút_jelszint")
        if rr_coverage < 0.85:
            failed.append("elégtelen_RR_lefedettség")
        if not transient_artifact_ok:
            failed.append("off_QRS_artifact")
        blocks.append(
            QRSBlock(
                start=start_timestamp + core_first / sample_rate,
                end=start_timestamp + core_last / sample_rate,
                quality=quality,
                usable=usable,
                reason="megfelelő" if usable else "+".join(failed),
                peaks=start_timestamp + accepted / sample_rate,
                agreement=agreement,
                plausible_rr=plausible_ratio,
                prominence_snr=prominence_snr,
                morphology_consistency=morphology_consistency,
                off_qrs_p99_ratio=off_qrs_p99_ratio,
                contaminated_fraction=contaminated_fraction,
            )
        )

    segments: list[ReferenceSegment] = []
    run: list[QRSBlock] = []

    def finish_run(items: list[QRSBlock]) -> None:
        if not items:
            return
        peaks = np.unique(np.concatenate([item.peaks for item in items]))
        rr = np.diff(peaks)
        valid = (rr >= 0.30) & (rr <= 2.0)
        segments.append(
            ReferenceSegment(
                start=items[0].start,
                end=items[-1].end,
                rr_midpoint=((peaks[:-1] + peaks[1:]) / 2.0)[valid],
                instantaneous_hr=(60.0 / rr)[valid],
            )
        )

    for block in blocks:
        if block.usable and (
            not run or math.isclose(run[-1].end, block.start, abs_tol=1e-6)
        ):
            run.append(block)
        else:
            finish_run(run)
            run = [block] if block.usable else []
    finish_run(run)
    if not segments:
        raise ValueError("Egyetlen EKG blokk sem ment át a QRS minőségkapun.")
    return ECGReference(blocks=blocks, segments=segments)


def _alignment_score(watch: np.ndarray, reference: np.ndarray) -> float:
    valid = np.isfinite(watch) & np.isfinite(reference)
    if np.count_nonzero(valid) < 30:
        return math.inf
    residual = watch[valid] - reference[valid]
    centered = residual - np.median(residual)
    absolute = np.abs(centered)
    cutoff = np.percentile(absolute, 90.0)
    trimmed = absolute[absolute <= cutoff]
    if not len(trimmed) or np.std(reference[valid]) < 1.0:
        return math.inf
    return float(np.mean(trimmed))


@dataclass(frozen=True)
class Alignment:
    offset_seconds: float
    drift_seconds_per_second: float
    center_timestamp: float
    score: float
    zero_offset_score: float
    drift_applied: bool
    local_offsets: tuple[tuple[float, float], ...]

    def reference_time(self, watch_timestamp: np.ndarray) -> np.ndarray:
        time = np.asarray(watch_timestamp, dtype=np.float64)
        return time + self.offset_seconds + self.drift_seconds_per_second * (
            time - self.center_timestamp
        )


@dataclass
class AlignedReference:
    alignment: Alignment
    training_cutoff: float
    training_count: int
    reference_time: np.ndarray
    reference_hr: np.ndarray
    quality: np.ndarray
    usable: np.ndarray


def estimate_offset(
    watch_timestamp: Sequence[float],
    watch_hr: Sequence[float],
    reference_lookup: Callable[[np.ndarray], np.ndarray],
    max_lag_seconds: float = 30.0,
    coarse_step: float = 0.5,
    fine_step: float = 0.1,
) -> tuple[float, float, float]:
    timestamp = np.asarray(watch_timestamp, dtype=np.float64)
    heart_rate = np.asarray(watch_hr, dtype=np.float64)

    def score(lag: float) -> float:
        return _alignment_score(heart_rate, reference_lookup(timestamp + lag))

    coarse = np.arange(-max_lag_seconds, max_lag_seconds + coarse_step / 2, coarse_step)
    coarse_scores = np.asarray([score(float(lag)) for lag in coarse])
    if not np.any(np.isfinite(coarse_scores)):
        raise ValueError("Nincs elég jó EKG referencia az időillesztéshez.")
    coarse_best = float(coarse[int(np.nanargmin(coarse_scores))])
    fine = np.arange(coarse_best - coarse_step, coarse_best + coarse_step + fine_step / 2, fine_step)
    fine_scores = np.asarray([score(float(lag)) for lag in fine])
    best_index = int(np.nanargmin(fine_scores))
    best = float(np.round(fine[best_index], 6))
    return best, float(fine_scores[best_index]), score(0.0)


def estimate_alignment(
    watch_timestamp: np.ndarray,
    watch_hr: np.ndarray,
    reference: ECGReference,
    max_lag_seconds: float,
) -> Alignment:
    lookup = lambda time: reference.evaluate(time)[0]
    offset, score, zero_score = estimate_offset(
        watch_timestamp, watch_hr, lookup, max_lag_seconds=max_lag_seconds
    )
    center = float(np.median(watch_timestamp))
    aligned_reference, _ = reference.evaluate(watch_timestamp + offset)
    valid_time = watch_timestamp[np.isfinite(aligned_reference)]
    local_offsets: list[tuple[float, float]] = []
    if len(valid_time) >= 1200 and np.ptp(valid_time) >= 3600:
        edges = np.linspace(float(valid_time.min()), float(valid_time.max()), 6)
        for left, right in zip(edges[:-1], edges[1:]):
            mask = (watch_timestamp >= left) & (watch_timestamp <= right)
            if np.count_nonzero(mask) < 180:
                continue
            try:
                local, _, _ = estimate_offset(
                    watch_timestamp[mask],
                    watch_hr[mask],
                    lambda time: lookup(time + offset),
                    max_lag_seconds=3.0,
                    coarse_step=0.5,
                    fine_step=0.2,
                )
                local_offsets.append((float(np.median(watch_timestamp[mask])), offset + local))
            except ValueError:
                continue

    final_offset = offset
    drift = 0.0
    drift_applied = False
    if len(local_offsets) >= 4:
        local_time = np.asarray([item[0] for item in local_offsets])
        local_lag = np.asarray([item[1] for item in local_offsets])
        slope, intercept = np.polyfit(local_time - center, local_lag, 1)
        fitted = intercept + slope * (local_time - center)
        residual_mad = float(np.median(np.abs(local_lag - fitted)))
        total_drift = abs(float(slope * np.ptp(local_time)))
        ss_total = float(np.sum((local_lag - np.mean(local_lag)) ** 2))
        r_squared = 1.0 - float(np.sum((local_lag - fitted) ** 2)) / ss_total if ss_total else 0.0
        candidate_reference, _ = reference.evaluate(
            watch_timestamp + intercept + slope * (watch_timestamp - center)
        )
        candidate_score = _alignment_score(watch_hr, candidate_reference)
        if (
            total_drift >= 1.0
            and residual_mad <= 0.30
            and r_squared >= 0.80
            and candidate_score <= 0.98 * score
        ):
            final_offset = float(intercept)
            drift = float(slope)
            score = candidate_score
            drift_applied = True
    return Alignment(
        offset_seconds=final_offset,
        drift_seconds_per_second=drift,
        center_timestamp=center,
        score=score,
        zero_offset_score=zero_score,
        drift_applied=drift_applied,
        local_offsets=tuple(local_offsets),
    )


def align_ecg_reference(
    watch: WatchData,
    in_overlap: np.ndarray,
    reference: ECGReference,
    train_fraction: float,
    max_lag_seconds: float,
) -> AlignedReference:
    """Fit alignment on the early ECG-gated subset and evaluate all watch rows."""
    nominal_reference, _ = reference.evaluate(watch.timestamp)
    alignment_candidate = (
        in_overlap
        & np.isfinite(nominal_reference)
        & np.isfinite(watch.heart_rate)
        & (watch.heart_rate >= 25)
        & (watch.heart_rate <= 240)
    )
    candidate_index = np.flatnonzero(alignment_candidate)
    if len(candidate_index) < 200:
        raise ValueError("Nincs legalább 200 névlegesen használható pár az időillesztéshez.")
    training_size = max(100, int(len(candidate_index) * train_fraction))
    training_index = candidate_index[:training_size]
    training_cutoff = float(watch.timestamp[training_index[-1]])
    alignment = estimate_alignment(
        watch.timestamp[training_index],
        watch.heart_rate[training_index],
        reference,
        max_lag_seconds,
    )
    reference_time = alignment.reference_time(watch.timestamp)
    reference_hr, quality = reference.evaluate(reference_time)
    reference_hr[~in_overlap] = np.nan
    usable = (
        in_overlap
        & np.isfinite(reference_hr)
        & (watch.heart_rate >= 25)
        & (watch.heart_rate <= 240)
    )
    reference_hr[~usable] = np.nan
    return AlignedReference(
        alignment=alignment,
        training_cutoff=training_cutoff,
        training_count=len(training_index),
        reference_time=reference_time,
        reference_hr=reference_hr,
        quality=quality,
        usable=usable,
    )


def robust_affine_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    design = np.column_stack((x, np.ones(len(x))))
    coefficient = np.linalg.lstsq(design, y, rcond=None)[0]
    for _ in range(30):
        residual = y - design @ coefficient
        scale = 1.4826 * np.median(np.abs(residual - np.median(residual))) + 1e-9
        weight = np.minimum(1.0, 1.345 * scale / np.maximum(np.abs(residual), 1e-9))
        next_coefficient = np.linalg.lstsq(
            design * np.sqrt(weight[:, None]), y * np.sqrt(weight), rcond=None
        )[0]
        if np.max(np.abs(next_coefficient - coefficient)) < 1e-9:
            coefficient = next_coefficient
            break
        coefficient = next_coefficient
    return float(coefficient[0]), float(coefficient[1])


def metrics(prediction: np.ndarray, reference: np.ndarray) -> dict[str, float | int]:
    valid = np.isfinite(prediction) & np.isfinite(reference)
    prediction, reference = prediction[valid], reference[valid]
    if not len(prediction):
        return {key: math.nan for key in ("mae", "rmse", "median_abs", "bias", "correlation", "within5", "within10")} | {"n": 0}
    error = prediction - reference
    correlation = (
        float(np.corrcoef(prediction, reference)[0, 1])
        if len(prediction) >= 2 and np.std(prediction) > 0 and np.std(reference) > 0
        else math.nan
    )
    return {
        "n": int(len(prediction)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "median_abs": float(np.median(np.abs(error))),
        "bias": float(np.mean(error)),
        "correlation": correlation,
        "within5": float(np.mean(np.abs(error) <= 5.0)),
        "within10": float(np.mean(np.abs(error) <= 10.0)),
    }


def block_mae_support(
    timestamp: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
    block_seconds: float = 300.0,
    minimum_samples: int = 60,
) -> tuple[int, float, float]:
    """Return evaluable blocks, candidate win rate, and median MAE gain."""
    valid = (
        np.isfinite(timestamp)
        & np.isfinite(baseline)
        & np.isfinite(candidate)
        & np.isfinite(reference)
    )
    if not np.any(valid):
        return 0, 0.0, math.nan
    time = timestamp[valid]
    baseline = baseline[valid]
    candidate = candidate[valid]
    reference = reference[valid]
    block_id = np.floor((time - float(time.min())) / block_seconds).astype(int)
    gains = []
    for identifier in np.unique(block_id):
        mask = block_id == identifier
        if np.count_nonzero(mask) < minimum_samples:
            continue
        baseline_mae = float(np.mean(np.abs(baseline[mask] - reference[mask])))
        candidate_mae = float(np.mean(np.abs(candidate[mask] - reference[mask])))
        gains.append(baseline_mae - candidate_mae)
    if not gains:
        return 0, 0.0, math.nan
    gain_array = np.asarray(gains)
    return len(gains), float(np.mean(gain_array > 0.0)), float(np.median(gain_array))


@dataclass(frozen=True)
class Calibration:
    name: str
    slope: float
    intercept: float
    split_timestamp: float
    candidates: dict[str, dict[str, float | int]]

    def apply(self, heart_rate: np.ndarray) -> np.ndarray:
        return self.slope * np.asarray(heart_rate, dtype=np.float64) + self.intercept


def select_calibration(
    timestamp: np.ndarray,
    watch_hr: np.ndarray,
    reference_hr: np.ndarray,
    train_fraction: float = 0.70,
    minimum_split_timestamp: float | None = None,
) -> Calibration:
    valid = np.isfinite(timestamp) & np.isfinite(watch_hr) & np.isfinite(reference_hr)
    valid_index = np.flatnonzero(valid)
    if len(valid_index) < 200:
        raise ValueError("Legalább 200 használható pár szükséges a kalibrációhoz.")
    split_position = min(len(valid_index) - 50, max(100, int(len(valid_index) * train_fraction)))
    split_timestamp = float(timestamp[valid_index[split_position]])
    if minimum_split_timestamp is not None:
        minimum_split_timestamp = float(minimum_split_timestamp)
        if not np.isfinite(minimum_split_timestamp):
            raise ValueError("A minimum calibration split timestamp nem véges.")
        split_timestamp = max(
            split_timestamp,
            float(np.nextafter(minimum_split_timestamp, math.inf)),
        )
    train = valid & (timestamp < split_timestamp)
    validation = valid & ~train
    if np.count_nonzero(train) < 100 or np.count_nonzero(validation) < 50:
        raise ValueError("A kronologikus train/validation felosztás túl kicsi.")

    bias = float(np.median(reference_hr[train] - watch_hr[train]))
    affine_slope, affine_intercept = robust_affine_fit(watch_hr[train], reference_hr[train])
    model = {
        "identity": (1.0, 0.0),
        "bias": (1.0, bias),
        "affine": (affine_slope, affine_intercept),
    }
    candidates: dict[str, dict[str, float | int]] = {}
    for name, (slope, intercept) in model.items():
        candidate_prediction = slope * watch_hr[validation] + intercept
        candidate_metrics = metrics(candidate_prediction, reference_hr[validation])
        block_count, block_win_rate, median_block_gain = block_mae_support(
            timestamp[validation],
            watch_hr[validation],
            candidate_prediction,
            reference_hr[validation],
        )
        candidate_metrics["slope"] = slope
        candidate_metrics["intercept"] = intercept
        candidate_metrics["block_count"] = block_count
        candidate_metrics["block_win_rate"] = block_win_rate
        candidate_metrics["median_block_mae_gain"] = median_block_gain
        candidates[name] = candidate_metrics

    identity = candidates["identity"]
    eligible = ["identity"]
    for name in ("bias", "affine"):
        candidate = candidates[name]
        plausible = True
        if name == "affine":
            predicted_bounds = affine_slope * np.asarray([30.0, 220.0]) + affine_intercept
            plausible = (
                0.80 <= affine_slope <= 1.20
                and abs(affine_intercept) <= 20.0
                and predicted_bounds[0] >= 20.0
                and predicted_bounds[1] <= 240.0
            )
        improves = (
            float(candidate["mae"]) <= 0.98 * float(identity["mae"])
            and float(candidate["mae"]) <= float(identity["mae"]) - 0.10
            and float(candidate["rmse"]) <= float(identity["rmse"])
            and int(candidate["block_count"]) >= 5
            and float(candidate["block_win_rate"]) >= 0.70
            and float(candidate["median_block_mae_gain"]) >= 0.05
        )
        if plausible and improves:
            eligible.append(name)
    # Prefer the simpler bias model unless affine adds a material holdout gain
    # over bias itself. A tiny numerical win cannot justify a slope estimate
    # from one recording.
    selected = "bias" if "bias" in eligible else "identity"
    if "affine" in eligible:
        affine_validation = candidates["affine"]
        bias_validation = candidates["bias"]
        affine_block_count, affine_block_win_rate, affine_median_block_gain = block_mae_support(
            timestamp[validation],
            watch_hr[validation] + bias,
            affine_slope * watch_hr[validation] + affine_intercept,
            reference_hr[validation],
        )
        affine_validation["vs_bias_block_count"] = affine_block_count
        affine_validation["vs_bias_block_win_rate"] = affine_block_win_rate
        affine_validation["vs_bias_median_block_mae_gain"] = affine_median_block_gain
        affine_beats_bias = (
            float(affine_validation["mae"]) <= 0.98 * float(bias_validation["mae"])
            and float(affine_validation["mae"]) <= float(bias_validation["mae"]) - 0.10
            and float(affine_validation["rmse"]) <= float(bias_validation["rmse"])
            and affine_block_count >= 5
            and affine_block_win_rate >= 0.70
            and affine_median_block_gain >= 0.05
        )
        if affine_beats_bias:
            selected = "affine"
    slope, intercept = model[selected]
    return Calibration(
        name=selected,
        slope=slope,
        intercept=intercept,
        split_timestamp=split_timestamp,
        candidates=candidates,
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _local_iso(timestamp: float, timezone: ZoneInfo) -> str:
    return datetime.fromtimestamp(float(timestamp), timezone).isoformat(timespec="milliseconds")


def _format_interval(interval: tuple[float, float], timezone: ZoneInfo) -> str:
    return f"{_local_iso(interval[0], timezone)} – {_local_iso(interval[1], timezone)}"


def _format_metrics(row: dict[str, float | int]) -> str:
    if int(row["n"]) == 0:
        return "nincs adat"
    return (
        f"n={int(row['n'])}, MAE={row['mae']:.2f}, RMSE={row['rmse']:.2f}, "
        f"medián |hiba|={row['median_abs']:.2f}, bias={row['bias']:+.2f} bpm, "
        f"r={row['correlation']:.3f}, ±5 bpm={100*row['within5']:.1f}%, "
        f"±10 bpm={100*row['within10']:.1f}%"
    )


POSTURE_LABEL = {1: "fekvő", 2: "álló/ülő nyugalom", 3: "álló + mozgás"}


def read_mox_motion(bdf: "BDFReader") -> dict[str, np.ndarray] | None:
    """Másodpercenkénti mozgás-count és testhelyzet a MOX gyorsulásmérőjéből.

    Az `Intensity` csatorna durva (néhány tucat egész érték), ezért a 25 Hz-es
    X/Y/Z-ből számolunk saját mozgásmértéket: a gravitációval együtt mért
    vektorhossz másodpercenkénti szórását. Ez nulla, ha a szenzor áll, és a
    testhelyzettől független, mert a szórás kiejti az állandó gravitációt.
    A `Classification` csatorna a MOX saját testhelyzet-besorolása.
    """
    try:
        axes = [bdf.find_signal(label) for label in ("X", "Y", "Z")]
    except ValueError:
        return None
    rate = bdf.sample_rate(axes[0])
    if rate < 5 or any(bdf.sample_rate(index) != rate for index in axes):
        return None
    components = [bdf.read_signal(index) for index in axes]
    length = min(len(component) for component in components)
    magnitude = np.sqrt(sum(component[:length].astype(np.float64) ** 2 for component in components))
    per_second = int(round(rate))
    seconds = length // per_second
    if seconds < 2:
        return None
    tiles = magnitude[: seconds * per_second].reshape(seconds, per_second)
    result = {
        "time": bdf.start.timestamp() + np.arange(seconds, dtype=np.float64),
        "count": tiles.std(axis=1),
        "magnitude": tiles.mean(axis=1),
    }
    try:
        posture_index = bdf.find_signal("Classification")
    except ValueError:
        return result
    posture = bdf.read_signal(posture_index)
    posture_rate = bdf.sample_rate(posture_index)
    sample = np.clip(np.rint(np.arange(seconds) * posture_rate).astype(int), 0, len(posture) - 1)
    result["posture"] = np.rint(posture[sample])
    return result


def sample_mox_motion(
    motion: dict[str, np.ndarray] | None, reference_time: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """A másodperces mozgássorból a kért referencia-időkre vett legközelebbi érték."""
    count = np.full(len(reference_time), np.nan)
    posture = np.full(len(reference_time), np.nan)
    if motion is None or not len(motion["time"]):
        return count, posture
    start = motion["time"][0]
    nearest = np.clip(
        np.rint(reference_time - start).astype(int), 0, len(motion["time"]) - 1
    )
    inside = (reference_time >= start - 0.5) & (reference_time <= motion["time"][-1] + 0.5)
    count[inside] = motion["count"][nearest[inside]]
    if "posture" in motion:
        posture[inside] = motion["posture"][nearest[inside]]
    return count, posture


def write_comparison(
    path: Path,
    watch: WatchData,
    timezone: ZoneInfo,
    reference_time: np.ndarray,
    reference_hr: np.ndarray,
    calibrated_hr: np.ndarray,
    quality: np.ndarray,
    in_overlap: np.ndarray,
    split_timestamp: float,
    activity: np.ndarray,
    motion_count: np.ndarray,
    posture: np.ndarray,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "watch_timestamp_raw",
                "watch_time_local",
                "reference_time_local",
                "watch_hr_bpm",
                "ecg_reference_hr_bpm",
                "calibrated_hr_bpm",
                "raw_error_bpm",
                "calibrated_error_bpm",
                "qrs_quality_0_1",
                "usable",
                "split",
                "activity_intensity",
                "mox_motion_count",
                "mox_posture",
            ]
        )
        for index in range(len(watch.timestamp)):
            usable = bool(in_overlap[index] and np.isfinite(reference_hr[index]))
            split = "train" if usable and watch.timestamp[index] < split_timestamp else "validation" if usable else "excluded"
            ref = reference_hr[index]
            writer.writerow(
                [
                    int(watch.raw_timestamp[index]) if watch.raw_timestamp[index].is_integer() else f"{watch.raw_timestamp[index]:.6f}",
                    _local_iso(watch.timestamp[index], timezone),
                    _local_iso(reference_time[index], timezone),
                    f"{watch.heart_rate[index]:.3f}",
                    f"{ref:.3f}" if usable else "",
                    f"{calibrated_hr[index]:.3f}",
                    f"{watch.heart_rate[index] - ref:.3f}" if usable else "",
                    f"{calibrated_hr[index] - ref:.3f}" if usable else "",
                    f"{quality[index]:.4f}",
                    "1" if usable else "0",
                    split,
                    f"{activity[index]:.3f}" if np.isfinite(activity[index]) else "",
                    f"{motion_count[index]:.2f}" if np.isfinite(motion_count[index]) else "",
                    POSTURE_LABEL.get(int(posture[index]), "") if np.isfinite(posture[index]) else "",
                ]
            )


def write_plot(
    path: Path,
    watch_time: np.ndarray,
    watch_hr: np.ndarray,
    reference_hr: np.ndarray,
    calibrated_hr: np.ndarray,
    quality: np.ndarray,
    activity: np.ndarray,
    motion_count: np.ndarray,
    posture: np.ndarray,
    timezone: ZoneInfo,
) -> None:
    valid = np.isfinite(reference_hr)
    local_time = [datetime.fromtimestamp(float(t), timezone) for t in watch_time]
    figure, axes = plt.subplots(4, 1, figsize=(14, 12), constrained_layout=True)
    axes[0].plot(local_time, watch_hr, color="#4C78A8", linewidth=0.7, alpha=0.65, label="Huawei nyers")
    axes[0].plot(local_time, reference_hr, color="#E45756", linewidth=1.0, label="EKG referencia")
    axes[0].plot(
        local_time,
        np.where(valid, calibrated_hr, np.nan),
        color="#54A24B",
        linewidth=0.8,
        alpha=0.8,
        label="Kalibrált",
    )
    axes[0].set_ylabel("Pulzus (bpm)")
    axes[0].set_title("Idősor és minőségkapuzott összehasonlítás")
    axes[0].legend(ncol=3, loc="upper right")

    raw_error = watch_hr[valid] - reference_hr[valid]
    calibrated_error = calibrated_hr[valid] - reference_hr[valid]
    valid_time = np.asarray(local_time, dtype=object)[valid]
    axes[1].scatter(valid_time, raw_error, s=3, alpha=0.25, color="#4C78A8", label="Nyers hiba")
    axes[1].scatter(valid_time, calibrated_error, s=3, alpha=0.25, color="#54A24B", label="Kalibrált hiba")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Hiba (bpm)")
    axes[1].legend(ncol=2, loc="upper right")

    mean_hr = (watch_hr[valid] + reference_hr[valid]) / 2.0
    axes[2].scatter(mean_hr, raw_error, s=5, alpha=0.25, color="#4C78A8", label="Nyers")
    calibrated_mean = (calibrated_hr[valid] + reference_hr[valid]) / 2.0
    axes[2].scatter(calibrated_mean, calibrated_error, s=5, alpha=0.25, color="#54A24B", label="Kalibrált")
    axes[2].axhline(0, color="black", linewidth=0.8)
    axes[2].set_xlabel("Átlagos HR (bpm)")
    axes[2].set_ylabel("Óra − EKG (bpm)")
    axes[2].set_title("Bland–Altman jellegű hibakép")
    axes[2].legend(ncol=2, loc="upper right")

    axes[3].plot(local_time, quality, color="#B279A2", linewidth=0.8, label="QRS minőség")
    axes[3].set_ylabel("QRS minőség (0–1)")
    axes[3].set_ylim(-0.05, 1.05)
    activity_axis = axes[3].twinx()
    if np.any(np.isfinite(motion_count)):
        # A 25 Hz-es gyorsulásmérőből számolt mozgás-count sokkal finomabb
        # felbontású, mint a MOX 1 Hz-es Intensity csatornája.
        activity_axis.plot(
            local_time, motion_count, color="#F58518", linewidth=0.6, alpha=0.55,
            label="MOX mozgás-count",
        )
        activity_axis.set_ylabel("MOX mozgás-count (mg)")
        for value, colour in ((1.0, "#9EC5AB"), (3.0, "#F2B880")):
            marked = np.isfinite(posture) & (posture == value)
            if np.any(marked):
                axes[3].fill_between(
                    local_time, -0.05, 1.05, where=marked, color=colour, alpha=0.18,
                    step="mid", linewidth=0,
                    label="fekvő" if value == 1.0 else "álló + mozgás",
                )
    else:
        activity_axis.plot(local_time, activity, color="#F58518", linewidth=0.6, alpha=0.55, label="Aktivitás")
        activity_axis.set_ylabel("MOX intenzitás")
    axes[3].set_xlabel(f"Helyi idő ({timezone.key})")
    axes[3].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=timezone))
    lines = axes[3].lines + activity_axis.lines
    axes[3].legend(lines, [line.get_label() for line in lines], ncol=2, loc="upper right")
    for axis in (axes[0], axes[1]):
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=timezone))
    figure.savefig(path, dpi=150)
    plt.close(figure)


def select_qrs_diagnostic_intervals(
    reference_time: np.ndarray,
    newly_usable: np.ndarray,
    high_hr: np.ndarray,
    maximum_intervals: int = 2,
) -> list[tuple[float, float, int]]:
    """Select the strongest newly recovered high-HR regions without HR-based SQI."""
    candidate = newly_usable & high_hr & np.isfinite(reference_time)
    timestamps = np.sort(np.asarray(reference_time[candidate], dtype=np.float64))
    if not len(timestamps):
        return []
    # Brief dips below 130 bpm must not fragment one exercise episode.
    intervals, _ = contiguous_intervals(timestamps, gap_threshold=90.0)
    scored = []
    for left, right in intervals:
        count = int(np.count_nonzero((timestamps >= left) & (timestamps <= right)))
        scored.append((left, right, count))
    selected = sorted(scored, key=lambda row: (-row[2], row[0]))[:maximum_intervals]
    return sorted(selected, key=lambda row: row[0])


def write_qrs_diagnostics(
    path: Path,
    ecg: np.ndarray,
    sample_rate: float,
    start_timestamp: float,
    reference: ECGReference,
    intervals: list[tuple[float, float, int]],
    timezone: ZoneInfo,
) -> None:
    """Plot reproducible raw/filtered excerpts and accepted R peaks."""
    if not intervals:
        return
    filtered = _filter_ecg(ecg, sample_rate)
    accepted_peaks = np.unique(
        np.concatenate([block.peaks for block in reference.blocks if block.usable])
    )
    figure, axes = plt.subplots(
        2 * len(intervals),
        1,
        figsize=(16, 4.2 * len(intervals)),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    excerpt_seconds = 15.0
    for interval_index, (left, right, high_count) in enumerate(intervals):
        center = (left + right) / 2.0
        excerpt_left = max(start_timestamp, center - excerpt_seconds / 2.0)
        excerpt_right = min(
            start_timestamp + len(ecg) / sample_rate,
            center + excerpt_seconds / 2.0,
        )
        first = max(0, int(math.floor((excerpt_left - start_timestamp) * sample_rate)))
        last = min(len(ecg), int(math.ceil((excerpt_right - start_timestamp) * sample_rate)))
        sample_time = start_timestamp + np.arange(first, last) / sample_rate
        local_time = [datetime.fromtimestamp(float(value), timezone) for value in sample_time]
        raw_axis = axes[2 * interval_index]
        filtered_axis = axes[2 * interval_index + 1]
        raw_axis.plot(local_time, ecg[first:last], color="#4C78A8", linewidth=0.55)
        filtered_axis.plot(
            local_time, filtered[first:last], color="#E45756", linewidth=0.7
        )
        peak_time = accepted_peaks[
            (accepted_peaks >= sample_time[0]) & (accepted_peaks <= sample_time[-1])
        ]
        peak_index = np.clip(
            np.rint((peak_time - start_timestamp) * sample_rate).astype(int),
            0,
            len(filtered) - 1,
        )
        peak_local_time = [
            datetime.fromtimestamp(float(value), timezone) for value in peak_time
        ]
        filtered_axis.scatter(
            peak_local_time,
            filtered[peak_index],
            s=22,
            facecolors="none",
            edgecolors="#111111",
            linewidths=0.8,
            label=f"elfogadott R-csúcs (n={len(peak_time)})",
            zorder=3,
        )
        episode = (
            f"{datetime.fromtimestamp(left, timezone):%H:%M:%S}–"
            f"{datetime.fromtimestamp(right, timezone):%H:%M:%S}, "
            f"visszaengedett Band ≥130 bpm sorok: {high_count}"
        )
        raw_axis.set_title(f"{interval_index + 1}. magas-HR epizód ({episode}) – nyers EKG")
        raw_axis.set_ylabel("mV")
        filtered_axis.set_title("0,7 Hz HP + 5–30 Hz BP, zero-phase; 15 s reprezentatív részlet")
        filtered_axis.set_ylabel("mV")
        filtered_axis.legend(loc="upper right")
        for axis in (raw_axis, filtered_axis):
            axis.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S", tz=timezone))
            axis.grid(alpha=0.15)
    axes[-1].set_xlabel(f"Helyi idő ({timezone.key})")
    figure.suptitle("QRS-feldolgozás a korábban kizárt magas-HR szakaszokban", fontsize=14)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_report(
    path: Path,
    bdf_path: Path,
    db_path: Path,
    bdf_hash: str,
    db_hash: str,
    bdf: BDFReader,
    ecg_index: int,
    watch: WatchData,
    watch_intervals: list[tuple[float, float]],
    common_intervals: list[tuple[float, float]],
    gap_threshold: float,
    reference: ECGReference,
    alignment: Alignment,
    alignment_train_cutoff: float,
    alignment_train_count: int,
    calibration: Calibration,
    split_metrics: dict[str, dict[str, dict[str, float | int]]],
    full_raw: dict[str, float | int],
    full_calibrated: dict[str, float | int],
    usable_count: int,
    method_comparison: dict[str, object],
) -> None:
    timezone = bdf.timezone
    good_seconds = sum(block.end - block.start for block in reference.blocks if block.usable)
    total_seconds = sum(block.end - block.start for block in reference.blocks)
    common_seconds = sum(right - left for left, right in common_intervals)
    candidate_lines = []
    for name in ("identity", "bias", "affine"):
        row = calibration.candidates[name]
        block_text = (
            f"5 perces blokkok={int(row['block_count'])}, "
            f"block-win={100 * row['block_win_rate']:.1f}%, "
            f"medián blokk-MAE nyereség={row['median_block_mae_gain']:.3f} bpm"
        )
        if name == "affine" and "vs_bias_block_count" in row:
            block_text += (
                f"; bias ellen: blokkok={int(row['vs_bias_block_count'])}, "
                f"block-win={100 * row['vs_bias_block_win_rate']:.1f}%, "
                f"medián nyereség={row['vs_bias_median_block_mae_gain']:.3f} bpm"
            )
        candidate_lines.append(
            f"- `{name}` (a={row['slope']:.6f}, b={row['intercept']:+.3f}): "
            f"{_format_metrics(row)}; {block_text}"
        )
    common_lines = "\n".join(f"- {_format_interval(item, timezone)}" for item in common_intervals)
    local_offsets = (
        ", ".join(
            f"{datetime.fromtimestamp(time, timezone):%H:%M}={lag:+.1f}s"
            for time, lag in alignment.local_offsets
        )
        if alignment.local_offsets
        else "nem volt elegendő külön blokk"
    )
    correction = (
        "identity (nincs támogatott adatalapú korrekció)"
        if calibration.name == "identity"
        else (
            "session-specific, provisional "
            f"{calibration.name}-jelölt: korrigált HR = {calibration.slope:.6f} × "
            f"óra HR {calibration.intercept:+.3f} bpm"
        )
    )
    reason_text = ", ".join(
        f"{reason}={count}"
        for reason, count in sorted(method_comparison["rejection_reasons"].items())
    )
    legacy_metrics = method_comparison["legacy_metrics"]
    common_legacy_metrics = method_comparison["common_legacy_metrics"]
    common_robust_metrics = method_comparison["common_robust_metrics"]
    robust_only_metrics = method_comparison["robust_only_metrics"]
    diagnostic_lines = ", ".join(
        f"{datetime.fromtimestamp(left, timezone):%H:%M:%S}–"
        f"{datetime.fromtimestamp(right, timezone):%H:%M:%S} ({count} magas-HR sor)"
        for left, right, count in method_comparison["diagnostic_intervals"]
    ) or "nem volt újonnan elfogadott magas-HR epizód"
    agreement_change = float(full_raw["mae"]) - float(legacy_metrics["mae"])
    comparison_interpretation = (
        "A lefedettség nőtt, miközben a Band–EKG egyezési metrikák romlottak. "
        "Ez önmagában nem bizonyít EKG-artifactot, mert a kapu nem használ Band-adatot, "
        "de a visszaengedett dinamikus szakaszokat különösen óvatosan kell értelmezni."
        if agreement_change > 0.05
        else "A nagyobb lefedettség nem járt érdemi Band–EKG metrikaromlással."
    )
    text = f"""# Huawei Band 10 – MOX EKG összehasonlítás

## Bemenetek és időtartam

- MOX BDF: `{bdf_path.name}`, SHA-256 `{bdf_hash}`
  - kezdés: {_local_iso(bdf.start.timestamp(), timezone)}
  - vég: {_local_iso(bdf.end_timestamp, timezone)}
  - időtartam: {bdf.duration_seconds / 3600:.3f} óra
  - EKG: `{bdf.signals[ecg_index].label}`, {bdf.sample_rate(ecg_index):.3f} Hz, {bdf.signals[ecg_index].physical_dimension}
- Gadgetbridge input: `{db_path.name}`, SHA-256 `{db_hash}`
  - forrás: {f"ZIP member `{watch.archive_member}`" if watch.archive_member else "közvetlen SQLite-fájl"}
  - tábla: `{watch.table}`, {watch.source_row_count} teljes HR-sor; ebből {len(watch.timestamp)} sor esik a BDF session időtartamába; timestamp egysége: {watch.timestamp_unit}
  - kezdés: {_local_iso(watch.timestamp[0], timezone)}
  - vég: {_local_iso(watch.timestamp[-1], timezone)}
  - {len(watch_intervals)} folytonos óra-intervallum; résküszöb: {gap_threshold:.3f} s

## Közös időszakok

{common_lines}

Névleges közös idő: {common_seconds / 3600:.3f} óra. Az összes közös intervallumot feldolgoztuk; az óra réseit nem hidaltuk át.

## EKG referencia és minőség

Az EKG alapvonalát 0,7 Hz-es, másodrendű zero-phase magasáteresztő távolítja el, ezt 5–30 Hz-es, harmadrendű zero-phase Butterworth sávszűrés követi; így az R-csúcsok időzítése nem tolódik el. A 30 s-os core blokkok detektálása ±5 s lokális paddinggel történik. Az amplitúdó/prominencia-alapú R-csúcsokat 100 ms-os deriváltenergia-detektor ellenőrzi legfeljebb 120 ms eltéréssel, majd a beat-eket lokális medián QRS-template morfológiai korrelációja szűri. A kapu RR-plauzibilitást, legalább 85%-os időlefedettséget, prominence/SNR-t és beatablakon kívüli kontaminációt is vizsgál. Az off-QRS p99 ≤0,55, RMS ≤0,16 és nagy-amplitúdójú mintaarány ≤2% a saját medián QRS-prominenciához normalizált; ez váltja a korábbi teljes blokkos 0,15 mV zajplafont. A 0,05 mV abszolút jel/prominencia-padló továbbra is kizárja a flatline adaptív tévesztését. Az óra nem vett részt a QRS-küszöbök meghatározásában.

- Minőségkapun átment EKG: {good_seconds / 60:.1f} / {total_seconds / 60:.1f} perc ({100 * good_seconds / total_seconds:.1f}%).
- Metrikába került óraminták: {usable_count} / {len(watch.timestamp)} ({100 * usable_count / len(watch.timestamp):.1f}%).
- Elfogadott/elutasított 30 s-os blokkok a teljes BDF-ben: {method_comparison['accepted_blocks']} / {method_comparison['rejected_blocks']}; a névleges közös időben: {method_comparison['common_accepted_blocks']} / {method_comparison['common_rejected_blocks']}.
- Elutasítási okok előfordulása (egy blokk több okot is kaphat): {reason_text}.
- A referencia-HR a használható RR-ek 10 másodperces, középre igazított ablakának mediánja; rossz EKG-blokkon nincs interpoláció.

## Régi és agilis QRS-módszer összehasonlítása

- Régi 60 s / globális zajplafon: {method_comparison['legacy_usable']} pár, {method_comparison['legacy_good_minutes']:.1f} perc elfogadott EKG, offset {method_comparison['legacy_offset']:+.1f} s; {_format_metrics(legacy_metrics)}.
- Új 30 s / template + off-QRS SQI: {usable_count} pár, {method_comparison['robust_good_minutes']:.1f} perc elfogadott EKG, offset {alignment.offset_seconds:+.1f} s; {_format_metrics(full_raw)}.
- Változás: {usable_count - method_comparison['legacy_usable']:+d} pár; MAE {float(full_raw['mae']) - float(legacy_metrics['mae']):+.2f}, RMSE {float(full_raw['rmse']) - float(legacy_metrics['rmse']):+.2f} bpm, ±5 bpm arány {100 * (float(full_raw['within5']) - float(legacy_metrics['within5'])):+.1f} százalékpont.
- Az előző sor eltérő mintahalmazokat hasonlít. Azonos, mindkét módszernél használható {method_comparison['common_count']} páron a régi referencia: {_format_metrics(common_legacy_metrics)}; az új referencia: {_format_metrics(common_robust_metrics)}.
- A kizárólag az új módszerrel elfogadott {method_comparison['robust_only_count']} pár önálló metrikája: {_format_metrics(robust_only_metrics)}. Ez mutatja meg közvetlenül a visszaengedett, nehezebb dinamikus szakasz egyezését.
- Band ≥130 bpm minták: névlegesen {method_comparison['high_total']}, régi párosított {method_comparison['legacy_high']}, új párosított {method_comparison['robust_high']}; kizárólag az új módszerrel visszaengedett {method_comparison['robust_only_high']}, kizárólag a régivel {method_comparison['legacy_only_high']}.
- A magas-HR párokat tartalmazó {method_comparison['high_block_count']} elfogadott EKG-blokk medián bizonyítékai: detektoregyezés {method_comparison['high_agreement']:.3f}, template-konzisztencia {method_comparison['high_morphology']:.3f}, RR-plauzibilitás {method_comparison['high_rr']:.3f}, prominence/SNR {method_comparison['high_snr']:.2f}, off-QRS p99/QRS {method_comparison['high_off_qrs_p99']:.3f}, kontaminált mintaarány {100 * method_comparison['high_contaminated_fraction']:.2f}%.
- A `qrs_diagnostics.png` 15 s-os nyers/szűrt részletet és elfogadott R-csúcsokat mutat a két legerősebb visszaengedett epizódból: {diagnostic_lines}.

{comparison_interpretation}

## Időillesztés

Az előjel definíciója: `referencia-idő = óra timestamp + offset + drift × (timestamp − középidő)`. A becsült offset **{alignment.offset_seconds:+.1f} s**. Ez azt jelenti, hogy a negatív érték az óra adott sorát ennyivel korábbi EKG-időhöz hasonlítja. A robusztus illesztési score 0 s-nál {alignment.zero_offset_score:.3f}, illesztve {alignment.score:.3f} bpm.

- Az alignment kizárólag a névleges, órafüggetlen EKG-minőségkapu első {alignment_train_count} óramintáját használta; cutoff: {_local_iso(alignment_train_cutoff, timezone)}. A későbbi calibration holdout HR-értékei az időillesztés számára vakok maradtak.
- Lokális offset-ellenőrzések: {local_offsets}.
- Drift: {"alkalmazva, " + f"{alignment.drift_seconds_per_second * 3600:+.3f} s/óra" if alignment.drift_applied else "nem alkalmazva; az offset-only modell maradt, mert a szigorú stabilitási és javulási feltételek nem teljesültek"}.

## Kronologikus kalibráció

- Train: a használható párok {_local_iso(calibration.split_timestamp, timezone)} előtti része.
- Validation/holdout: az ettől kezdődő, használható rész; a modellválasztás csak ezen történt.
- Elfogadási szabály: a korrekció legalább 2% és 0,10 bpm validation-MAE javulást adjon romló RMSE nélkül, továbbá legyen legalább öt értékelhető 5 perces blokk, legalább 70% block-win arány és legalább 0,05 bpm medián blokk-MAE nyereség. Az affine modellnek ugyanezzel a küszöbbel a bias modellt is meg kell előznie. Így puszta numerikus zaj vagy autokorrelált 1 Hz-es mintatömeg miatt nem választunk korrekciót.
- Validation jelöltek:
{chr(10).join(candidate_lines)}
- Kiválasztott korrekció: **{correction}**.

Train nyers: {_format_metrics(split_metrics['train']['raw'])}

Train kalibrált: {_format_metrics(split_metrics['train']['calibrated'])}

Validation nyers: {_format_metrics(split_metrics['validation']['raw'])}

Validation kalibrált: {_format_metrics(split_metrics['validation']['calibrated'])}

## Teljes használható szakasz metrikái

- Nyers óra: {_format_metrics(full_raw)}
- Kalibrált óra: {_format_metrics(full_calibrated)}

## Értelmezési korlátok

Ez technikai összehasonlítás, nem orvosi diagnózis. Egyetlen, részben rossz EKG-minőségű felvételből nem következik általános Huawei Band 10 korrekció. A holdout ugyan csökkenti a túlillesztés kockázatát, de ugyanazon mérési napból származik. Ritmuszavar, laza elektróda, mozgás, eltérő szenzorkésés és az óra saját simítása külön hibaforrás lehet. {"A jelen sessionön az identity maradt, tehát nincs támogatott korrekció." if calibration.name == "identity" else "A kiválasztott képlet kizárólag **session-specific, provisional jelölt**; csak hasonló körülmények között, új felvételeken végzett külső validáció után érdemes általános korrekcióként mérlegelni."}

## Következő felvétel gyakorlati menete

1. A MOX és a Band óráját ugyanahhoz a pontos helyi időhöz szinkronizáld; jegyezd fel az időzónát és az esetleges kézi indítási eseményt.
2. Legyen legalább 60–90 perc jó elektródakontaktus, benne nyugalom, séta és fokozatos pulzusváltozás; klinikai panasz esetén kövesd az orvos utasítását.
3. A nyers BDF-et és az eredeti Gadgetbridge SQLite- vagy ZIP-exportot változatlanul, egyedi fájlnéven tartsd meg. Ne másold rá az új felvételt a régire.
4. Futtasd új output könyvtárba ugyanazt a parancsot, majd először az `overview.png` minőség- és idősorábráját ellenőrizd.
5. Általános korrekciót csak több külön napon, külön kronologikus holdoutokon ismételten javuló eredmény alapján fogadj el.
"""
    path.write_text(text, encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    bdf_path = Path(args.bdf).resolve()
    db_path = Path(args.watch_db).resolve()
    output = Path(args.output).resolve()
    if not bdf_path.is_file() or not db_path.is_file():
        raise FileNotFoundError("A megadott BDF vagy SQLite input nem található.")
    output.mkdir(parents=True, exist_ok=True)
    bdf_hash, db_hash = sha256(bdf_path), sha256(db_path)

    bdf = BDFReader(bdf_path, args.timezone)
    ecg_index = bdf.find_signal("ECG")
    ecg = bdf.read_signal(ecg_index)
    sample_rate = bdf.sample_rate(ecg_index)
    reference = detect_qrs(ecg, sample_rate, bdf.start.timestamp())
    legacy_reference = _detect_qrs_legacy(ecg, sample_rate, bdf.start.timestamp())

    source_watch = load_watch_data(db_path)
    session_mask = (
        (source_watch.timestamp >= bdf.start.timestamp())
        & (source_watch.timestamp <= bdf.end_timestamp)
    )
    if not np.any(session_mask):
        raise ValueError("A watch inputban nincs a BDF session időtartamába eső HR-adat.")
    watch = WatchData(
        raw_timestamp=source_watch.raw_timestamp[session_mask],
        timestamp=source_watch.timestamp[session_mask],
        heart_rate=source_watch.heart_rate[session_mask],
        table=source_watch.table,
        timestamp_unit=source_watch.timestamp_unit,
        source_row_count=source_watch.source_row_count,
        archive_member=source_watch.archive_member,
    )
    watch_intervals, gap_threshold = contiguous_intervals(watch.timestamp)
    common_intervals = intersect_intervals(
        watch_intervals, bdf.start.timestamp(), bdf.end_timestamp
    )
    if not common_intervals:
        raise ValueError("A BDF és az óra között nincs névleges közös időszak.")
    in_overlap = np.zeros(len(watch.timestamp), dtype=bool)
    for left, right in common_intervals:
        in_overlap |= (watch.timestamp >= left) & (watch.timestamp <= right)

    robust_evaluation = align_ecg_reference(
        watch,
        in_overlap,
        reference,
        args.train_fraction,
        args.max_lag,
    )
    alignment = robust_evaluation.alignment
    alignment_train_cutoff = robust_evaluation.training_cutoff
    reference_time = robust_evaluation.reference_time
    reference_hr = robust_evaluation.reference_hr
    quality = robust_evaluation.quality
    usable = robust_evaluation.usable

    legacy_evaluation = align_ecg_reference(
        watch,
        in_overlap,
        legacy_reference,
        args.train_fraction,
        args.max_lag,
    )

    calibration = select_calibration(
        watch.timestamp,
        watch.heart_rate,
        reference_hr,
        args.train_fraction,
        minimum_split_timestamp=alignment_train_cutoff,
    )
    calibrated_hr = calibration.apply(watch.heart_rate)
    train = usable & (watch.timestamp < calibration.split_timestamp)
    validation = usable & ~train
    split_metrics = {
        "train": {
            "raw": metrics(watch.heart_rate[train], reference_hr[train]),
            "calibrated": metrics(calibrated_hr[train], reference_hr[train]),
        },
        "validation": {
            "raw": metrics(watch.heart_rate[validation], reference_hr[validation]),
            "calibrated": metrics(calibrated_hr[validation], reference_hr[validation]),
        },
    }
    full_raw = metrics(watch.heart_rate[usable], reference_hr[usable])
    full_calibrated = metrics(calibrated_hr[usable], reference_hr[usable])

    legacy_metrics = metrics(
        watch.heart_rate[legacy_evaluation.usable],
        legacy_evaluation.reference_hr[legacy_evaluation.usable],
    )
    high_hr = in_overlap & (watch.heart_rate >= 130.0)
    robust_high = usable & high_hr
    legacy_high = legacy_evaluation.usable & high_hr
    robust_only_high = robust_high & ~legacy_high
    common_usable = usable & legacy_evaluation.usable
    robust_only_usable = usable & ~legacy_evaluation.usable
    diagnostic_intervals = select_qrs_diagnostic_intervals(
        reference_time,
        robust_only_usable,
        high_hr,
    )
    high_reference_time = np.sort(reference_time[robust_high])
    high_blocks = []
    for block in reference.blocks:
        if not block.usable or not len(high_reference_time):
            continue
        left = np.searchsorted(high_reference_time, block.start, side="left")
        right = np.searchsorted(high_reference_time, block.end, side="left")
        if right > left:
            high_blocks.append(block)

    rejection_reasons: dict[str, int] = {}
    for block in reference.blocks:
        if block.usable:
            continue
        for reason in block.reason.split("+"):
            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
    common_blocks = [
        block
        for block in reference.blocks
        if any(block.end > left and block.start <= right for left, right in common_intervals)
    ]
    method_comparison = {
        "legacy_usable": int(np.count_nonzero(legacy_evaluation.usable)),
        "legacy_metrics": legacy_metrics,
        "common_count": int(np.count_nonzero(common_usable)),
        "common_legacy_metrics": metrics(
            watch.heart_rate[common_usable],
            legacy_evaluation.reference_hr[common_usable],
        ),
        "common_robust_metrics": metrics(
            watch.heart_rate[common_usable], reference_hr[common_usable]
        ),
        "robust_only_count": int(np.count_nonzero(robust_only_usable)),
        "robust_only_metrics": metrics(
            watch.heart_rate[robust_only_usable], reference_hr[robust_only_usable]
        ),
        "legacy_offset": legacy_evaluation.alignment.offset_seconds,
        "legacy_good_minutes": sum(
            block.end - block.start for block in legacy_reference.blocks if block.usable
        )
        / 60.0,
        "robust_good_minutes": sum(
            block.end - block.start for block in reference.blocks if block.usable
        )
        / 60.0,
        "high_total": int(np.count_nonzero(high_hr)),
        "legacy_high": int(np.count_nonzero(legacy_high)),
        "robust_high": int(np.count_nonzero(robust_high)),
        "robust_only_high": int(np.count_nonzero(robust_only_high)),
        "legacy_only_high": int(np.count_nonzero(legacy_high & ~robust_high)),
        "accepted_blocks": sum(block.usable for block in reference.blocks),
        "rejected_blocks": sum(not block.usable for block in reference.blocks),
        "common_accepted_blocks": sum(block.usable for block in common_blocks),
        "common_rejected_blocks": sum(not block.usable for block in common_blocks),
        "rejection_reasons": rejection_reasons,
        "high_block_count": len(high_blocks),
        "high_agreement": float(np.median([block.agreement for block in high_blocks]))
        if high_blocks
        else math.nan,
        "high_morphology": float(
            np.median([block.morphology_consistency for block in high_blocks])
        )
        if high_blocks
        else math.nan,
        "high_rr": float(np.median([block.plausible_rr for block in high_blocks]))
        if high_blocks
        else math.nan,
        "high_snr": float(np.median([block.prominence_snr for block in high_blocks]))
        if high_blocks
        else math.nan,
        "high_off_qrs_p99": float(
            np.median([block.off_qrs_p99_ratio for block in high_blocks])
        )
        if high_blocks
        else math.nan,
        "high_contaminated_fraction": float(
            np.median([block.contaminated_fraction for block in high_blocks])
        )
        if high_blocks
        else math.nan,
        "diagnostic_intervals": diagnostic_intervals,
    }

    activity = np.full(len(watch.timestamp), np.nan)
    try:
        activity_index = bdf.find_signal("Intensity")
        activity_value = bdf.read_signal(activity_index)
        activity_rate = bdf.sample_rate(activity_index)
        activity_time = bdf.start.timestamp() + np.arange(len(activity_value)) / activity_rate
        nearest = np.clip(
            np.rint((reference_time - bdf.start.timestamp()) * activity_rate).astype(int),
            0,
            len(activity_value) - 1,
        )
        inside = (reference_time >= activity_time[0]) & (reference_time <= activity_time[-1])
        activity[inside] = activity_value[nearest[inside]]
    except ValueError:
        pass

    motion = read_mox_motion(bdf)
    motion_count, posture = sample_mox_motion(motion, reference_time)

    write_comparison(
        output / "comparison.csv",
        watch,
        bdf.timezone,
        reference_time,
        reference_hr,
        calibrated_hr,
        quality,
        in_overlap,
        calibration.split_timestamp,
        activity,
        motion_count,
        posture,
    )
    write_plot(
        output / "overview.png",
        watch.timestamp,
        watch.heart_rate,
        reference_hr,
        calibrated_hr,
        quality,
        activity,
        motion_count,
        posture,
        bdf.timezone,
    )
    write_qrs_diagnostics(
        output / "qrs_diagnostics.png",
        ecg,
        sample_rate,
        bdf.start.timestamp(),
        reference,
        diagnostic_intervals,
        bdf.timezone,
    )
    del ecg
    write_report(
        output / "report.md",
        bdf_path,
        db_path,
        bdf_hash,
        db_hash,
        bdf,
        ecg_index,
        watch,
        watch_intervals,
        common_intervals,
        gap_threshold,
        reference,
        alignment,
        alignment_train_cutoff,
        robust_evaluation.training_count,
        calibration,
        split_metrics,
        full_raw,
        full_calibrated,
        int(np.count_nonzero(usable)),
        method_comparison,
    )
    print(f"Kész: {output}")
    print(f"Használható párok: {np.count_nonzero(usable)}")
    print(f"Időoffset: {alignment.offset_seconds:+.1f} s; drift: {'igen' if alignment.drift_applied else 'nem'}")
    print(f"Korrekció: {calibration.name} (a={calibration.slope:.6f}, b={calibration.intercept:+.3f})")
    print(f"Nyers: {_format_metrics(full_raw)}")
    print(f"Kalibrált: {_format_metrics(full_calibrated)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Huawei/Gadgetbridge HR összevetése MOX BDF EKG referenciával."
    )
    parser.add_argument("--bdf", default=DEFAULT_BDF, help="MOX BDF input")
    parser.add_argument("--watch-db", default=DEFAULT_DB, help="Gadgetbridge SQLite input")
    parser.add_argument("--output", default="results", help="Output könyvtár")
    parser.add_argument("--timezone", default="Europe/Budapest", help="A BDF helyi időzónája")
    parser.add_argument("--max-lag", type=float, default=30.0, help="Maximális abszolút offset keresés, s")
    parser.add_argument("--train-fraction", type=float, default=0.70, help="Kronologikus train arány")
    args = parser.parse_args()
    if args.max_lag <= 0 or not 0.5 <= args.train_fraction <= 0.9:
        parser.error("--max-lag legyen pozitív, --train-fraction 0.5 és 0.9 közötti.")
    return args


if __name__ == "__main__":
    run(parse_args())
