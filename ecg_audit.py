#!/usr/bin/env python3
"""Beat-szintű MOX EKG audit ritmusanomáliák hullámforma-szintű ellenőrzésével.

Az `analyze_hr.py` pipeline célja egy megbízható referencia *pulzus*, ezért 10 s-os
mediánnal simít, és a lokális QRS-templatetől eltérő ütéseket eldobja. Egészségügyi
célú átnézéshez ez épp a rossz irány: a klinikailag érdekes ütések definíció szerint
eltérnek a templatetől. Ez a modul ezért külön, inkluzív úton dolgozik:

1. jelállapot-szegmentálás 2 s-os epochokban, a saját QRS-amplitúdóhoz normalizált
   QRS-en kívüli zaj- és alapvonal-metrikákkal;
2. inkluzív beat-detektálás abszolút amplitúdópadlóval, T-hullám-elnyomással,
   morfológiai szűrés nélkül;
3. beat-szintű jellemzők: prominencia, meredekség, QRS-szélesség, lokális
   template-korreláció;
4. RR-alapú anomáliakeresés (szünet, korai ütés, sorozat, brady/tachy, szabálytalan
   szakasz);
5. minden esemény hullámforma-szintű verifikálása szigorú lokális (±2 s) jelminőségi
   bizonyítékkal, és `valódi` / `bizonytalan` / `artifact` verdikt indoklással;
6. a valódinak ítélt korai ütések morfológiai konzisztencia-vizsgálata: a valódi
   ektópia jellemzően monomorf, a műtermék véletlenszerű alakú;
7. minden eseményről EKG-csík PNG, hogy szemmel is ellenőrizhető legyen.

Ez kutatási/technikai eszköz, NEM orvostechnikai eszköz, és a kimenete NEM
diagnózis. Bármely itt jelölt esemény értékelése szakorvos feladata.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, find_peaks, sosfiltfilt

from analyze_hr import BDFReader

# --- jelállapot ---------------------------------------------------------------
# A küszöbök a felvételek tényleges eloszlásából származnak: tiszta epochon
# off_p99/R ~ 0,06–0,12, off_rms/R ~ 0,02–0,04, alapvonal/R ~ 0,25–0,50.
EPOCH_SECONDS = 2.0
DEAD_AMPLITUDE_MV = 0.20      # ez alatt nincs fiziológiás jel (leszakadt elektróda)
MIN_LIVE_SECONDS = 60.0
CLEAN_OFF_P99 = 0.20          # QRS-en kívüli p99 / lokális medián R-prominencia
CLEAN_OFF_RMS = 0.065
CLEAN_BASELINE = 0.90         # QRS-en kívüli alapvonal-kilengés / R-prominencia
AMPLITUDE_INFLATION = 2.5     # a felvétel medián amplitúdójához képest
GROSS_OFF_P99 = 0.45          # e fölött a szakasz durván műtermékes
GROSS_BASELINE = 1.80

# --- detektálás ---------------------------------------------------------------
ABS_PROMINENCE_MV = 0.05
SIGMA_FACTOR = 3.0
MIN_PEAK_DISTANCE_S = 0.20
DOUBLE_DETECTION_S = 0.45     # T-hullám ablak: a T 0,20–0,45 s-mal követi az R-t
DOUBLE_PROMINENCE_RATIO = 0.45
DOUBLE_SLOPE_RATIO = 0.25

# --- anomáliaküszöbök ---------------------------------------------------------
PAUSE_ABSOLUTE_S = 2.0
PAUSE_RELATIVE = 1.75
PREMATURITY = 0.20
ABRUPTNESS = 0.92             # az ektópia előtti RR-nek normálnak kell lennie: a
                              # fokozatos gyorsulás légzési sinus arrhythmia, nem ektópia
RSA_AUTOCORRELATION = 0.30    # az RR-sor lag-1 autokorrelációja; RSA sima, AF véletlenszerű
WIDE_QRS_FACTOR = 1.40        # e fölött kamrai jellegű
MAX_QRS_FACTOR = 3.00         # e fölött már nem QRS, hanem elkent műtermék
MIN_QRS_FACTOR = 0.45         # e alatt tüske, nem QRS
LOW_CORRELATION = 0.70
MIN_COUPLING_S = 0.28         # ennél rövidebb kapcsolási idő a kamrai refrakter
                              # periódus miatt izolált ektópiánál nem lehetséges
TALL_WIDE_PROMINENCE = 1.15   # egyszerre sokkal szélesebb és magasabb: mozgási tranziens
TALL_WIDE_WIDTH = 2.00
MIN_PROMINENCE_RATIO = 0.35
MIN_SLOPE_RATIO = 0.25
NEIGHBOUR_CORRELATION = 0.90  # a környező normál ütések template-egyezése
BRADY_BPM = 45.0
TACHY_REST_BPM = 120.0
EXTREME_TACHY_BPM = 190.0
EPISODE_SECONDS = 10.0
IRREGULAR_WINDOW_S = 30.0
IRREGULAR_RMSSD = 0.15
MONOMORPHIC_CORRELATION = 0.85


# ==============================================================================
# jelfeldolgozás
# ==============================================================================

def bands(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Diagnosztikai (0,5–40 Hz) és detektálási (5–30 Hz) sáv, mindkettő zero-phase."""
    high_passed = sosfiltfilt(butter(2, 0.5, btype="highpass", fs=fs, output="sos"), x)
    wide = sosfiltfilt(butter(3, 40.0, btype="lowpass", fs=fs, output="sos"), high_passed)
    narrow = sosfiltfilt(
        butter(3, (5.0, 30.0), btype="bandpass", fs=fs, output="sos"), high_passed
    )
    return wide, narrow


def rolling_sigma(signal: np.ndarray, fs: float, window: float = 10.0) -> np.ndarray:
    """Csúszó MAD-szigma, `window` másodperces csomópontok között interpolálva."""
    step = int(round(window * fs))
    n = len(signal)
    centers = np.arange(step // 2, n + step, step)
    values = np.empty(len(centers), dtype=np.float64)
    for k, center in enumerate(centers):
        segment = signal[max(0, center - step // 2) : min(n, center + step // 2)]
        values[k] = (
            1.4826 * float(np.median(np.abs(segment - np.median(segment)))) + 1e-12
            if len(segment)
            else 1e-12
        )
    return np.interp(np.arange(n), np.clip(centers, 0, n - 1), values)


def slope_energy(narrow: np.ndarray, fs: float) -> np.ndarray:
    derivative = np.diff(narrow, prepend=narrow[0])
    return uniform_filter1d(
        derivative * derivative, size=max(3, int(round(0.06 * fs))), mode="nearest"
    )


def suppress_double_detection(
    peaks: np.ndarray, prominence: np.ndarray, energy: np.ndarray, fs: float
) -> np.ndarray:
    """Egy QRS-en belüli másodlagos csúcs (T- vagy S-hullám) eldobása.

    Csak akkor dob, ha a szomszéd prominenciában ÉS meredekség-energiában is
    egyértelműen gyengébb. Ez megvédi a valódi korai kamrai ütéseket, amelyek
    amplitúdója és meredeksége jellemzően nem kisebb a normál ütésénél, és a gyors,
    de szabályos ritmust is, ahol az egymást követő ütések hasonló erősségűek.
    """
    keep = np.ones(len(peaks), dtype=bool)
    score = prominence * np.sqrt(np.maximum(energy[peaks], 1e-18))
    i = 0
    while i < len(peaks) - 1:
        j = i + 1
        while j < len(peaks) and not keep[j]:
            j += 1
        if j >= len(peaks):
            break
        if (peaks[j] - peaks[i]) / fs < DOUBLE_DETECTION_S:
            strong, weak = (i, j) if score[i] >= score[j] else (j, i)
            if prominence[weak] < DOUBLE_PROMINENCE_RATIO * prominence[strong] and (
                energy[peaks[weak]] < DOUBLE_SLOPE_RATIO * energy[peaks[strong]]
            ):
                keep[weak] = False
                if weak == i:
                    i = j
                continue
        i = j
    return keep


def live_runs(x: np.ndarray, fs: float) -> list[tuple[int, int]]:
    """Összefüggő szakaszok, ahol egyáltalán van fiziológiás amplitúdójú jel."""
    step = int(round(10.0 * fs))
    usable = (len(x) // step) * step
    if usable < step:
        return []
    tiles = x[:usable].reshape(-1, step)
    amplitude = np.percentile(tiles, 99, axis=1) - np.percentile(tiles, 1, axis=1)
    live = amplitude > DEAD_AMPLITUDE_MV
    edges = np.diff(np.r_[0, live.view(np.int8), 0])
    runs = []
    for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if (end - start) * 10.0 >= MIN_LIVE_SECONDS:
            runs.append((int(start * step), int(min(len(x), end * step))))
    return runs


# ==============================================================================
# beat- és epoch-jellemzők
# ==============================================================================

@dataclass
class Run:
    """Egy összefüggő élő szakasz minden beat- és epoch-szintű jellemzővel."""

    offset: int
    fs: float
    t0: float
    raw: np.ndarray
    wide: np.ndarray
    narrow: np.ndarray
    oriented: np.ndarray
    sigma: np.ndarray
    energy: np.ndarray
    polarity: float
    peaks: np.ndarray
    prominence: np.ndarray
    width: np.ndarray
    correlation: np.ndarray
    clips: np.ndarray
    epoch_class: np.ndarray
    epoch_amplitude: np.ndarray
    epoch_off_p99: np.ndarray
    epoch_off_rms: np.ndarray
    epoch_baseline: np.ndarray
    epoch_reference: np.ndarray

    def beat_time(self) -> np.ndarray:
        return self.t0 + (self.offset + self.peaks) / self.fs

    def epoch_of(self, sample: int) -> int:
        return int(np.clip(sample / self.fs / EPOCH_SECONDS, 0, len(self.epoch_class) - 1))


def qrs_width(wide: np.ndarray, fs: float, peaks: np.ndarray) -> np.ndarray:
    """QRS-szélesség proxy: a meredekség-energia 5–95%-os energiahatárai."""
    derivative = np.diff(wide, prepend=wide[0])
    energy = derivative * derivative
    half = int(round(0.12 * fs))
    out = np.full(len(peaks), np.nan)
    for j, peak in enumerate(peaks):
        first, last = peak - half, peak + half + 1
        if first < 0 or last > len(energy):
            continue
        cumulative = np.cumsum(energy[first:last])
        total = cumulative[-1]
        if total <= 0:
            continue
        lo = int(np.searchsorted(cumulative, 0.05 * total))
        hi = int(np.searchsorted(cumulative, 0.95 * total))
        out[j] = (hi - lo) / fs
    return out


def beat_clips(oriented: np.ndarray, fs: float, peaks: np.ndarray) -> np.ndarray:
    """Normalizált QRS-kivágatok a template-korrelációhoz és a klaszterezéshez."""
    before = max(2, int(round(0.08 * fs)))
    after = max(3, int(round(0.12 * fs)))
    clips = np.full((len(peaks), before + after + 1), np.nan)
    for j, peak in enumerate(peaks):
        if peak - before < 0 or peak + after >= len(oriented):
            continue
        clip = oriented[peak - before : peak + after + 1].astype(np.float64)
        clip = clip - np.median(clip)
        norm = float(np.linalg.norm(clip))
        if norm > 1e-12:
            clips[j] = clip / norm
    return clips


def template_correlation(
    clips: np.ndarray, times: np.ndarray, rr_normal: np.ndarray
) -> np.ndarray:
    """Minden ütés korrelációja a saját környezetének domináns QRS-templatejével.

    A template kizárólag a lokális mediánhoz közeli RR-ű ütésekből épül, hogy a
    korai/eltérő ütések ne rontsák el a viszonyítási alapot.
    """
    correlation = np.full(len(clips), np.nan)
    if not len(times):
        return correlation
    finite = np.isfinite(clips[:, 0])
    for tile_start in np.arange(0, times[-1] + 60.0, 60.0):
        in_tile = (times >= tile_start) & (times < tile_start + 60.0)
        if not np.any(in_tile):
            continue
        pool = (times >= tile_start - 30.0) & (times < tile_start + 90.0)
        source = pool & rr_normal & finite
        if np.count_nonzero(source) < 5:
            source = pool & finite
        if np.count_nonzero(source) < 3:
            continue
        template = np.median(clips[source], axis=0)
        norm = float(np.linalg.norm(template))
        if norm <= 1e-12:
            continue
        target = in_tile & finite
        correlation[target] = clips[target] @ (template / norm)
    return correlation


def epoch_metrics(
    wide: np.ndarray, narrow: np.ndarray, fs: float, peaks: np.ndarray, prominence: np.ndarray
) -> dict[str, np.ndarray]:
    """2 s-os epochonként a saját QRS-amplitúdóhoz normalizált jelminőségi mérőszámok."""
    step = int(round(EPOCH_SECONDS * fs))
    count = max(1, len(narrow) // step)
    exclusion = int(round(0.14 * fs))
    off_mask = np.ones(len(narrow), dtype=bool)
    for peak in peaks:
        off_mask[max(0, peak - exclusion) : min(len(narrow), peak + exclusion + 1)] = False
    residual = np.abs(narrow - np.median(narrow))

    beat_seconds = peaks / fs
    reference = np.full(count, np.nan)
    for k in range(count):
        center = (k + 0.5) * EPOCH_SECONDS
        near = (beat_seconds >= center - 30.0) & (beat_seconds <= center + 30.0)
        if np.count_nonzero(near) >= 5:
            reference[k] = float(np.median(prominence[near]))

    amplitude = np.full(count, np.nan)
    off_p99 = np.full(count, np.inf)
    off_rms = np.full(count, np.inf)
    baseline = np.full(count, np.inf)
    for k in range(count):
        first, last = k * step, min(len(narrow), (k + 1) * step)
        block = wide[first:last]
        amplitude[k] = float(np.percentile(block, 99.5) - np.percentile(block, 0.5))
        mask = off_mask[first:last]
        off = residual[first:last][mask]
        if len(off) < 20 or not np.isfinite(reference[k]) or reference[k] <= 0:
            continue
        off_p99[k] = float(np.percentile(off, 99.0) / reference[k])
        off_rms[k] = float(np.sqrt(np.mean(off**2)) / reference[k])
        off_wide = block[mask]
        baseline[k] = float(
            (np.percentile(off_wide, 99.5) - np.percentile(off_wide, 0.5)) / reference[k]
        )
    return {
        "amplitude": amplitude,
        "off_p99": off_p99,
        "off_rms": off_rms,
        "baseline": baseline,
        "reference": reference,
    }


def classify_epochs(metrics: dict[str, np.ndarray]) -> np.ndarray:
    """0 = elemezhető, 1 = zajos, 2 = jel nélküli.

    A zajos besorolás egy epochnyit terjed mindkét irányba, mert a mozgási
    tranziens a szomszédos epoch detektálását is elrontja.
    """
    amplitude = metrics["amplitude"]
    healthy = amplitude[np.isfinite(amplitude) & (amplitude > DEAD_AMPLITUDE_MV)]
    inflated = AMPLITUDE_INFLATION * float(np.median(healthy)) if len(healthy) else np.inf
    out = np.zeros(len(amplitude), dtype=np.int8)
    dead = ~np.isfinite(amplitude) | (amplitude < DEAD_AMPLITUDE_MV)
    noisy = (
        (metrics["off_p99"] > CLEAN_OFF_P99)
        | (metrics["off_rms"] > CLEAN_OFF_RMS)
        | (metrics["baseline"] > CLEAN_BASELINE)
        | (amplitude > inflated)
    )
    spread = noisy.copy()
    spread[1:] |= noisy[:-1]
    spread[:-1] |= noisy[1:]
    out[spread] = 1
    out[dead] = 2
    return out


def analyse_run(raw: np.ndarray, offset: int, fs: float, t0: float) -> Run | None:
    wide, narrow = bands(raw, fs)
    sigma = rolling_sigma(narrow, fs)
    polarity = (
        1.0 if abs(np.percentile(narrow, 99.9)) >= abs(np.percentile(narrow, 0.1)) else -1.0
    )
    oriented = polarity * narrow
    energy = slope_energy(narrow, fs)
    peaks, properties = find_peaks(
        oriented,
        distance=int(round(MIN_PEAK_DISTANCE_S * fs)),
        prominence=np.maximum(SIGMA_FACTOR * sigma, ABS_PROMINENCE_MV),
        height=0.5 * sigma,
    )
    if len(peaks) < 20:
        return None
    prominence = properties["prominences"]
    keep = suppress_double_detection(peaks, prominence, energy, fs)
    peaks, prominence = peaks[keep], prominence[keep]

    rr = np.diff(peaks) / fs
    plausible = rr[(rr > 0.25) & (rr < 2.5)]
    median_rr = float(np.median(plausible)) if len(plausible) else 0.8
    rr_normal = np.zeros(len(peaks), dtype=bool)
    if len(rr):
        near = np.abs(rr - median_rr) < 0.12 * median_rr
        rr_normal[:-1] |= near
        rr_normal[1:] |= near

    width = qrs_width(wide, fs, peaks)
    clips = beat_clips(oriented, fs, peaks)
    correlation = template_correlation(clips, peaks / fs, rr_normal)
    metrics = epoch_metrics(wide, narrow, fs, peaks, prominence)
    return Run(
        offset=offset,
        fs=fs,
        t0=t0,
        raw=raw,
        wide=wide,
        narrow=narrow,
        oriented=oriented,
        sigma=sigma,
        energy=energy,
        polarity=polarity,
        peaks=peaks,
        prominence=prominence,
        width=width,
        correlation=correlation,
        clips=clips,
        epoch_class=classify_epochs(metrics),
        epoch_amplitude=metrics["amplitude"],
        epoch_off_p99=metrics["off_p99"],
        epoch_off_rms=metrics["off_rms"],
        epoch_baseline=metrics["baseline"],
        epoch_reference=metrics["reference"],
    )


# ==============================================================================
# anomáliakeresés
# ==============================================================================

@dataclass
class Event:
    kind: str
    start: float
    end: float
    beat: int
    run_id: int = 0
    metrics: dict = field(default_factory=dict)
    verdict: str = "bizonytalan"
    reason: str = ""
    evidence: dict = field(default_factory=dict)


def local_median_rr(rr: np.ndarray, half: int = 10) -> np.ndarray:
    """Lokális RR-alapvonal: a környező ±half RR mediánja, önmagát kihagyva."""
    out = np.empty(len(rr))
    for i in range(len(rr)):
        first, last = max(0, i - half), min(len(rr), i + half + 1)
        neighbourhood = np.delete(rr[first:last], i - first)
        neighbourhood = neighbourhood[(neighbourhood > 0.25) & (neighbourhood < 2.5)]
        out[i] = float(np.median(neighbourhood)) if len(neighbourhood) else np.nan
    return out


def find_events(run: Run, run_id: int) -> list[Event]:
    times = run.beat_time()
    rr = np.diff(times)
    if len(rr) < 20:
        return []
    baseline = local_median_rr(rr)
    normal_width = float(np.nanmedian(run.width))
    normal_prominence = float(np.median(run.prominence))
    events: list[Event] = []

    valid = np.isfinite(baseline)
    ratio = np.full(len(rr), np.nan)
    ratio[valid] = rr[valid] / baseline[valid]
    premature = np.zeros(len(rr), dtype=bool)
    premature[valid] = ratio[valid] <= (1.0 - PREMATURITY)
    # Valódi ektópia hirtelen: az előtte lévő ciklus normál hosszú. A fokozatosan
    # rövidülő RR-sor légzési sinus arrhythmia, nem korai ütés.
    settled = np.zeros(len(rr), dtype=bool)
    settled[1:] = ratio[:-1] >= ABRUPTNESS
    settled[0] = True
    gradual = premature & ~settled
    premature &= settled
    pause = np.zeros(len(rr), dtype=bool)
    pause[valid] = (rr[valid] >= PAUSE_ABSOLUTE_S) | (ratio[valid] >= PAUSE_RELATIVE)

    for i in np.flatnonzero(pause):
        events.append(
            Event(
                kind="szünet",
                start=times[i],
                end=times[i + 1],
                beat=int(i),
                run_id=run_id,
                metrics={
                    "rr_s": float(rr[i]),
                    "alapvonal_rr_s": float(baseline[i]),
                    "arány": float(ratio[i]),
                    "megfelelő_hr": float(60.0 / rr[i]),
                },
            )
        )

    chained = premature.copy()
    for i in np.flatnonzero(gradual):
        if i > 0 and chained[i - 1]:
            chained[i] = True
    early = np.flatnonzero(chained)
    grouped: list[list[int]] = []
    for i in early:
        if grouped and i == grouped[-1][-1] + 1:
            grouped[-1].append(int(i))
        else:
            grouped.append([int(i)])
    for group in grouped:
        first, last = group[0], group[-1]
        beat = first + 1
        following = rr[last + 1] if last + 1 < len(rr) else np.nan
        compensatory = (
            (rr[first] + following) / baseline[first] if np.isfinite(following) else np.nan
        )
        width_ratio = float(run.width[beat] / normal_width) if normal_width > 0 else np.nan
        correlation = float(run.correlation[beat])
        wide_qrs = np.isfinite(width_ratio) and WIDE_QRS_FACTOR <= width_ratio <= MAX_QRS_FACTOR
        odd_shape = np.isfinite(correlation) and correlation < LOW_CORRELATION
        if len(group) >= 2:
            kind = "korai_sorozat"
        elif wide_qrs or odd_shape:
            kind = "korai_kamrai_jellegű"
        else:
            kind = "korai_pitvari_jellegű"
        events.append(
            Event(
                kind=kind,
                start=times[beat],
                end=times[min(len(times) - 1, beat + len(group))],
                beat=beat,
                run_id=run_id,
                metrics={
                    "koraiság": float(1.0 - ratio[first]),
                    "rr_s": float(rr[first]),
                    "alapvonal_rr_s": float(baseline[first]),
                    "kompenzációs_arány": float(compensatory)
                    if np.isfinite(compensatory)
                    else math.nan,
                    "szélesség_arány": width_ratio,
                    "template_korreláció": correlation,
                    "prominencia_arány": float(run.prominence[beat] / normal_prominence),
                    "ütésszám": float(len(group)),
                },
            )
        )

    instantaneous = 60.0 / rr
    midpoints = (times[:-1] + times[1:]) / 2.0
    plausible = (rr > 0.25) & (rr < 2.5)
    for kind, mask in (
        ("bradycardia", plausible & (instantaneous < BRADY_BPM)),
        ("tachycardia", plausible & (instantaneous > TACHY_REST_BPM)),
        ("extrém_tachycardia", plausible & (instantaneous > EXTREME_TACHY_BPM)),
    ):
        edges = np.diff(np.r_[0, mask.view(np.int8), 0])
        for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            duration = midpoints[end - 1] - midpoints[start]
            if duration < EPISODE_SECONDS:
                continue
            window = instantaneous[start:end]
            events.append(
                Event(
                    kind=kind,
                    start=midpoints[start],
                    end=midpoints[end - 1],
                    beat=int(start),
                    run_id=run_id,
                    metrics={
                        "hossz_s": float(duration),
                        "medián_hr": float(np.median(window)),
                        "szélsőérték_hr": float(
                            np.min(window) if kind == "bradycardia" else np.max(window)
                        ),
                        "ütésszám": float(end - start),
                    },
                )
            )

    if len(times) > 2:
        step = IRREGULAR_WINDOW_S / 2.0
        for start_time in np.arange(times[0], times[-1] - IRREGULAR_WINDOW_S, step):
            inside = (times[:-1] >= start_time) & (times[:-1] < start_time + IRREGULAR_WINDOW_S)
            count = int(np.count_nonzero(inside))
            if count < 15:
                continue
            block = rr[inside]
            block = block[(block > 0.25) & (block < 2.5)]
            if len(block) < 15:
                continue
            rmssd = float(np.sqrt(np.mean(np.diff(block) ** 2)))
            median = float(np.median(block))
            ectopic = float(np.mean(premature[inside]))
            centred = block - np.mean(block)
            denominator = float(np.sum(centred**2))
            autocorrelation = (
                float(np.sum(centred[:-1] * centred[1:]) / denominator)
                if denominator > 0
                else math.nan
            )
            if rmssd / median >= IRREGULAR_RMSSD and ectopic < 0.10:
                events.append(
                    Event(
                        kind="légzési_sinus_arrhythmia"
                        if np.isfinite(autocorrelation)
                        and autocorrelation >= RSA_AUTOCORRELATION
                        else "szabálytalan_szakasz",
                        start=float(start_time),
                        end=float(start_time + IRREGULAR_WINDOW_S),
                        beat=int(np.flatnonzero(inside)[0]),
                        run_id=run_id,
                        metrics={
                            "rmssd_s": rmssd,
                            "rmssd_arány": rmssd / median,
                            "medián_rr_s": median,
                            "korai_arány": ectopic,
                            "rr_autokorreláció": autocorrelation,
                            "ütésszám": float(count),
                        },
                    )
                )

    events.sort(key=lambda e: e.start)
    return merge_overlapping(events)


def merge_overlapping(events: list[Event]) -> list[Event]:
    """Az azonos típusú, átfedő ablakos eseményeket (szabálytalan, brady/tachy) összevonja."""
    merged: list[Event] = []
    windowed = {"szabálytalan_szakasz", "légzési_sinus_arrhythmia", "bradycardia", "tachycardia", "extrém_tachycardia"}
    for event in events:
        if (
            merged
            and event.kind in windowed
            and merged[-1].kind == event.kind
            and event.start <= merged[-1].end + 1.0
        ):
            previous = merged[-1]
            previous.end = max(previous.end, event.end)
            previous.metrics["hossz_s"] = previous.end - previous.start
            for key in ("medián_hr", "rmssd_arány"):
                if key in event.metrics and key in previous.metrics:
                    previous.metrics[key] = float(
                        np.mean([previous.metrics[key], event.metrics[key]])
                    )
            if "szélsőérték_hr" in event.metrics:
                low = previous.kind == "bradycardia"
                previous.metrics["szélsőérték_hr"] = float(
                    min(previous.metrics["szélsőérték_hr"], event.metrics["szélsőérték_hr"])
                    if low
                    else max(previous.metrics["szélsőérték_hr"], event.metrics["szélsőérték_hr"])
                )
            continue
        merged.append(event)
    return merged


# ==============================================================================
# hullámforma-szintű verifikáció
# ==============================================================================

def _window(run: Run, event: Event, pad: float = 2.0) -> tuple[int, int]:
    fs = run.fs
    first = int(round((event.start - run.t0) * fs)) - run.offset - int(round(pad * fs))
    last = int(round((event.end - run.t0) * fs)) - run.offset + int(round(pad * fs))
    return max(0, first), min(len(run.wide), last)


def local_signal_quality(run: Run, first: int, last: int) -> dict[str, float]:
    """Az esemény szűk környezetének jelminősége a legrosszabb epoch szerint."""
    step = int(round(EPOCH_SECONDS * run.fs))
    lo = int(np.clip(first // step, 0, len(run.epoch_class) - 1))
    hi = int(np.clip(last // step + 1, lo + 1, len(run.epoch_class)))
    classes = run.epoch_class[lo:hi]
    return {
        "elemezhető_epoch_arány": float(np.mean(classes == 0)),
        "jel_nélküli_epoch_arány": float(np.mean(classes == 2)),
        "off_qrs_p99": float(np.nanmax(run.epoch_off_p99[lo:hi])),
        "off_qrs_rms": float(np.nanmax(run.epoch_off_rms[lo:hi])),
        "alapvonal_kilengés": float(np.nanmax(run.epoch_baseline[lo:hi])),
        "amplitúdó_mv": float(np.nanmax(run.epoch_amplitude[lo:hi])),
    }


def neighbour_consistency(run: Run, beat: int, span: int = 4) -> float:
    """A környező ütések template-egyezésének mediánja, az esemény ütését kihagyva."""
    first, last = max(0, beat - span), min(len(run.correlation), beat + span + 1)
    values = np.delete(run.correlation[first:last], beat - first)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if len(values) else math.nan


def rescue_search(run: Run, first: int, last: int, factor: float = 1.2) -> tuple[np.ndarray, np.ndarray]:
    """A résben lényegesen alacsonyabb küszöbbel keres elmaradt R-hullámot."""
    if last - first < int(round(0.25 * run.fs)):
        return np.array([], dtype=int), np.array([])
    segment = run.oriented[first:last]
    threshold = np.maximum(factor * run.sigma[first:last], 0.02)
    peaks, properties = find_peaks(
        segment, distance=int(round(MIN_PEAK_DISTANCE_S * run.fs)), prominence=threshold
    )
    return peaks + first, properties["prominences"]


def verify(run: Run, event: Event) -> None:
    """Verdikt kizárólag mérhető hullámforma-bizonyíték alapján."""
    fs = run.fs
    first, last = _window(run, event)
    quality = local_signal_quality(run, first, last)
    neighbours = neighbour_consistency(run, event.beat)
    quality["szomszéd_korreláció"] = neighbours
    event.evidence = dict(quality)

    gross = (
        quality["off_qrs_p99"] > GROSS_OFF_P99
        or quality["alapvonal_kilengés"] > GROSS_BASELINE
        or quality["jel_nélküli_epoch_arány"] > 0.0
        or (np.isfinite(neighbours) and neighbours < 0.70)
    )
    mild = (
        quality["elemezhető_epoch_arány"] < 1.0
        or quality["off_qrs_p99"] > CLEAN_OFF_P99
        or quality["off_qrs_rms"] > CLEAN_OFF_RMS
        or quality["alapvonal_kilengés"] > CLEAN_BASELINE
        or (np.isfinite(neighbours) and neighbours < NEIGHBOUR_CORRELATION)
    )

    normal_prominence = float(np.median(run.prominence))
    normal_energy = float(np.median(run.energy[run.peaks]))
    normal_width = float(np.nanmedian(run.width))

    if event.kind == "szünet":
        gap_first = run.peaks[event.beat] + int(round(0.25 * fs))
        gap_last = run.peaks[event.beat + 1] - int(round(0.25 * fs))
        recovered, recovered_prominence = rescue_search(run, gap_first, gap_last)
        strong = int(np.count_nonzero(recovered_prominence >= 0.20 * normal_prominence))
        gap = run.wide[gap_first:gap_last] if gap_last > gap_first else np.array([])
        gap_amplitude = (
            float(np.percentile(gap, 99.5) - np.percentile(gap, 0.5)) if len(gap) > 10 else math.nan
        )
        event.evidence["résben_talált_ütés"] = float(strong)
        event.evidence["rés_amplitúdó_mv"] = gap_amplitude
        if strong:
            event.verdict = "artifact"
            event.reason = (
                f"a résben alacsonyabb küszöbbel {strong} R-hullám kimutatható: "
                "detektálási kiesés, nem valódi szünet"
            )
        elif np.isfinite(gap_amplitude) and gap_amplitude < DEAD_AMPLITUDE_MV:
            event.verdict = "artifact"
            event.reason = "a rés alatt a jel amplitúdója fiziológiátlanul kicsi: megszűnt elektróda-kontaktus"
        elif gross:
            event.verdict = "artifact"
            event.reason = "a rés durván zajos vagy mozgásterhelt szakaszra esik, a szünet nem bizonyítható"
        elif mild:
            event.verdict = "bizonytalan"
            event.reason = "részben zajos környezet: a szünet sem nem igazolható, sem nem zárható ki"
        else:
            event.verdict = "valódi"
            event.reason = "tiszta jel, a rés előtt és után normál ütés, a résben nincs kimutatható R-hullám"

    elif event.kind.startswith("korai"):
        beat = event.beat
        prominence_ratio = (
            float(run.prominence[beat] / normal_prominence) if normal_prominence > 0 else math.nan
        )
        slope_ratio = (
            float(run.energy[run.peaks[beat]] / normal_energy) if normal_energy > 0 else math.nan
        )
        width_ratio = float(run.width[beat] / normal_width) if normal_width > 0 else math.nan
        event.evidence.update(
            {
                "prominencia_arány": prominence_ratio,
                "meredekség_arány": slope_ratio,
                "szélesség_arány": width_ratio,
                "template_korreláció": float(run.correlation[beat]),
            }
        )
        coupling = float(event.metrics.get("rr_s", math.nan))
        event.evidence["kapcsolási_idő_s"] = coupling
        if np.isfinite(coupling) and coupling < MIN_COUPLING_S:
            event.verdict = "artifact"
            event.reason = (
                f"a kapcsolási idő {coupling:.3f} s, ami a kamrai refrakter periódus alatt van: "
                "fiziológiailag nem lehet ütés"
            )
        elif np.isfinite(width_ratio) and width_ratio > MAX_QRS_FACTOR:
            event.verdict = "artifact"
            event.reason = (
                f"a kitérés {width_ratio:.1f}× szélesebb a normál QRS-nél: elkent mozgási műtermék, nem ütés"
            )
        elif np.isfinite(width_ratio) and width_ratio < MIN_QRS_FACTOR:
            event.verdict = "artifact"
            event.reason = "tüskeszerű, fiziológiátlanul keskeny kitérés, nem QRS"
        elif prominence_ratio < MIN_PROMINENCE_RATIO and slope_ratio < MIN_SLOPE_RATIO:
            event.verdict = "artifact"
            event.reason = "az amplitúdó és a meredekség is töredéke a normálénak: T-hullám túlérzékelés"
        elif gross:
            event.verdict = "artifact"
            event.reason = "durván zajos vagy mozgásterhelt szakasz, a korai ütés műterméknek felel meg"
        elif (
            np.isfinite(width_ratio)
            and width_ratio > TALL_WIDE_WIDTH
            and prominence_ratio > TALL_WIDE_PROMINENCE
        ):
            event.verdict = "bizonytalan"
            event.reason = (
                "egyszerre lényegesen szélesebb és magasabb a normál QRS-nél; egyelvezetéses "
                "felvételen ez mozgási tranziens is lehet"
            )
        elif mild:
            event.verdict = "bizonytalan"
            event.reason = "részben zajos környezet, a korai ütés valódisága nem dönthető el"
        else:
            event.verdict = "valódi"
            event.reason = "tiszta jel, fiziológiás amplitúdójú, meredekségű és szélességű QRS a normál ciklus előtt"

    else:
        if gross:
            event.verdict = "artifact"
            event.reason = "az epizód döntő része zajos vagy jel nélküli szakaszra esik"
        elif mild:
            event.verdict = "bizonytalan"
            event.reason = "az epizód részben zajos szakaszra esik"
        else:
            event.verdict = "valódi"
            event.reason = "az epizód végig elemezhető minőségű jelen alapul"
        if event.kind in ("szabálytalan_szakasz", "légzési_sinus_arrhythmia") and event.verdict == "valódi":
            in_window = (run.peaks >= first) & (run.peaks < last)
            correlations = run.correlation[in_window]
            correlations = correlations[np.isfinite(correlations)]
            poor = float(np.mean(correlations < LOW_CORRELATION)) if len(correlations) else math.nan
            event.evidence["gyenge_morfológia_arány"] = poor
            if np.isfinite(poor) and poor > 0.15:
                event.verdict = "bizonytalan"
                event.reason = (
                    "a szakasz ütéseinek jelentős része eltérő morfológiájú, detektálási hiba nem zárható ki"
                )


def morphology_clusters(
    runs: list[Run], events: list[Event], threshold: float = MONOMORPHIC_CORRELATION
) -> list[list[int]]:
    """A valódinak ítélt korai ütések alakcsoportjai.

    Valódi ektópia jellemzően monomorf vagy néhány jól elkülönülő alakot mutat;
    a véletlen műtermékek nem korrelálnak egymással.
    """
    selected = [
        i
        for i, event in enumerate(events)
        if event.verdict == "valódi" and event.kind.startswith("korai")
    ]
    clips = []
    keep = []
    for i in selected:
        event = events[i]
        clip = runs[event.run_id].clips[event.beat]
        if np.all(np.isfinite(clip)):
            clips.append(clip)
            keep.append(i)
    if len(clips) < 2:
        return [[i] for i in keep]
    matrix = np.vstack(clips)
    similarity = matrix @ matrix.T
    remaining = set(range(len(keep)))
    clusters: list[list[int]] = []
    while remaining:
        index = np.array(sorted(remaining))
        counts = (similarity[np.ix_(index, index)] >= threshold).sum(axis=1)
        seed = index[int(np.argmax(counts))]
        members = [j for j in index if similarity[seed, j] >= threshold]
        clusters.append([keep[j] for j in members])
        remaining -= set(members)
    clusters.sort(key=len, reverse=True)
    return clusters


# ==============================================================================
# ábrák
# ==============================================================================

VERDICT_COLOR = {"valódi": "#B3261E", "bizonytalan": "#B8860B", "artifact": "#5B6770"}


def _local(timestamp: float, timezone: ZoneInfo) -> str:
    return datetime.fromtimestamp(timestamp, timezone).strftime("%H:%M:%S")


def render_strip(run: Run, event: Event, path: Path, timezone: ZoneInfo, pad: float = 4.0) -> None:
    fs = run.fs
    first, last = _window(run, event, pad)
    if last - first < int(fs):
        return
    seconds = np.arange(first, last) / fs + run.offset / fs + run.t0
    clock = np.array([datetime.fromtimestamp(t, timezone) for t in seconds])
    figure, axes = plt.subplots(2, 1, figsize=(16, 6.6), sharex=True)
    axes[0].plot(clock, run.raw[first:last], lw=0.7, color="0.35")
    axes[0].set_ylabel("nyers EKG (mV)")
    axes[1].plot(clock, run.wide[first:last], lw=0.9, color="black")
    axes[1].set_ylabel("0,5–40 Hz (mV)")
    inside = np.flatnonzero((run.peaks >= first) & (run.peaks < last))
    beat_time = run.beat_time()
    for j in inside:
        moment = datetime.fromtimestamp(beat_time[j], timezone)
        colour = "#1F6FB2" if j != event.beat else VERDICT_COLOR.get(event.verdict, "red")
        axes[1].plot([moment], [run.wide[run.peaks[j]]], "v", ms=7, color=colour)
        if j + 1 < len(beat_time) and j + 1 in inside:
            axes[1].annotate(
                f"{beat_time[j + 1] - beat_time[j]:.2f}",
                (moment, run.wide[run.peaks[j]]),
                textcoords="offset points",
                xytext=(3, 9),
                fontsize=7.5,
                color="#1F6FB2",
            )
    for axis in axes:
        axis.axvspan(
            datetime.fromtimestamp(event.start, timezone),
            datetime.fromtimestamp(event.end, timezone),
            color=VERDICT_COLOR.get(event.verdict, "red"),
            alpha=0.10,
        )
        axis.grid(alpha=0.3)
    detail = "  ".join(f"{k}={v:.3g}" for k, v in event.metrics.items() if isinstance(v, float))
    evidence = "  ".join(
        f"{k}={v:.3g}"
        for k, v in event.evidence.items()
        if k in ("off_qrs_p99", "alapvonal_kilengés", "szomszéd_korreláció", "szélesség_arány")
    )
    axes[0].set_title(
        f"{event.kind}  ·  {_local(event.start, timezone)}  ·  verdikt: {event.verdict}\n"
        f"{detail}\n{evidence}\n{event.reason}",
        fontsize=9.5,
    )
    axes[1].set_xlabel("helyi idő")
    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)


def render_overview(runs: list[Run], events: list[Event], path: Path, timezone: ZoneInfo) -> None:
    figure, axes = plt.subplots(3, 1, figsize=(19, 11))
    for run in runs:
        base = run.t0 + run.offset / run.fs
        clock = [
            datetime.fromtimestamp(base + k * EPOCH_SECONDS, timezone)
            for k in range(len(run.epoch_class))
        ]
        axes[0].scatter(
            clock, run.epoch_class, s=3, c=run.epoch_class, cmap="RdYlGn_r", vmin=0, vmax=2
        )
        beat_time = run.beat_time()
        rr = np.diff(beat_time)
        ok = (rr > 0.2) & (rr < 3.0)
        midpoints = ((beat_time[:-1] + beat_time[1:]) / 2)[ok]
        axes[1].plot(
            [datetime.fromtimestamp(t, timezone) for t in midpoints], rr[ok], ".", ms=1.2, color="0.25"
        )
        both = ok[:-1] & ok[1:]
        axes[2].plot(rr[:-1][both], rr[1:][both], ".", ms=1.0, alpha=0.35, color="0.3")
    for event in events:
        if event.verdict == "artifact":
            continue
        axes[1].axvline(
            datetime.fromtimestamp(event.start, timezone),
            color=VERDICT_COLOR.get(event.verdict, "red"),
            alpha=0.55,
            lw=0.9,
        )
    axes[0].set_yticks([0, 1, 2])
    axes[0].set_yticklabels(["elemezhető", "zajos", "jel nélkül"])
    axes[0].set_title("Jelállapot 2 s-os epochonként")
    axes[1].set_ylabel("RR (s)")
    axes[1].set_title(
        "RR-tachogram; a függőleges vonalak a valódinak (piros) és bizonytalannak (sárga) ítélt események"
    )
    axes[1].set_ylim(0.2, 2.2)
    axes[2].set_xlabel("RR(n) (s)")
    axes[2].set_ylabel("RR(n+1) (s)")
    axes[2].set_title("Poincaré-diagram")
    axes[2].set_xlim(0.2, 1.6)
    axes[2].set_ylim(0.2, 1.6)
    for axis in axes:
        axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def render_morphology(
    runs: list[Run], events: list[Event], clusters: list[list[int]], path: Path
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(19, 5.4))
    width = np.concatenate([r.width for r in runs])
    correlation = np.concatenate([r.correlation for r in runs])
    ok = np.isfinite(width) & np.isfinite(correlation)
    axes[0].plot(width[ok] * 1000, correlation[ok], ".", ms=1.0, alpha=0.20, color="0.4")
    for event in events:
        if event.verdict != "valódi" or not event.kind.startswith("korai"):
            continue
        run = runs[event.run_id]
        axes[0].plot(
            run.width[event.beat] * 1000, run.correlation[event.beat], "o", ms=4,
            color="#B3261E", alpha=0.8,
        )
    axes[0].set_xlabel("QRS meredekség-szélesség (ms)")
    axes[0].set_ylabel("lokális template-korreláció")
    axes[0].set_title("Minden ütés (szürke) és a valódinak ítélt korai ütések (piros)")
    axes[0].set_xlim(0, 160)

    normal = []
    for run in runs:
        good = np.isfinite(run.clips[:, 0]) & (run.correlation > 0.95)
        if np.any(good):
            normal.append(np.median(run.clips[good], axis=0))
    if normal:
        template = np.median(np.vstack(normal), axis=0)
        axes[1].plot(template, color="black", lw=2, label="normál template")
    for rank, cluster in enumerate(clusters[:4]):
        stack = np.vstack([runs[events[i].run_id].clips[events[i].beat] for i in cluster])
        axes[1].plot(np.median(stack, axis=0), lw=1.6, label=f"{rank+1}. alakcsoport (n={len(cluster)})")
    axes[1].set_title("Korai ütések átlagos alakja alakcsoportonként")
    axes[1].set_xlabel("minta (0,08 s a csúcs előtt … 0,12 s után)")
    axes[1].legend(fontsize=8)

    sizes = [len(c) for c in clusters[:12]]
    if sizes:
        axes[2].bar(range(1, len(sizes) + 1), sizes, color="#B3261E", alpha=0.8)
    axes[2].set_xlabel("alakcsoport sorszáma")
    axes[2].set_ylabel("valódinak ítélt korai ütés")
    axes[2].set_title(
        "Monomorf ektópia egy nagy csoportot adna;\na csupa 1-es hisztogram detektálási zajra utal"
    )
    for axis in axes:
        axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


# ==============================================================================
# futtatás és riport
# ==============================================================================

def beat_epoch_class(run: Run) -> np.ndarray:
    index = np.clip(
        (run.peaks / run.fs / EPOCH_SECONDS).astype(int), 0, len(run.epoch_class) - 1
    )
    return run.epoch_class[index]


def audit(bdf: Path, timezone_name: str, output: Path, strip_limit: int) -> dict:
    timezone = ZoneInfo(timezone_name)
    reader = BDFReader(bdf, timezone_name)
    index = reader.find_signal("ECG")
    fs = reader.sample_rate(index)
    signal = reader.read_signal(index)
    t0 = reader.start.timestamp()

    runs: list[Run] = []
    for first, last in live_runs(signal, fs):
        run = analyse_run(signal[first:last], first, fs, t0)
        if run is not None:
            runs.append(run)
    if not runs:
        raise SystemExit(f"Nincs elemezhető élő EKG-szakasz: {bdf}")

    events: list[Event] = []
    for run_id, run in enumerate(runs):
        found = find_events(run, run_id)
        for event in found:
            verify(run, event)
        events.extend(found)
    events.sort(key=lambda e: e.start)
    clusters = morphology_clusters(runs, events)

    output.mkdir(parents=True, exist_ok=True)
    strips = output / "strips"
    strips.mkdir(exist_ok=True)
    for existing in strips.glob("*.png"):
        existing.unlink()

    with (output / "beats.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "beat_time_local", "run", "rr_prev_s", "prominence_mv", "qrs_width_s",
                "template_correlation", "epoch_class",
            ]
        )
        for run_id, run in enumerate(runs):
            times = run.beat_time()
            rr = np.r_[np.nan, np.diff(times)]
            classes = beat_epoch_class(run)
            for j in range(len(times)):
                writer.writerow(
                    [
                        datetime.fromtimestamp(times[j], timezone).isoformat(timespec="milliseconds"),
                        run_id,
                        f"{rr[j]:.4f}" if np.isfinite(rr[j]) else "",
                        f"{run.prominence[j]:.4f}",
                        f"{run.width[j]:.4f}" if np.isfinite(run.width[j]) else "",
                        f"{run.correlation[j]:.4f}" if np.isfinite(run.correlation[j]) else "",
                        int(classes[j]),
                    ]
                )

    priority = {"valódi": 0, "bizonytalan": 1, "artifact": 2}
    ordered = sorted(
        enumerate(events), key=lambda pair: (priority.get(pair[1].verdict, 3), pair[1].start)
    )
    drawn = {position for position, _ in ordered[:strip_limit]}
    seen: dict[tuple[str, str], int] = {}
    for position, event in enumerate(events):
        key = (event.kind, event.verdict)
        if seen.get(key, 0) < 3:
            seen[key] = seen.get(key, 0) + 1
            drawn.add(position)

    with (output / "events.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["id", "kind", "start_local", "end_local", "verdict", "reason", "metrics", "evidence", "strip"]
        )
        for position, event in enumerate(events):
            name = ""
            if position in drawn:
                name = f"{position:04d}_{event.kind}_{event.verdict}.png"
                render_strip(runs[event.run_id], event, strips / name, timezone)
            writer.writerow(
                [
                    position,
                    event.kind,
                    datetime.fromtimestamp(event.start, timezone).isoformat(timespec="milliseconds"),
                    datetime.fromtimestamp(event.end, timezone).isoformat(timespec="milliseconds"),
                    event.verdict,
                    event.reason,
                    "; ".join(f"{k}={v:.4g}" for k, v in event.metrics.items()),
                    "; ".join(f"{k}={v:.4g}" for k, v in event.evidence.items()),
                    name,
                ]
            )

    render_overview(runs, events, output / "audit_overview.png", timezone)
    render_morphology(runs, events, clusters, output / "audit_morphology.png")

    classes = np.concatenate([r.epoch_class for r in runs])
    clean_rr = []
    for run in runs:
        times = run.beat_time()
        klass = beat_epoch_class(run)
        rr = np.diff(times)
        good = (klass[:-1] == 0) & (klass[1:] == 0) & (rr > 0.25) & (rr < 2.5)
        clean_rr.append(rr[good])
    clean = np.concatenate(clean_rr) if clean_rr else np.array([])

    return {
        "reader": reader,
        "fs": fs,
        "runs": runs,
        "events": events,
        "clusters": clusters,
        "timezone": timezone,
        "duration_hours": len(signal) / fs / 3600.0,
        "live_minutes": sum(len(r.wide) for r in runs) / fs / 60.0,
        "analysable_minutes": float(np.count_nonzero(classes == 0) * EPOCH_SECONDS / 60.0),
        "noisy_minutes": float(np.count_nonzero(classes == 1) * EPOCH_SECONDS / 60.0),
        "dead_minutes": float(np.count_nonzero(classes == 2) * EPOCH_SECONDS / 60.0),
        "beats": int(sum(len(r.peaks) for r in runs)),
        "clean_rr": clean,
        "hr": 60.0 / clean if len(clean) else np.array([]),
        "output": output,
        "strips": len(drawn),
    }


def write_report(name: str, bdf: Path, result: dict) -> None:
    timezone = result["timezone"]
    events: list[Event] = result["events"]
    clusters: list[list[int]] = result["clusters"]
    hr = result["hr"]
    lines: list[str] = []
    add = lines.append

    add(f"# MOX EKG beat-szintű audit – {name}")
    add("")
    add("> Ez technikai átvizsgálás, **nem orvosi diagnózis**. Az automatikus verdiktek a")
    add("> hullámforma mérhető tulajdonságain alapulnak, nem klinikai megítélésen. Minden")
    add("> itt jelölt esemény értékelése szakorvos feladata.")
    add("")
    add("## Mit vizsgál ez az audit, és miben tér el a HR-pipeline-tól")
    add("")
    add("Az `analyze_hr.py` célja megbízható referencia *pulzus*, ezért 10 s-os mediánnal simít,")
    add("és a lokális QRS-templatetől eltérő ütéseket eldobja. Egészségügyi átnézéshez ez épp a")
    add("fordítottja kell: a klinikailag érdekes ütések definíció szerint eltérnek a templatetől.")
    add("Ez az audit ezért inkluzív detektálást használ morfológiai szűrés nélkül, és minden")
    add("RR-anomáliát egyenként, a nyers hullámformán ellenőriz.")
    add("")
    add("## Bemenet és jelállapot")
    add("")
    add(f"- BDF: `{bdf.name}`, {result['fs']:.0f} Hz, teljes hossz {result['duration_hours']:.2f} óra")
    add(f"- Fiziológiás amplitúdójú (élő) szakasz: {result['live_minutes']:.1f} perc, {len(result['runs'])} futamban")
    add(f"- Ebből 2 s-os epochok szerint **elemezhető: {result['analysable_minutes']:.1f} perc**, "
        f"zajos: {result['noisy_minutes']:.1f} perc, jel nélküli: {result['dead_minutes']:.1f} perc")
    add(f"- Detektált ütés: {result['beats']}, ebből tiszta epochban {len(result['clean_rr'])} értékelhető RR-intervallum")
    add("")
    add("A „zajos” minősítés nem hibás mérést jelent, hanem azt, hogy azon a szakaszon egy")
    add("ritmuseltérés jelenléte vagy hiánya **nem bizonyítható**. Az itteni terhelési adatok")
    add("ezért kizárólag az elemezhető időre vonatkoznak.")
    add("")
    if len(hr):
        add("## Ritmus alapadatok az elemezhető szakaszokon")
        add("")
        add(f"- Pulzus: min {np.min(hr):.0f}, p5 {np.percentile(hr, 5):.0f}, medián {np.median(hr):.0f}, "
            f"p95 {np.percentile(hr, 95):.0f}, max {np.max(hr):.0f} bpm")
        add(f"- Leghosszabb RR: {np.max(result['clean_rr']):.3f} s ({60 / np.max(result['clean_rr']):.0f} bpm), "
            f"legrövidebb RR: {np.min(result['clean_rr']):.3f} s ({60 / np.min(result['clean_rr']):.0f} bpm)")
        rmssd = float(np.sqrt(np.mean(np.diff(result["clean_rr"]) ** 2)))
        add(f"- RMSSD: {rmssd * 1000:.0f} ms; SDNN: {np.std(result['clean_rr']) * 1000:.0f} ms")
        add("")

    verdicts = ["valódi", "bizonytalan", "artifact"]
    add("## Eseménytábla: típus × verdikt")
    add("")
    add("| esemény | valódi | bizonytalan | artifact | összes |")
    add("| --- | ---: | ---: | ---: | ---: |")
    for kind in sorted({e.kind for e in events}):
        row = [sum(1 for e in events if e.kind == kind and e.verdict == v) for v in verdicts]
        add(f"| {kind} | {row[0]} | {row[1]} | {row[2]} | {sum(row)} |")
    total = [sum(1 for e in events if e.verdict == v) for v in verdicts]
    add(f"| **összes** | **{total[0]}** | **{total[1]}** | **{total[2]}** | **{len(events)}** |")
    add("")

    hours = max(result["analysable_minutes"] / 60.0, 1e-9)
    confirmed = [e for e in events if e.verdict == "valódi"]
    ventricular = [e for e in confirmed if e.kind == "korai_kamrai_jellegű"]
    atrial = [e for e in confirmed if e.kind == "korai_pitvari_jellegű"]
    bursts = [e for e in confirmed if e.kind == "korai_sorozat"]
    pauses = [e for e in confirmed if e.kind == "szünet"]
    add("## Igazolt ritmuseltérések terhelése")
    add("")
    add(f"- Elemezhető idő: **{result['analysable_minutes']:.1f} perc** ({hours:.2f} óra)")
    add(f"- Kamrai jellegű korai ütés: {len(ventricular)} db → {len(ventricular) / hours:.1f}/óra")
    add(f"- Pitvari jellegű korai ütés: {len(atrial)} db → {len(atrial) / hours:.1f}/óra")
    add(f"- Korai sorozat (≥2 egymást követő): {len(bursts)} db")
    if pauses:
        longest = max(pauses, key=lambda e: e.metrics.get("rr_s", 0.0))
        add(f"- Igazolt szünet: {len(pauses)} db, leghosszabb {longest.metrics['rr_s']:.2f} s "
            f"({_local(longest.start, timezone)})")
    else:
        add("- Igazolt szünet: 0 db")
    add("")
    add("### Alakcsoport-ellenőrzés")
    add("")
    add("Valódi ektópia jellemzően monomorf: ugyanaz a góc ugyanazt a QRS-alakot adja. Ha a")
    add("valódinak ítélt korai ütések nem korrelálnak egymással, az inkább maradék detektálási")
    add("zajra utal, mint tényleges ektópiára.")
    add("")
    if clusters:
        sizes = [len(c) for c in clusters]
        add(f"- Valódinak ítélt korai ütés: {sum(sizes)} db, {len(clusters)} alakcsoportban "
            f"(korrelációs küszöb {MONOMORPHIC_CORRELATION}).")
        add(f"- A legnagyobb csoport {sizes[0]} ütés ({sizes[0] / max(sum(sizes), 1) * 100:.0f}%), "
            f"az első három együtt {sum(sizes[:3])} ütés.")
        add(f"- Egyelemű csoport: {sum(1 for s in sizes if s == 1)} db.")
    else:
        add("- Nincs valódinak ítélt korai ütés, alakcsoport-elemzés nem értelmezhető.")
    add("")

    for verdict, title, note in (
        ("valódi", "Valódinak ítélt események",
         "A hullámforma alátámasztja őket, a környezet tiszta. Szakorvosi átnézésre ezek ajánlottak."),
        ("bizonytalan", "Bizonytalan események",
         "A jelminőség nem engedi meg a döntést: sem kizárni, sem igazolni nem lehet őket."),
    ):
        selected = [(i, e) for i, e in enumerate(events) if e.verdict == verdict]
        add(f"## {title} ({len(selected)} db)")
        add("")
        add(note)
        add("")
        if not selected:
            add("Nincs ilyen esemény.")
            add("")
            continue
        add("| id | idő | típus | kulcsszámok | indoklás |")
        add("| ---: | --- | --- | --- | --- |")
        for position, event in selected[:150]:
            key = "; ".join(
                f"{k}={v:.3g}" for k, v in list(event.metrics.items())[:4] if isinstance(v, float)
            )
            add(f"| {position} | {_local(event.start, timezone)} | {event.kind} | {key} | {event.reason} |")
        if len(selected) > 150:
            add(f"| … | | | | további {len(selected) - 150} sor az `events.csv`-ben |")
        add("")

    rejected = [e for e in events if e.verdict == "artifact"]
    add(f"## Műterméknek ítélt események ({len(rejected)} db)")
    add("")
    add("Ezeket a hullámforma cáfolja: vagy a rés kitölthető alacsonyabb küszöbbel, vagy a")
    add("kitérés nem QRS-alakú, vagy a környezet mérhetően zajos.")
    add("")
    if rejected:
        causes: dict[str, int] = {}
        for event in rejected:
            causes[event.reason] = causes.get(event.reason, 0) + 1
        add("| ok | darab |")
        add("| --- | ---: |")
        for reason, count in sorted(causes.items(), key=lambda item: -item[1]):
            add(f"| {reason} | {count} |")
    else:
        add("Nincs ilyen esemény.")
    add("")
    add("## Kimenetek")
    add("")
    add("- `beats.csv`: minden detektált ütés ideje, prominenciája, QRS-szélessége, template-korrelációja, epoch-osztálya.")
    add("- `events.csv`: minden esemény verdikttel, indoklással, mért bizonyítékokkal és a csík fájlnevével.")
    add(f"- `strips/`: {result['strips']} db EKG-csík (nyers + 0,5–40 Hz), RR-értékekkel és megjelölt eseménnyel.")
    add("- `audit_overview.png`: jelállapot-idősor, RR-tachogram az eseményekkel, Poincaré-diagram.")
    add("- `audit_morphology.png`: QRS-szélesség/korreláció eloszlás, alakcsoportok átlagos hullámformája és mérete.")
    add("")
    add("## Módszertani küszöbök")
    add("")
    add("A jelminőségi küszöbök a felvételek tényleges eloszlásából származnak: tiszta epochon")
    add("a QRS-en kívüli p99 a saját R-prominencia 6–12%-a, az RMS 2–4%-a.")
    add("")
    add(f"- Jel nélküli epoch: 2 s-os amplitúdó < {DEAD_AMPLITUDE_MV} mV.")
    add(f"- Zajos epoch: QRS-en kívüli p99/R > {CLEAN_OFF_P99}, RMS/R > {CLEAN_OFF_RMS}, "
        f"alapvonal-kilengés/R > {CLEAN_BASELINE}, vagy amplitúdó > {AMPLITUDE_INFLATION}× a felvétel mediánjának. "
        "A zajos jelölés egy epochnyit terjed mindkét irányba.")
    add(f"- Durva műtermék (azonnali elutasítás): p99/R > {GROSS_OFF_P99}, alapvonal/R > {GROSS_BASELINE}, "
        "jel nélküli epoch az ablakban, vagy a szomszédos ütések template-korrelációja < 0,70.")
    add(f"- Szünet: RR ≥ {PAUSE_ABSOLUTE_S} s vagy ≥ {PAUSE_RELATIVE}× a lokális medián RR; a résben "
        "kötelező alacsonyabb küszöbű újrakeresés.")
    add(f"- Korai ütés: RR ≤ {1 - PREMATURITY:.2f}× a lokális medián RR (±10 ütés mediánja, önmagát kihagyva).")
    add(f"- Fiziológiai padló: {MIN_COUPLING_S} s alatti kapcsolási idő automatikusan műtermék.")
    add(f"- Egyszerre > {TALL_WIDE_WIDTH}× szélesebb és > {TALL_WIDE_PROMINENCE}× magasabb kitérés: bizonytalan.")
    add(f"- Elfogadott QRS-szélesség korai ütésnél: {MIN_QRS_FACTOR}–{MAX_QRS_FACTOR}× a normálénak; "
        f"kamrai jelleg ≥ {WIDE_QRS_FACTOR}× vagy template-korreláció < {LOW_CORRELATION}.")
    add(f"- Bradycardia: ≥ {EPISODE_SECONDS:.0f} s tartósan < {BRADY_BPM:.0f} bpm; tachycardia > {TACHY_REST_BPM:.0f} bpm.")
    add(f"- Szabálytalan szakasz: {IRREGULAR_WINDOW_S:.0f} s ablakban RMSSD/medián RR ≥ {IRREGULAR_RMSSD} korai ütések nélkül.")
    add(f"- Ezen belül légzési sinus arrhythmia, ha az RR-sor lag-1 autokorrelációja ≥ {RSA_AUTOCORRELATION} "
        "(sima, periodikus ingadozás); pitvarfibrilláció-gyanús mintázat ennél alacsonyabb autokorrelációnál lenne.")
    add(f"- Korai ütés csak akkor ektópia, ha hirtelen: az előtte lévő RR ≥ {ABRUPTNESS}× a lokális medián. "
        "A fokozatosan rövidülő RR-sor légzési sinus arrhythmia.")
    add("")
    (result["output"] / "report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--session",
        action="append",
        metavar="NÉV=BDF[:KIMENET]",
        help="Session neve, BDF útvonal, opcionálisan kimeneti könyvtár; többször megadható.",
    )
    parser.add_argument("--timezone", default="Europe/Budapest")
    parser.add_argument("--strips", type=int, default=400, help="Legfeljebb ennyi EKG-csíkot rajzol.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    specs = args.session or [
        "2026-08-28=08_28/15673_20260828_164058.bdf:08_28/audit",
        "2026-09-01=09_01/15673_20260901_164946.bdf:09_01/audit",
    ]
    for spec in specs:
        name, _, rest = spec.partition("=")
        bdf_text, _, output_text = rest.partition(":")
        bdf = Path(bdf_text)
        output = Path(output_text) if output_text else bdf.parent / "audit"
        result = audit(bdf, args.timezone, output, args.strips)
        write_report(name, bdf, result)
        counts: dict[str, int] = {}
        for event in result["events"]:
            counts[event.verdict] = counts.get(event.verdict, 0) + 1
        print(
            f"{name}: {result['analysable_minutes']:.1f}/{result['live_minutes']:.1f} perc elemezhető, "
            f"{result['beats']} ütés, {len(result['events'])} esemény "
            f"({', '.join(f'{k}={v}' for k, v in sorted(counts.items()))}) -> {output}"
        )


if __name__ == "__main__":
    main()
