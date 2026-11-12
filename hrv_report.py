#!/usr/bin/env python3
"""HRV-diagnosztika a beat-szintű audit NN-sorából: Poincaré, spektrum, DFA, entrópia.

Bemenet az `ecg_audit.py` `beats.csv` kimenete. Csak az elemezhető (0) epochba eső
ütések számítanak, és csak azok az RR-ek, amelyek mindkét végükön tiszta ütéshez
tartoznak. Az ektópiák és a hozzájuk tartozó kompenzációs szünetek kiesnek: az NN-sor
kizárólag normál ciklusokat tartalmaz, különben a szórásmérőszámokat néhány
extraszisztolé uralná.

Ez kutatási/technikai eszköz, nem orvostechnikai eszköz, és a kimenete nem diagnózis.
A HRV-mérőszámok stacionárius szakaszt feltételeznek, ezért nyugalmi és terheléses
részt külön is számolunk.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import lombscargle

NN_MIN_S = 0.30
NN_MAX_S = 2.00
NN_JUMP = 0.20            # szomszédos NN-ek közti megengedett relatív ugrás
MIN_SEGMENT_BEATS = 30
REST_HR_MAX = 100.0       # e fölött terhelésnek tekintjük


def load_nn(path: Path) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Idő és NN-sor, valamint összefüggő NN-szakaszok listája."""
    times, rr, clean = [], [], []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            times.append(datetime.fromisoformat(row["beat_time_local"]).timestamp())
            rr.append(float(row["rr_prev_s"]) if row["rr_prev_s"] else np.nan)
            clean.append(row["epoch_class"] == "0")
    times = np.asarray(times)
    rr = np.asarray(rr)
    clean = np.asarray(clean)

    valid = np.zeros(len(rr), dtype=bool)
    valid[1:] = (
        clean[1:] & clean[:-1] & (rr[1:] >= NN_MIN_S) & (rr[1:] <= NN_MAX_S)
    )
    # ektópia-szűrés: a szomszédjához képest nagyot ugró intervallum és a rákövetkező is kiesik
    jump = np.zeros(len(rr), dtype=bool)
    jump[2:] = np.abs(rr[2:] - rr[1:-1]) > NN_JUMP * rr[1:-1]
    drop = jump.copy()
    drop[1:] |= jump[:-1]
    valid &= ~drop

    segments = []
    edges = np.diff(np.r_[0, valid.view(np.int8), 0])
    for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if end - start >= MIN_SEGMENT_BEATS:
            segments.append(np.arange(start, end))
    return times, rr, segments


def time_domain(nn: np.ndarray) -> dict[str, float]:
    differences = np.diff(nn)
    histogram, edges = np.histogram(nn, bins=np.arange(nn.min(), nn.max() + 1 / 128, 1 / 128))
    return {
        "ütés": float(len(nn)),
        "átlag NN (ms)": float(np.mean(nn) * 1000),
        "átlag HR (bpm)": float(60.0 / np.mean(nn)),
        "SDNN (ms)": float(np.std(nn, ddof=1) * 1000),
        "RMSSD (ms)": float(np.sqrt(np.mean(differences**2)) * 1000),
        "pNN50 (%)": float(np.mean(np.abs(differences) > 0.050) * 100),
        "CVNN (%)": float(np.std(nn, ddof=1) / np.mean(nn) * 100),
        "trianguláris index": float(len(nn) / histogram.max()) if histogram.max() else np.nan,
    }


def poincare(nn: np.ndarray) -> dict[str, float]:
    """SD1 a rövid távú, SD2 a hosszú távú szórás; SD1/SD2 az arányuk."""
    x, y = nn[:-1], nn[1:]
    sd1 = float(np.std((y - x) / np.sqrt(2), ddof=1))
    sd2 = float(np.std((y + x) / np.sqrt(2), ddof=1))
    return {
        "SD1 (ms)": sd1 * 1000,
        "SD2 (ms)": sd2 * 1000,
        "SD1/SD2": sd1 / sd2 if sd2 > 0 else np.nan,
        "ellipszis terület (ms²)": float(np.pi * sd1 * sd2 * 1e6),
    }


def spectrum(times: np.ndarray, nn: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Lomb–Scargle periodogram: az NN-sor egyenetlen mintavételű, ezért nem FFT."""
    centred = nn - np.mean(nn)
    frequency = np.linspace(0.003, 0.45, 2000)
    power = lombscargle(times, centred, 2 * np.pi * frequency, normalize=False)
    power = power * 2.0 / len(times) * 1e6  # ms²/Hz nagyságrend
    def band(lo, hi):
        mask = (frequency >= lo) & (frequency < hi)
        return float(np.trapezoid(power[mask], frequency[mask])) if np.any(mask) else np.nan
    vlf, lf, hf = band(0.003, 0.04), band(0.04, 0.15), band(0.15, 0.40)
    return frequency, power, {
        "VLF (ms²)": vlf,
        "LF (ms²)": lf,
        "HF (ms²)": hf,
        "LF/HF": lf / hf if hf and np.isfinite(hf) and hf > 0 else np.nan,
        "LF n.u.": 100 * lf / (lf + hf) if (lf + hf) > 0 else np.nan,
    }


def detrended_fluctuation(nn: np.ndarray, scales: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """DFA: az integrált, trendmentesített sor fluktuációja skálánként."""
    walk = np.cumsum(nn - np.mean(nn))
    fluctuation = []
    used = []
    for scale in scales:
        count = len(walk) // scale
        if count < 4:
            continue
        blocks = walk[: count * scale].reshape(count, scale)
        index = np.arange(scale)
        residual = []
        for block in blocks:
            slope, intercept = np.polyfit(index, block, 1)
            residual.append(block - (slope * index + intercept))
        fluctuation.append(float(np.sqrt(np.mean(np.concatenate(residual) ** 2))))
        used.append(scale)
    return np.asarray(used), np.asarray(fluctuation)


def dfa_alpha(nn: np.ndarray) -> dict[str, float]:
    short, short_f = detrended_fluctuation(nn, np.arange(4, 17))
    long, long_f = detrended_fluctuation(nn, np.arange(16, 65, 4))
    out = {}
    if len(short) >= 3:
        out["DFA α1"] = float(np.polyfit(np.log(short), np.log(short_f), 1)[0])
    if len(long) >= 3:
        out["DFA α2"] = float(np.polyfit(np.log(long), np.log(long_f), 1)[0])
    return out


def sample_entropy(nn: np.ndarray, m: int = 2, tolerance: float = 0.2, cap: int = 3000) -> float:
    """SampEn(m=2, r=0.2·SD); nagy sorozatnál az elejét használjuk, mert O(n²)."""
    series = nn[:cap]
    r = tolerance * float(np.std(series, ddof=1))
    if r <= 0 or len(series) < m + 2:
        return float("nan")

    def count(dimension: int) -> int:
        vectors = np.lib.stride_tricks.sliding_window_view(series, dimension)
        total = 0
        for i in range(len(vectors) - 1):
            distance = np.max(np.abs(vectors[i + 1 :] - vectors[i]), axis=1)
            total += int(np.count_nonzero(distance <= r))
        return total

    a, b = count(m + 1), count(m)
    return float(-np.log(a / b)) if a > 0 and b > 0 else float("nan")


def analyse(nn: np.ndarray, times: np.ndarray) -> dict[str, float]:
    result: dict[str, float] = {}
    result.update(time_domain(nn))
    result.update(poincare(nn))
    result.update(spectrum(times, nn)[2])
    result.update(dfa_alpha(nn))
    result["SampEn"] = sample_entropy(nn)
    return result


def render(name: str, times: np.ndarray, nn: np.ndarray, rest: np.ndarray, path: Path) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(18, 10))

    x, y = nn[:-1] * 1000, nn[1:] * 1000
    axes[0][0].plot(x, y, ".", ms=1.5, alpha=0.30, color="#1F6FB2")
    sd1 = np.std((y - x) / np.sqrt(2), ddof=1)
    sd2 = np.std((y + x) / np.sqrt(2), ddof=1)
    centre = (float(np.mean(x)), float(np.mean(y)))
    angle = np.deg2rad(45.0)
    theta = np.linspace(0, 2 * np.pi, 200)
    ellipse_x = sd2 * np.cos(theta)
    ellipse_y = sd1 * np.sin(theta)
    axes[0][0].plot(
        centre[0] + ellipse_x * np.cos(angle) - ellipse_y * np.sin(angle),
        centre[1] + ellipse_x * np.sin(angle) + ellipse_y * np.cos(angle),
        color="#B3261E", lw=2,
    )
    axes[0][0].set_xlabel("NN(n) (ms)")
    axes[0][0].set_ylabel("NN(n+1) (ms)")
    axes[0][0].set_title(f"Poincaré-plot  ·  SD1={sd1:.0f} ms, SD2={sd2:.0f} ms, SD1/SD2={sd1/sd2:.2f}")
    axes[0][0].set_aspect("equal")

    axes[0][1].hist(nn * 1000, bins=np.arange(nn.min() * 1000, nn.max() * 1000 + 8, 7.8125),
                    color="#1F6FB2", alpha=0.85)
    axes[0][1].set_xlabel("NN (ms)")
    axes[0][1].set_ylabel("darab")
    axes[0][1].set_title("NN-hisztogram (7,8 ms-os rekeszek, trianguláris index alapja)")

    axes[0][2].plot([datetime.fromtimestamp(t) for t in times], nn * 1000, ".", ms=1.2, color="0.3")
    axes[0][2].set_ylabel("NN (ms)")
    axes[0][2].set_title("NN-tachogram (csak normál ciklusok, tiszta jelen)")
    axes[0][2].tick_params(axis="x", rotation=30)

    frequency, power, bands = spectrum(times, nn)
    axes[1][0].semilogy(frequency, np.maximum(power, 1e-6), lw=0.9, color="#1F6FB2")
    for lo, hi, colour, label in (
        (0.003, 0.04, "#CCCCCC", "VLF"), (0.04, 0.15, "#F2C14E", "LF"), (0.15, 0.40, "#7FB069", "HF")
    ):
        axes[1][0].axvspan(lo, hi, color=colour, alpha=0.35)
        axes[1][0].text((lo + hi) / 2, power.max(), label, ha="center", fontsize=9)
    axes[1][0].set_xlabel("frekvencia (Hz)")
    axes[1][0].set_ylabel("teljesítmény")
    axes[1][0].set_title(f"Lomb–Scargle spektrum  ·  LF/HF={bands['LF/HF']:.2f}")

    for label, series, colour in (("teljes", nn, "#1F6FB2"), ("nyugalmi", rest, "#B3261E")):
        if len(series) < 100:
            continue
        scales, fluctuation = detrended_fluctuation(series, np.arange(4, 65, 2))
        axes[1][1].loglog(scales, fluctuation, "o-", ms=4, lw=1.2, color=colour, label=label)
    axes[1][1].set_xlabel("ablakméret (ütés)")
    axes[1][1].set_ylabel("fluktuáció")
    axes[1][1].set_title("DFA log-log; α1 a 4–16, α2 a 16–64 ütéses tartomány meredeksége")
    axes[1][1].legend(fontsize=9)

    axes[1][2].plot(nn[:-1] * 1000, np.diff(nn) * 1000, ".", ms=1.2, alpha=0.3, color="#1F6FB2")
    axes[1][2].set_xlabel("NN(n) (ms)")
    axes[1][2].set_ylabel("ΔNN (ms)")
    axes[1][2].set_title("Különbségi (Lorenz) diagram")

    for row in axes:
        for axis in row:
            axis.grid(alpha=0.3)
    figure.suptitle(f"HRV-diagnosztika – {name}", fontsize=13)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", action="append", metavar="NÉV=BEATS.CSV")
    args = parser.parse_args()
    specs = args.session or [
        "2026-08-28=08_28/audit/beats.csv",
        "2026-09-01=09_01/audit/beats.csv",
    ]
    for spec in specs:
        name, _, path_text = spec.partition("=")
        path = Path(path_text)
        times, rr, segments = load_nn(path)
        if not segments:
            print(f"{name}: nincs elég hosszú NN-szakasz")
            continue
        index = np.concatenate(segments)
        nn, moment = rr[index], times[index]
        rest = nn[60.0 / nn <= REST_HR_MAX]
        report = [f"# HRV-diagnosztika – {name}", "",
                  "> Technikai elemzés, nem diagnózis. A HRV-mérőszámok stacionárius szakaszt",
                  "> feltételeznek, ezért a nyugalmi részt külön is kiszámoljuk.", "",
                  f"NN-szakaszok: {len(segments)} db, összesen {len(nn)} normál ciklus "
                  f"({np.sum(nn)/60:.1f} perc). Az ektópiák és a hozzájuk tartozó szünetek kiestek.", ""]
        for label, series, moments in (
            ("Teljes elemezhető felvétel", nn, moment),
            (f"Nyugalmi rész (HR ≤ {REST_HR_MAX:.0f} bpm)", rest, moment[60.0 / nn <= REST_HR_MAX]),
        ):
            if len(series) < 100:
                continue
            report += [f"## {label}", "", "| mérőszám | érték |", "| --- | ---: |"]
            for key, value in analyse(series, moments).items():
                report.append(f"| {key} | {value:.4g} |")
            report.append("")
        # 5 perces ablakok: a klinikai referenciaértékek stacionárius szakaszra
        # vonatkoznak, a teljes felvétel SDNN-jét a terhelési trend felfújja.
        windows = []
        for start in np.arange(moment[0], moment[-1], 300.0):
            inside = (moment >= start) & (moment < start + 300.0)
            block = nn[inside]
            if len(block) < 200:
                continue
            differences = np.diff(block)
            x, y = block[:-1], block[1:]
            windows.append(
                (
                    60.0 / np.mean(block),
                    np.std(block, ddof=1) * 1000,
                    np.sqrt(np.mean(differences**2)) * 1000,
                    np.mean(np.abs(differences) > 0.050) * 100,
                    np.std((y - x) / np.sqrt(2), ddof=1) * 1000,
                    np.std((y + x) / np.sqrt(2), ddof=1) * 1000,
                )
            )
        if windows:
            table = np.asarray(windows)
            report += [f"## 5 perces ablakok ({len(table)} db, stacionárius közelítés)", "",
                       "| mérőszám | medián | p10 | p90 |", "| --- | ---: | ---: | ---: |"]
            for column, label in enumerate(
                ["átlag HR (bpm)", "SDNN (ms)", "RMSSD (ms)", "pNN50 (%)", "SD1 (ms)", "SD2 (ms)"]
            ):
                values = table[:, column]
                report.append(
                    f"| {label} | {np.median(values):.1f} | {np.percentile(values, 10):.1f} | "
                    f"{np.percentile(values, 90):.1f} |"
                )
            report.append("")
        render(name, moment, nn, rest, path.parent / "hrv_overview.png")
        (path.parent / "hrv_report.md").write_text("\n".join(report), encoding="utf-8")
        summary = analyse(nn, moment)
        print(f"{name}: NN={len(nn)}  SDNN={summary['SDNN (ms)']:.0f}  RMSSD={summary['RMSSD (ms)']:.0f}  "
              f"pNN50={summary['pNN50 (%)']:.1f}%  SD1={summary['SD1 (ms)']:.0f}  SD2={summary['SD2 (ms)']:.0f}  "
              f"SD1/SD2={summary['SD1/SD2']:.2f}  LF/HF={summary['LF/HF']:.2f}  "
              f"α1={summary.get('DFA α1', float('nan')):.2f}  SampEn={summary['SampEn']:.2f}")


if __name__ == "__main__":
    main()
