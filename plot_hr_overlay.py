"""Nagy felbontású HR-átfedés ábra: csak a Huawei nyers és az EKG referencia.

A `comparison.csv` használható sorait rajzolja ki, az óra méréseinek réseinél
megszakítva a vonalat, hogy a kihagyott szakaszokat ne kösse össze látszólagos
folytonossággal.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

GAP_SECONDS = 5.0


def load(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    times, watch, reference, offsets = [], [], [], []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["usable"] != "1":
                continue
            watch_time = datetime.fromisoformat(row["watch_time_local"])
            reference_time = datetime.fromisoformat(row["reference_time_local"])
            offsets.append((reference_time - watch_time).total_seconds())
            # A matplotlib az időzónás értékeket UTC-ben rajzolná ki, ezért a
            # helyi faliórára váltunk, és a zónát a tengelyfeliratban közöljük.
            times.append(watch_time.replace(tzinfo=None))
            watch.append(float(row["watch_hr_bpm"]))
            reference.append(float(row["ecg_reference_hr_bpm"]))
    if not times:
        raise SystemExit(f"Nincs használható sor: {path}")
    return (
        np.asarray(times),
        np.asarray(watch, dtype=np.float64),
        np.asarray(reference, dtype=np.float64),
        float(np.median(offsets)),
    )


def split_gaps(times: np.ndarray, *series: np.ndarray) -> list[tuple[np.ndarray, ...]]:
    """Vonalszakaszok a réseknél elvágva."""
    seconds = np.array([t.timestamp() for t in times])
    breaks = np.flatnonzero(np.diff(seconds) > GAP_SECONDS) + 1
    pieces = []
    for chunk in np.split(np.arange(len(times)), breaks):
        if len(chunk) >= 2:
            pieces.append(tuple(np.asarray(s)[chunk] for s in (times, *series)))
    return pieces


def render(name: str, source: Path, destination: Path, dpi: int, width: float) -> None:
    times, watch, reference, offset = load(source)
    span_hours = (times[-1].timestamp() - times[0].timestamp()) / 3600.0

    figure, axes = plt.subplots(figsize=(width, width / 2.9), dpi=dpi)
    first = True
    for piece_times, piece_watch, piece_reference in split_gaps(times, watch, reference):
        axes.plot(
            piece_times,
            piece_reference,
            color="#E4453A",
            linewidth=1.0,
            label="MOX EKG referencia" if first else None,
            zorder=2,
        )
        axes.plot(
            piece_times,
            piece_watch,
            color="#3E6FA8",
            linewidth=1.0,
            label="Huawei Band 10 – nyers" if first else None,
            zorder=3,
        )
        first = False

    axes.set_title(
        f"Huawei Band 10 és MOX EKG – minőségkapuzott, időben illesztett HR  ·  {name}\n"
        f"EKG lookup offset: {offset:+.1f} s  ·  {len(times)} pár  ·  "
        f"{span_hours:.2f} óra  ·  a megszakítások kizárt EKG-blokkok",
        fontsize=13,
    )
    axes.set_xlabel("Helyi idő (Europe/Budapest)")
    axes.set_ylabel("Pulzus (bpm)")
    axes.grid(True, alpha=0.25, linewidth=0.6)
    handles, labels = axes.get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda i: "Huawei" not in labels[i])
    axes.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        loc="upper left",
        ncol=2,
        fontsize=11,
        framealpha=0.9,
    )
    axes.xaxis.set_major_locator(mdates.MinuteLocator(interval=10 if span_hours < 4 else 20))
    axes.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes.xaxis.set_minor_locator(mdates.MinuteLocator(interval=5))
    axes.margins(x=0.005)
    for spine in ("top", "right"):
        axes.spines[spine].set_visible(False)

    figure.tight_layout()
    figure.savefig(destination, dpi=dpi)
    plt.close(figure)
    image = plt.imread(destination)
    print(f"{destination}  ({image.shape[1]}×{image.shape[0]} px, n={len(times)})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--session",
        action="append",
        metavar="NÉV=CSV",
        help="Session neve és comparison.csv útvonala; többször megadható.",
    )
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--width", type=float, default=30.0, help="Ábra szélessége hüvelykben")
    args = parser.parse_args()

    specs = args.session or [
        "2026-08-28=08_28/results_v2/comparison.csv",
        "2026-09-01=09_01/results/comparison.csv",
    ]
    for spec in specs:
        name, _, path = spec.partition("=")
        source = Path(path)
        render(name, source, source.parent / "hr_overlay_raw_vs_ecg.png", args.dpi, args.width)


if __name__ == "__main__":
    main()
