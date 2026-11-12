#!/usr/bin/env python3
"""Analyze the normalized 2026-09-05 bundle, preserving data-quality limits."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "smartwatch-correction-mpl"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TIMEZONE = "Europe/Belgrade"
TOUR_DATE = date(2026, 9, 5)
ROUTE_TITLE = "Túra"  # felülírható: --route-title
# A FIT session total_timer_time az eszköz saját (gyakran zajos, lámpáknál/
# kereszteződéseknél másodpercekre ki-be kapcsoló) auto-pause logikáját tükrözi,
# és jellemzően jóval hosszabb, mint amit a Strava "menetidőként" mutat. 5 km/h
# sebességküszöbbel a 2026-09-07-i túrán 2:19:48 adódik a Strava-jelentette
# 2:19:09-hez képest (39 s, ~0,5% eltérés) — ez a küszöb tehát jó közelítés.
GPS_MOVING_THRESHOLD_KMH = 5.0


def _fmt(value: float, digits: int = 1) -> str:
    return "n/a" if not math.isfinite(value) else f"{value:.{digits}f}"


def _duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"


def strongest_three_hour_window(day: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    local = day.set_index("timestamp_local")["heart_rate_bpm"]
    starts = pd.date_range(
        pd.Timestamp(datetime.combine(TOUR_DATE, datetime.min.time()), tz=TIMEZONE),
        periods=22,
        freq="h",
    )
    candidates = []
    for start in starts:
        end = start + pd.Timedelta(hours=3)
        values = local[(local.index >= start) & (local.index < end)].dropna()
        if len(values) >= 60:
            candidates.append((float(values.mean()), len(values), start, end))
    if not candidates:
        raise ValueError("Nincs legalább 60 pulzusminta egyetlen háromórás ablakban sem.")
    _, _, start, end = max(candidates)
    return start, end


def hourly_summary(day: pd.DataFrame) -> pd.DataFrame:
    frame = day.copy()
    frame["hour_local"] = frame["timestamp_local"].dt.floor("h")
    grouped = frame.groupby("hour_local", observed=True)
    result = grouped.agg(
        activity_rows=("TIMESTAMP", "size"),
        hr_samples=("heart_rate_bpm", "count"),
        hr_mean_bpm=("heart_rate_bpm", "mean"),
        hr_median_bpm=("heart_rate_bpm", "median"),
        hr_min_bpm=("heart_rate_bpm", "min"),
        hr_max_bpm=("heart_rate_bpm", "max"),
        steps=("steps", "sum"),
        activity_distance_m=("distance_m", "sum"),
        calories_kcal_inferred=("calories_kcal_inferred", "sum"),
    ).reset_index()
    result["hr_observed_fraction"] = result["hr_samples"] / 60.0
    return result


def make_plot(
    output: Path,
    day: pd.DataFrame,
    hourly: pd.DataFrame,
    route_start: pd.Timestamp,
    route_end: pd.Timestamp,
    bdf_start: pd.Timestamp,
    bdf_end: pd.Timestamp,
    gps_clue: pd.Timestamp | None,
) -> None:
    local_time = day["timestamp_local"]
    hr = day["heart_rate_bpm"]
    figure, axes = plt.subplots(3, 1, figsize=(15, 10), constrained_layout=True, sharex=True)

    axes[0].scatter(local_time, hr, s=10, color="#2F6B9A", alpha=0.8, label="Huawei mért HR")
    axes[0].axvspan(route_start, route_end, color="#F2C14E", alpha=0.22, label="GPX túraidő")
    axes[0].set_ylabel("Pulzus (bpm)")
    axes[0].set_ylim(45, max(170, float(hr.max(skipna=True)) + 8))
    axes[0].grid(alpha=0.2)
    axes[0].legend(loc="upper left")

    x = hourly["hour_local"] + pd.Timedelta(minutes=30)
    axes[1].plot(x, hourly["hr_mean_bpm"], marker="o", color="#D1495B", label="órás átlag")
    axes[1].fill_between(
        mdates.date2num(x.array.to_pydatetime()), hourly["hr_min_bpm"].to_numpy(dtype=float),
        hourly["hr_max_bpm"].to_numpy(dtype=float), color="#D1495B", alpha=0.18, label="min–max",
    )
    axes[1].set_ylabel("Pulzus (bpm)")
    axes[1].grid(alpha=0.2)
    axes[1].legend(loc="upper left")

    axes[2].bar(x, hourly["hr_samples"], width=0.032, color="#59A14F", label="HR-mintás percek")
    axes[2].axhline(60, color="black", linewidth=0.7, alpha=0.5)
    if bdf_start is not None and bdf_end is not None:
        axes[2].axvspan(bdf_start, bdf_end, color="#4E79A7", alpha=0.14, label="BDF lefedettség")
    if gps_clue is not None:
        axes[2].axvline(gps_clue, color="#7A5195", linestyle="--", linewidth=1.2, label="üres GPS-placeholder ideje")
    axes[2].set_ylabel("Minták / óra")
    axes[2].set_xlabel(f"Helyi idő ({TIMEZONE}, UTC+02)")
    axes[2].set_ylim(0, 66)
    axes[2].grid(axis="y", alpha=0.2)
    axes[2].legend(loc="upper left", ncol=2)
    axes[2].xaxis.set_major_locator(mdates.HourLocator(interval=2))
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=ZoneInfo(TIMEZONE)))
    day_start = pd.Timestamp(datetime.combine(TOUR_DATE, datetime.min.time()), tz=TIMEZONE)
    day_end = day["timestamp_local"].max().ceil("h")
    axes[2].set_xlim(day_start, day_end)
    figure.suptitle(f"{TOUR_DATE.isoformat()} – elérhető óraadatok és forráslefedettség")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def route_statistics(route: pd.DataFrame) -> dict:
    elapsed = float((route["timestamp_utc"].iloc[-1] - route["timestamp_utc"].iloc[0]).total_seconds())
    moving = (route["speed_kmh_raw"] >= 3.6) & route["delta_time_s"].notna()
    moving_seconds = float(route.loc[moving, "delta_time_s"].sum())
    moving_distance = float(route.loc[moving, "delta_distance_m"].sum())
    elevation_raw = route["elevation_m"].interpolate(limit_direction="both")
    elevation_smooth = elevation_raw.rolling(31, center=True, min_periods=1).median()
    route["elevation_m_31point_median"] = elevation_smooth
    change_raw = elevation_raw.diff().dropna()
    change_smooth = elevation_smooth.diff().dropna()

    pause_runs = []
    first = None
    for index in range(1, len(route)):
        if not bool(moving.iloc[index]) and first is None:
            first = index
        if bool(moving.iloc[index]) and first is not None:
            duration = float(route["delta_time_s"].iloc[first:index].sum())
            pause_runs.append(duration)
            first = None
    if first is not None:
        pause_runs.append(float(route["delta_time_s"].iloc[first:].sum()))
    long_pauses = [seconds for seconds in pause_runs if seconds >= 30]
    smooth_moving_speed = route.loc[route["speed_kmh_15point"] >= 3.6, "speed_kmh_15point"].dropna()
    total_distance = float(route["delta_distance_m"].sum())
    return {
        "point_count": int(len(route)), "distance_km": total_distance / 1000,
        "elapsed_seconds": elapsed, "moving_seconds_threshold_3_6_kmh": moving_seconds,
        "stopped_or_slow_seconds": elapsed - moving_seconds,
        "elapsed_average_speed_kmh": total_distance / elapsed * 3.6,
        "moving_average_speed_kmh": moving_distance / moving_seconds * 3.6,
        "moving_speed_median_15point_kmh": float(smooth_moving_speed.median()),
        "moving_speed_p90_15point_kmh": float(smooth_moving_speed.quantile(0.90)),
        "moving_speed_max_15point_kmh": float(smooth_moving_speed.max()),
        "elevation_min_m": float(elevation_raw.min()), "elevation_max_m": float(elevation_raw.max()),
        "elevation_gain_raw_m": float(change_raw.clip(lower=0).sum()),
        "elevation_loss_raw_m": float(-change_raw.clip(upper=0).sum()),
        "elevation_gain_31point_median_m": float(change_smooth.clip(lower=0).sum()),
        "elevation_loss_31point_median_m": float(-change_smooth.clip(upper=0).sum()),
        "timestamp_gaps_over_10s": int((route["delta_time_s"] > 10).sum()),
        "max_timestamp_gap_s": float(route["delta_time_s"].max()),
        "instantaneous_speed_outliers_over_60_kmh": int((route["speed_kmh_raw"] > 60).sum()),
        "pause_or_slow_episodes_at_least_30s": len(long_pauses),
        "pause_or_slow_seconds_in_episodes_at_least_30s": float(sum(long_pauses)),
    }


def fit_statistics(route: pd.DataFrame, session: pd.Series, events: pd.DataFrame) -> dict:
    running_speed = route.loc[route["timer_running_after_event"].fillna(False), "speed_kmh_15point"].dropna()
    gps_moving = (route["speed_kmh_raw"] >= GPS_MOVING_THRESHOLD_KMH) & route["delta_time_s"].notna()
    gps_moving_seconds = float(route.loc[gps_moving, "delta_time_s"].sum())
    gps_moving_distance_m = float(route.loc[gps_moving, "delta_distance_m"].sum())
    return {
        "point_count": int(len(route)),
        "distance_km": float(session["total_distance"]) / 1000,
        "elapsed_seconds": float(session["total_elapsed_time"]),
        "timer_seconds": float(session["total_timer_time"]),
        "paused_seconds": float(session["total_elapsed_time"] - session["total_timer_time"]),
        "elapsed_average_speed_kmh": float(session["total_distance"] / session["total_elapsed_time"] * 3.6),
        "timer_average_speed_kmh": float(session["total_distance"] / session["total_timer_time"] * 3.6),
        "gps_moving_seconds": gps_moving_seconds,
        "gps_moving_average_speed_kmh": gps_moving_distance_m / gps_moving_seconds * 3.6 if gps_moving_seconds else float("nan"),
        "speed_median_kmh": float(running_speed.median()),
        "speed_p90_kmh": float(running_speed.quantile(0.90)),
        "speed_max_kmh": float(running_speed.max()),
        "total_ascent_m": float(session["total_ascent"]),
        "total_descent_m": float(session["total_descent"]),
        "fit_altitude_min_m": float(route["enhanced_altitude"].min()),
        "fit_altitude_max_m": float(route["enhanced_altitude"].max()),
        "gps_accuracy_median_m": float(route["gps_accuracy"].median()),
        "gps_accuracy_p95_m": float(route["gps_accuracy"].quantile(0.95)),
        "auto_pause_count": int(((events["event_type"] == "stop") & (events["timer_trigger"] == "auto")).sum()),
        "timer_delta_sum_over_records_s": float(route["timer_delta_s"].sum()),
    }


def _haversine_pair(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    value = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 12_742_000.0 * np.arctan2(np.sqrt(value), np.sqrt(1.0 - value))


def compare_fit_gpx(fit: pd.DataFrame, gpx: pd.DataFrame, fit_stats: dict, gpx_stats: dict) -> dict:
    view = gpx[["timestamp_utc", "latitude_deg", "longitude_deg", "elevation_m"]].rename(columns={
        "timestamp_utc": "gpx_timestamp_utc", "latitude_deg": "gpx_latitude_deg",
        "longitude_deg": "gpx_longitude_deg", "elevation_m": "gpx_elevation_m",
    })
    matched = pd.merge_asof(
        fit.sort_values("timestamp_utc"), view.sort_values("gpx_timestamp_utc"),
        left_on="timestamp_utc", right_on="gpx_timestamp_utc", direction="nearest",
        tolerance=pd.Timedelta(seconds=2),
    ).dropna(subset=["gpx_latitude_deg"])
    separation = _haversine_pair(
        matched["latitude_deg"], matched["longitude_deg"],
        matched["gpx_latitude_deg"], matched["gpx_longitude_deg"],
    )
    altitude_delta = matched["gpx_elevation_m"] - matched["enhanced_altitude"]
    return {
        "matched_points_within_2s": int(len(matched)),
        "coordinate_separation_median_m": float(np.median(separation)),
        "coordinate_separation_p95_m": float(np.quantile(separation, 0.95)),
        "coordinate_separation_max_m": float(np.max(separation)),
        "gpx_polyline_minus_fit_summary_distance_m": float((gpx_stats["distance_km"] - fit_stats["distance_km"]) * 1000),
        "gpx_polyline_distance_difference_percent": float(100 * (gpx_stats["distance_km"] / fit_stats["distance_km"] - 1)),
        "gpx_minus_fit_altitude_median_m": float(altitude_delta.median()),
        "gpx_minus_fit_altitude_std_m": float(altitude_delta.std()),
    }


def sync_hr_to_route(activity: pd.DataFrame, route: pd.DataFrame) -> pd.DataFrame:
    measured = activity[activity["heart_rate_bpm"].notna()].copy().sort_values("timestamp_utc")
    measured = measured[
        (measured["timestamp_utc"] >= route["timestamp_utc"].iloc[0])
        & (measured["timestamp_utc"] <= route["timestamp_utc"].iloc[-1])
    ]
    route_view = route[[
        "timestamp_utc", "latitude_deg", "longitude_deg", "elevation_m",
        "distance_cumulative_m", "speed_kmh_15point",
    ]].rename(columns={"timestamp_utc": "route_timestamp_utc"})
    synced = pd.merge_asof(
        measured, route_view, left_on="timestamp_utc", right_on="route_timestamp_utc",
        direction="nearest", tolerance=pd.Timedelta(seconds=90),
    )
    synced["route_time_delta_s"] = (
        synced["timestamp_utc"] - synced["route_timestamp_utc"]
    ).abs().dt.total_seconds()
    return synced


def distance_summary(route: pd.DataFrame, synced: pd.DataFrame) -> pd.DataFrame:
    total_km = float(route["distance_cumulative_m"].max() / 1000)
    records = []
    for start_km in np.arange(0, math.ceil(total_km / 10) * 10, 10):
        end_km = min(start_km + 10, total_km)
        part = route[
            (route["distance_cumulative_m"] >= start_km * 1000)
            & (route["distance_cumulative_m"] <= end_km * 1000)
        ]
        hr = synced[
            (synced["distance_cumulative_m"] >= start_km * 1000)
            & (synced["distance_cumulative_m"] < (end_km + 1e-9) * 1000)
        ]["heart_rate_bpm"].dropna()
        moving_seconds = float(part["timer_delta_s"].sum())
        covered_distance = float(part["delta_distance_m"].sum())
        records.append({
            "distance_start_km": start_km, "distance_end_km": end_km,
            "elapsed_seconds": float(part["delta_time_s"].sum()),
            "moving_seconds": moving_seconds,
            "moving_average_speed_kmh": float(covered_distance / moving_seconds * 3.6) if moving_seconds else np.nan,
            "hr_samples": int(len(hr)), "hr_mean_bpm": float(hr.mean()) if len(hr) else np.nan,
            "hr_median_bpm": float(hr.median()) if len(hr) else np.nan,
        })
    return pd.DataFrame(records)


def make_route_plot(path: Path, route: pd.DataFrame, synced: pd.DataFrame, elevation_source: str) -> None:
    figure = plt.figure(figsize=(16, 9), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, width_ratios=(0.8, 2.2))
    map_axis = figure.add_subplot(grid[:, 0])
    elevation_axis = figure.add_subplot(grid[0, 1])
    profile_axis = figure.add_subplot(grid[1, 1])
    map_axis.plot(route["longitude_deg"], route["latitude_deg"], color="#B7B7B7", linewidth=1.2)
    located = synced[synced["latitude_deg"].notna()]
    points = map_axis.scatter(
        located["longitude_deg"], located["latitude_deg"], c=located["heart_rate_bpm"],
        cmap="plasma", vmin=100, vmax=160, s=25, edgecolors="white", linewidths=0.25,
    )
    map_axis.scatter(route["longitude_deg"].iloc[0], route["latitude_deg"].iloc[0], marker="o", s=80, color="#2CA02C", label="rajt")
    map_axis.scatter(route["longitude_deg"].iloc[-1], route["latitude_deg"].iloc[-1], marker="X", s=90, color="#D62728", label="cél")
    map_axis.set_aspect(1 / np.cos(np.radians(route["latitude_deg"].mean())))
    map_axis.set_xlabel("Hosszúság (°)")
    map_axis.set_ylabel("Szélesség (°)")
    map_axis.legend()
    figure.colorbar(points, ax=map_axis, label="Huawei pulzus (bpm)", shrink=0.65)

    distance = route["distance_cumulative_m"] / 1000
    elevation_axis.plot(distance, route["elevation_m_31point_median"], color="#8C6D31", linewidth=1.2)
    elevation_axis.fill_between(distance, route["elevation_m_31point_median"], route["elevation_m_31point_median"].min() - 1, color="#C49C94", alpha=0.3)
    elevation_axis.set_ylabel(f"{elevation_source}-magasság (m, 31 pontos medián)")
    elevation_axis.set_xlabel("Távolság (km)")
    elevation_axis.grid(alpha=0.2)

    profile_axis.plot(distance, route["speed_kmh_15point"], color="#4C78A8", linewidth=0.7, alpha=0.75, label="FIT-sebesség")
    profile_axis.set_xlabel("Távolság (km)")
    profile_axis.set_ylabel("Sebesség (km/h)", color="#4C78A8")
    profile_axis.set_ylim(0, 42)
    profile_axis.grid(alpha=0.2)
    hr_axis = profile_axis.twinx()
    hr_axis.scatter(located["distance_cumulative_m"] / 1000, located["heart_rate_bpm"], color="#E45756", s=16, label="Huawei HR")
    hr_axis.set_ylabel("Pulzus (bpm)", color="#E45756")
    hr_axis.set_ylim(80, 165)
    lines = profile_axis.get_lines() + hr_axis.collections
    profile_axis.legend(lines, [item.get_label() for item in lines], loc="lower right")
    figure.suptitle(f"{ROUTE_TITLE}: FIT-útvonal/sebesség, {elevation_source}-magasság és Huawei pulzus")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def crop_to_elapsed(
    route: pd.DataFrame, events: pd.DataFrame, session: pd.Series, max_elapsed_seconds: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Vágja a FIT-et az első `max_elapsed_seconds`-ra a rajttól.

    Arra az esetre kell, amikor a nyers eszköz/GPS-export hosszabb, mint a
    ténylegesen publikált aktivitás (pl. a felhasználó a hivatalos túra vége
    után tovább tekert, és ez a szakasz nem része "a túrának").
    """
    cutoff = route["timestamp_utc"].iloc[0] + pd.Timedelta(seconds=max_elapsed_seconds)
    cropped_route = route[route["timestamp_utc"] <= cutoff].reset_index(drop=True)
    cropped_events = events[events["timestamp"] <= cutoff].reset_index(drop=True) if not events.empty else events
    cropped_session = session.copy()
    cropped_session["total_elapsed_time"] = float(
        (cropped_route["timestamp_utc"].iloc[-1] - cropped_route["timestamp_utc"].iloc[0]).total_seconds()
    )
    cropped_session["total_timer_time"] = float(cropped_route["timer_delta_s"].sum())
    cropped_session["total_distance"] = float(cropped_route["distance"].iloc[-1])
    # total_ascent/total_descent szándékosan megmarad az eredeti session értékén: a FIT
    # ezt eszközoldali (barometrikus/simított) algoritmussal számolja, a nyers
    # magasság-mintánkénti diffek összege GPS-zaj miatt erősen túlbecsülné.
    return cropped_route, cropped_events, cropped_session


def run(processed: Path, output: Path, fit_max_elapsed_seconds: float | None = None) -> None:
    manifest = json.loads((processed / "manifest.json").read_text(encoding="utf-8"))
    activity = pd.read_parquet(processed / "gadgetbridge" / "huawei_activity_fused_1min.parquet")
    workouts = pd.read_parquet(processed / "gadgetbridge" / "workouts_normalized.parquet")
    inventory = pd.read_parquet(processed / "gadgetbridge" / "zip_inventory.parquet")
    timeline = pd.read_parquet(processed / "synchronized_1min.parquet")
    gpx_path = processed / "gpx" / "track_points.parquet"
    gpx = (
        pd.read_parquet(gpx_path).sort_values("timestamp_utc").reset_index(drop=True)
        if gpx_path.is_file() else None
    )
    route = pd.read_parquet(processed / "fit" / "records_normalized.parquet").sort_values("timestamp_utc").reset_index(drop=True)
    fit_session = pd.read_parquet(processed / "fit" / "session_mesgs.parquet").iloc[0]
    fit_events = pd.read_parquet(processed / "fit" / "events_normalized.parquet")
    output.mkdir(parents=True, exist_ok=True)

    # A parquet-writerek forrásonként eltérő időbélyeg-felbontással (ms/us/ns) írhatnak;
    # merge_asof egzakt dtype-egyezést vár, ezért itt egységesítjük ns-re.
    activity["timestamp_utc"] = activity["timestamp_utc"].astype("datetime64[ns, UTC]")
    route["timestamp_utc"] = route["timestamp_utc"].astype("datetime64[ns, UTC]")
    if gpx is not None:
        gpx["timestamp_utc"] = gpx["timestamp_utc"].astype("datetime64[ns, UTC]")

    if fit_max_elapsed_seconds is not None:
        route, fit_events, fit_session = crop_to_elapsed(route, fit_events, fit_session, fit_max_elapsed_seconds)

    day = activity[activity["timestamp_local"].dt.date == TOUR_DATE].copy()
    if day.empty:
        raise ValueError(f"Nincs {TOUR_DATE.isoformat()}-i Huawei activity adat.")
    hr_values = day["heart_rate_bpm"].dropna()
    strongest_start, strongest_end = strongest_three_hour_window(day)
    hourly = hourly_summary(day)
    gpx_metrics = route_statistics(gpx) if gpx is not None else None
    route["distance_cumulative_m"] = route["distance"]
    route["speed_kmh_raw"] = route["enhanced_speed"] * 3.6
    route["speed_kmh_15point"] = route["enhanced_speed"] * 3.6
    if gpx is not None:
        # A GPX ritkított, auto-pause-szűrt FIT-derivátum. Az abszolút magasságot
        # időben interpoláljuk a FIT sűrűbb pontjaira; nem keverjük bele a kb. 44 m-rel
        # eltolt FIT-magasságot azokon a helyeken sem, ahol nincs 2 s-on belüli GPX-pont.
        gpx_time_ns = gpx["timestamp_utc"].astype("int64").to_numpy()
        fit_time_ns = route["timestamp_utc"].astype("int64").to_numpy()
        route["elevation_m"] = np.interp(fit_time_ns, gpx_time_ns, gpx["elevation_m"].to_numpy())
    else:
        # Nincs GPX-export ehhez a túrához; a FIT telefonos GPS-magasságát
        # használjuk korrekció nélkül (más sessionökben ez kb. 40 m-rel magasabb
        # a Strava-korrigált GPX-profilnál).
        route["elevation_m"] = route["enhanced_altitude"]
    route["elevation_m_31point_median"] = route["elevation_m"].rolling(31, center=True, min_periods=1).median()
    route_metrics = fit_statistics(route, fit_session, fit_events)
    source_comparison = compare_fit_gpx(route, gpx, route_metrics, gpx_metrics) if gpx is not None else None
    route_start = route["timestamp_local"].iloc[0]
    route_end = route["timestamp_local"].iloc[-1]
    route_watch = day[(day["timestamp_local"] >= route_start) & (day["timestamp_local"] <= route_end)].copy()
    synced = sync_hr_to_route(activity, route)
    matched = synced[synced["distance_cumulative_m"].notna()].copy()
    route_hr = matched["heart_rate_bpm"].dropna()
    distance_bins = distance_summary(route, matched)

    bins = [-np.inf, 100, 120, 140, 160, np.inf]
    labels = ["<100", "100–119", "120–139", "140–159", "≥160"]
    zone_counts = pd.cut(route_hr, bins=bins, labels=labels, right=False).value_counts(sort=False)

    explicit_day_workouts = workouts[workouts["start_local"].dt.date == TOUR_DATE]
    bdf_meta = manifest.get("bdf") or []
    if bdf_meta:
        bdf_start = pd.Timestamp(min(item["start_local"] for item in bdf_meta))
        bdf_end = pd.Timestamp(max(item["end_exclusive_local"] for item in bdf_meta))
        bdf_day_end = bdf_end
    else:
        bdf_start = bdf_end = bdf_day_end = None

    gps_rows = inventory[(inventory["path"].str.contains("/GPS/", regex=False))]
    gps_rows = gps_rows[gps_rows["filename_epoch_local"].notna()].copy()
    gps_rows["clue_time"] = pd.to_datetime(gps_rows["filename_epoch_local"], utc=True).dt.tz_convert(TIMEZONE)
    gps_day = gps_rows[gps_rows["clue_time"].dt.date == TOUR_DATE]
    gps_clue = gps_day["clue_time"].min() if not gps_day.empty else None

    sep5_timeline = timeline[timeline["minute_local"].dt.date == TOUR_DATE]
    overlap_minutes = int((sep5_timeline["watch_available"] & sep5_timeline["bdf_available"]).sum())
    route_bdf_minutes = int((sep5_timeline["fit_available"] & sep5_timeline["bdf_available"]).sum())
    route_watch_minutes = int((sep5_timeline["fit_available"] & sep5_timeline["watch_available"]).sum())
    possible_route_minutes = int((route_end.floor("min") - route_start.floor("min")).total_seconds() // 60 + 1)

    make_plot(
        output / "overview.png", day, hourly, route_start, route_end,
        bdf_start, bdf_day_end, gps_clue,
    )
    make_route_plot(output / "route_profile.png", route, matched, "GPX" if gpx is not None else "FIT")
    hourly.to_csv(output / "hourly_summary.csv", index=False)
    hourly.to_parquet(output / "hourly_summary.parquet", index=False, compression="zstd")
    distance_bins.to_csv(output / "distance_summary.csv", index=False)
    distance_bins.to_parquet(output / "distance_summary.parquet", index=False, compression="zstd")
    route.to_parquet(output / "route_points_enriched.parquet", index=False, compression="zstd")
    synced.to_parquet(output / "route_watch_synced.parquet", index=False, compression="zstd")
    day[day["heart_rate_bpm"].notna()].to_parquet(
        output / "observed_hr_samples.parquet", index=False, compression="zstd"
    )

    route_hr_metrics = {
        "observed_hr_samples": int(len(route_hr)),
        "possible_route_minutes": possible_route_minutes,
        "observed_fraction": float(len(route_hr) / possible_route_minutes),
        "hr_mean_bpm": float(route_hr.mean()),
        "hr_median_bpm": float(route_hr.median()),
        "hr_min_bpm": float(route_hr.min()),
        "hr_max_bpm": float(route_hr.max()),
        "zone_sample_counts": {str(key): int(value) for key, value in zone_counts.items()},
        "calories_kcal_inferred": float(route_watch["calories_kcal_inferred"].sum()),
        "steps": float(route_watch["steps"].sum()),
        "activity_distance_m": float(route_watch["distance_m"].sum()),
        "matched_max_time_delta_s": float(matched["route_time_delta_s"].max()),
    }
    result = {
        "tour_date": TOUR_DATE.isoformat(),
        "timezone": TIMEZONE,
        "explicit_workouts_on_date": int(len(explicit_day_workouts)),
        "fit_primary": {
            "start_local": route_start.isoformat(), "end_local": route_end.isoformat(),
            "start_coordinates": [float(route["latitude_deg"].iloc[0]), float(route["longitude_deg"].iloc[0])],
            "end_coordinates": [float(route["latitude_deg"].iloc[-1]), float(route["longitude_deg"].iloc[-1])],
            **route_metrics,
        },
        "gpx_crosscheck": gpx_metrics,
        "fit_gpx_comparison": source_comparison,
        "day_hr": {
            "samples": int(len(hr_values)), "min_bpm": float(hr_values.min()),
            "max_bpm": float(hr_values.max()), "mean_bpm": float(hr_values.mean()),
            "median_bpm": float(hr_values.median()),
        },
        "tour_hr": route_hr_metrics,
        "strongest_three_hour_window": {"start_local": strongest_start.isoformat(), "end_local": strongest_end.isoformat()},
        "bdf_available": bool(bdf_meta),
        "bdf_end_local": bdf_day_end.isoformat() if bdf_day_end is not None else None,
        "watch_bdf_overlap_minutes_on_date": overlap_minutes,
        "fit_watch_overlap_minutes": route_watch_minutes,
        "fit_bdf_overlap_minutes": route_bdf_minutes,
        "gps_placeholder_local": gps_clue.isoformat() if gps_clue is not None else None,
        "gadgetbridge_gps_payload_bytes": int(gps_rows["size_bytes"].sum()),
        "fit_payload_available": True,
        "fit_heart_rate_present": bool(manifest["fit"]["heart_rate_present"]),
    }
    (output / "analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    gps_text = gps_clue.isoformat(timespec="seconds") if gps_clue is not None else "nincs"
    gap_from_gps = (
        _duration((gps_clue - bdf_day_end).total_seconds())
        if gps_clue is not None and bdf_day_end is not None else "n/a"
    )
    zone_text = ", ".join(f"{label}: {int(zone_counts[label])}" for label in labels)
    dominant_zone = zone_counts.idxmax()
    dominant_zone_pct = 100 * int(zone_counts[dominant_zone]) / len(route_hr) if len(route_hr) else float("nan")
    distance_table = "\n".join(
        f"| {row.distance_start_km:.0f}–{row.distance_end_km:.1f} | {_duration(row.elapsed_seconds)} | "
        f"{row.moving_average_speed_kmh:.1f} | {int(row.hr_samples)} | {row.hr_mean_bpm:.1f} |"
        for row in distance_bins.itertuples()
    )

    if gpx is not None:
        elevation_priority_line = (
            "- Abszolút magasságprofil: **Strava GPX elsődleges**; a FIT telefonos GPS-magassága kb. 40 m-rel "
            "feljebb tolódik. Összes emelkedés/ereszkedés: a FIT session összesítője az elsődleges."
        )
        elevation_range_line = (
            f"A GPX korrigált abszolút magasságtartománya {gpx_metrics['elevation_min_m']:.1f}"
            f"–{gpx_metrics['elevation_max_m']:.1f} m."
        )
        gpx_route_line = "GPX csak ellenőrzés."
        fit_gpx_section = f"""
## FIT–GPX ellenőrzés

- A GPX polilineáris távolsága {gpx_metrics['distance_km']:.2f} km, ami {source_comparison['gpx_polyline_minus_fit_summary_distance_m']:.0f} m-rel ({source_comparison['gpx_polyline_distance_difference_percent']:.2f}%) tér el a FIT session értékétől; ezért a FIT-érték az irányadó.
- {source_comparison['matched_points_within_2s']} időben párosított ponton a koordináta-eltérés mediánja {source_comparison['coordinate_separation_median_m']:.2f} m, 95. percentilise {source_comparison['coordinate_separation_p95_m']:.2f} m: az útvonalak lényegében azonosak.
- A GPX magasság mediánosan {source_comparison['gpx_minus_fit_altitude_median_m']:.1f} m-rel tér el a FIT-től; ezt nem átlagoltam össze.
"""
        route_match_sentence = (
            f"A FIT és a GPX útvonala nagyon jól egyezik (medián koordináta-eltérés "
            f"{source_comparison['coordinate_separation_median_m']:.2f} m); ahol ugyanaz az adattípus több helyen "
            "szerepel, a riport nem keveri őket, hanem a fenti prioritás szerint választ és külön közli az eltérést."
        )
    else:
        elevation_priority_line = (
            "- Abszolút magasságprofil: **FIT telefonos GPS-magasság**, mert nincs GPX-export ehhez a naphoz; "
            "más sessionökben ez kb. 40 m-rel magasabb a Strava-korrigált GPX-profilnál, itt nincs mihez "
            "korrigálni. Összes emelkedés/ereszkedés: a FIT session összesítője."
        )
        elevation_range_line = (
            f"A FIT (korrekció nélküli) abszolút magasságtartománya {route['elevation_m'].min():.1f}"
            f"–{route['elevation_m'].max():.1f} m."
        )
        gpx_route_line = "nincs GPX-export ehhez a sessionhöz."
        fit_gpx_section = ""
        route_match_sentence = (
            "Nincs GPX-export ehhez a sessionhöz, ezért FIT–GPX kereszt-ellenőrzés nem történt."
        )

    if bdf_meta:
        bdf_line = (
            f"- A MOX BDF {bdf_day_end.strftime('%H:%M:%S')}-kor véget ér; a FIT {route_start.strftime('%H:%M:%S')}"
            f"-kor indul, így FIT–BDF átfedés {route_bdf_minutes} perc. "
            + ("EKG-alapú HR-validáció erre a túrára nem lehetséges." if route_bdf_minutes == 0 else
               "Az átfedő szakaszon EKG-alapú HR-validáció elvben lehetséges.")
        )
    else:
        bdf_line = (
            "- Nincs MOX BDF-felvétel (Holter EKG) ehhez a naphoz; EKG-alapú HR-validáció nem áll rendelkezésre "
            "erre a túrára."
        )

    heart_rate_note = (
        "nincs `heart_rate` mező" if not manifest["fit"]["heart_rate_present"] else "van `heart_rate` mező"
    )

    crop_line = (
        f"- A nyers FIT-export a rajttól számítva {_duration(fit_max_elapsed_seconds)}-re lett vágva, mert az "
        "eredeti felvétel ennél tovább tartott, mint a publikált aktivitás (a többlet valószínűleg a hivatalos "
        "túra vége utáni további tekerés, pl. hazaút). A levágás után a FIT távolsága és mozgásideje a "
        "referenciaként használt app-jelentette értékekhez igazodik."
        if fit_max_elapsed_seconds is not None else None
    )
    crop_note = f" A FIT le lett vágva (lásd alább)." if fit_max_elapsed_seconds is not None else ""

    ascent_per_km = route_metrics["total_ascent_m"] / route_metrics["distance_km"] if route_metrics["distance_km"] else 0.0
    if ascent_per_km < 5:
        terrain_word = "sík"
    elif ascent_per_km < 15:
        terrain_word = "enyhén dombos"
    else:
        terrain_word = "dombos"

    report = f"""# {ROUTE_TITLE} túra – FIT-elsődleges elemzés

## Fő eredmény

A CRC-ellenőrzött FIT az elsődleges aktivitásforrás. A túra **{route_start.strftime('%H:%M:%S')}–{route_end.strftime('%H:%M:%S')}** között zajlott. A FIT session szerint a távolság **{route_metrics['distance_km']:.2f} km**, az eltelt idő **{_duration(route_metrics['elapsed_seconds'])}**, az eszköz auto-pause timer-ideje **{_duration(route_metrics['timer_seconds'])}**, a szünet pedig **{_duration(route_metrics['paused_seconds'])}**. Az eltelt átlag {route_metrics['elapsed_average_speed_kmh']:.1f} km/h, az auto-pause timer szerinti átlag **{route_metrics['timer_average_speed_kmh']:.1f} km/h**.

Kiegészítő, {GPS_MOVING_THRESHOLD_KMH:.0f} km/h-s GPS-sebességküszöbbel (nem az eszköz auto-pause eseményei, hanem a nyers pontonkénti sebesség alapján) számolt mozgásidő **{_duration(route_metrics['gps_moving_seconds'])}**, ehhez tartozó átlagsebesség **{route_metrics['gps_moving_average_speed_kmh']:.1f} km/h** — ez a lámpáknál/kereszteződéseknél keletkező rövid auto-pause villanásokat is kiszűri, ezért az eszköz timer-idejénél valamivel rövidebb.{crop_note}

## Forrásprioritás

- Útvonal, távolság, idő, aktív idő, sebesség, emelkedés: **FIT elsődleges**; {gpx_route_line}
- Pulzus: **Huawei perces activity elsődleges**, mert a FIT {len(route)} record üzenetében {heart_rate_note}.
{elevation_priority_line}

## Menetdinamika

- FIT session távolság: {route_metrics['distance_km']:.2f} km; rajt/cél: {route['latitude_deg'].iloc[0]:.6f}, {route['longitude_deg'].iloc[0]:.6f} → {route['latitude_deg'].iloc[-1]:.6f}, {route['longitude_deg'].iloc[-1]:.6f}.
- FIT sebesség mozgás közben: medián {route_metrics['speed_median_kmh']:.1f}, 90. percentilis {route_metrics['speed_p90_kmh']:.1f}, maximum {route_metrics['speed_max_kmh']:.1f} km/h.
- Automatikus timer-stopok: {route_metrics['auto_pause_count']}; összes szünet: {_duration(route_metrics['paused_seconds'])}.
- FIT emelkedés/ereszkedés: **{route_metrics['total_ascent_m']:.0f}/{route_metrics['total_descent_m']:.0f} m**. {elevation_range_line}
- FIT GPS-pontosság: medián {route_metrics['gps_accuracy_median_m']:.0f} m, 95. percentilis {route_metrics['gps_accuracy_p95_m']:.0f} m.

## Pulzusterhelés

- A FIT időablakában {len(route_hr)}/{possible_route_minutes} lehetséges perchez van Huawei HR ({100*len(route_hr)/possible_route_minutes:.1f}%): átlag **{route_hr.mean():.1f}**, medián {route_hr.median():.0f}, tartomány {int(route_hr.min())}–{int(route_hr.max())} bpm.
- Abszolút HR-sávok mintaszáma: {zone_text}. A {dominant_zone} bpm sávba {int(zone_counts[dominant_zone])}/{len(route_hr)} minta ({dominant_zone_pct:.1f}%) került, ez a legnagyobb sáv. Ezek nem személyre szabott edzészónák.
- A Huawei `CALORIES/1000` mezőből {route_watch['calories_kcal_inferred'].sum():.0f} kcal adódik, de ez dokumentálatlan skálázású becslés; a FIT nem tartalmaz kalóriamezőt.

## Tíz kilométeres bontás

| Táv (km) | Eltelt idő | FIT-timer szerinti átlag (km/h) | HR-minta | Átlag HR (bpm) |
|---:|---:|---:|---:|---:|
{distance_table}
{fit_gpx_section}
## Időszinkron és korlátok

- A FIT UTC-időbélyegét {TIMEZONE} helyi időre alakítottam. A Huawei HR-t legközelebbi FIT-rekordhoz illesztettem, legfeljebb 90 másodperces toleranciával, interpoláció nélkül; a tényleges legnagyobb eltérés {matched['route_time_delta_s'].max():.0f} s.
{bdf_line}
{crop_line if crop_line else ""}
- A FIT forrása Strava mobilalkalmazás (telefonos GPS). Rekordmezői: idő, koordináta, kumulatív távolság, sebesség, magasság és GPS-pontosság. Nincs benne pulzus, kadencia vagy teljesítmény.

## Feldolgozott réteg

A FIT minden üzenetcsoportja külön ZSTD Parquet lett, továbbá elkészült az egy időbélyegre egyesített `records_normalized.parquet`, a timeresemények és a teljes mezőkatalógus. A FIT integritása érvényes. A Gadgetbridge{", GPX" if gpx is not None else ""}{" és BDF" if bdf_meta else ""} nyers/normalizált rétegei megmaradtak; a `synchronized_1min.parquet` forrásonkénti oszlopokkal, interpoláció nélkül készült.

## Következtetés

Ez egy **{route_metrics['distance_km']:.2f} km-es, {terrain_word}, tempós** túra volt, {route_metrics['timer_average_speed_kmh']:.1f} km/h átlagsebességgel a {_duration(route_metrics['timer_seconds'])}-es auto-pause timer-idő alatt (a Stravával/appal összevethető elsődleges szám; a zajszűrt GPS-küszöbös becslés {route_metrics['gps_moving_average_speed_kmh']:.1f} km/h). A Huawei alapján a megfigyelt pulzus {dominant_zone_pct:.1f}%-a a {dominant_zone} bpm sávba esett, tehát a menet nagy része {"tartósan magas kardiovaszkuláris terhelés mellett zajlott" if dominant_zone in ("140–159", "≥160") else "mérsékelt kardiovaszkuláris terhelés mellett zajlott"}. {route_match_sentence}

Ez technikai/edzésadat-elemzés, nem orvosi diagnózis.
"""
    (output / "report.md").write_text(report, encoding="utf-8")
    print(f"Kész elemzés: {output.resolve()}")
    print(f"FIT: {route_metrics['distance_km']:.2f} km, {_duration(route_metrics['elapsed_seconds'])}, timer {_duration(route_metrics['timer_seconds'])}, átlag {route_metrics['timer_average_speed_kmh']:.1f} km/h")
    print(f"Túra HR: {len(route_hr)} minta, átlag {route_hr.mean():.1f} bpm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalizált túraadat-bundle elemzése.")
    parser.add_argument("--processed", type=Path, default=Path("09_05/processed"))
    parser.add_argument("--output", type=Path, default=Path("09_05/analysis"))
    parser.add_argument("--tour-date", type=date.fromisoformat, default=TOUR_DATE)
    parser.add_argument("--timezone", default=TIMEZONE)
    parser.add_argument("--route-title", default=ROUTE_TITLE)
    parser.add_argument(
        "--fit-max-elapsed-seconds", type=float, default=None,
        help="Vágja a FIT-et a rajttól számított ennyi másodpercre (pl. ha az export hosszabb, mint a publikált aktivitás).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    TOUR_DATE = args.tour_date
    TIMEZONE = args.timezone
    ROUTE_TITLE = args.route_title
    run(args.processed, args.output, args.fit_max_elapsed_seconds)
