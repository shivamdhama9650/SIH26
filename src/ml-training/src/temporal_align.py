"""
temporal_align.py
==================
Stage: ALIGN TIME

Rule from the brief: inspect actual temporal resolution first, then align
to a common daily time coordinate. Do NOT interpolate time unless
justified.

Logic (explicit, not magic):
  - If a dataset is already daily (median spacing ~1 day): just reindex
    onto the common daily calendar built from the intersection of all
    datasets' time ranges, using exact-match reindex (no interpolation).
    Missing days become NaN and are reported.
  - If a dataset is sub-daily (e.g. hourly): resample with daily mean.
  - If a dataset is coarser than daily (e.g. weekly/monthly, common for
    some altimetry products): forward-fill onto the daily calendar and
    print a clear warning - this is a documented assumption, not a
    silent decision.
"""

import numpy as np
import pandas as pd
import xarray as xr


def _median_spacing_days(time_index: pd.DatetimeIndex) -> float:
    if len(time_index) < 2:
        return np.nan
    diffs = np.diff(np.sort(time_index.values)).astype("timedelta64[s]").astype(float) / 86400.0
    return float(np.median(diffs))


def build_common_daily_calendar(datasets: dict) -> pd.DatetimeIndex:
    """Intersection of [min,max] time range across all AVAILABLE datasets,
    at daily frequency."""
    starts, ends = [], []
    for name, ds in datasets.items():
        if ds is None or "time" not in ds.coords:
            continue
        t = pd.to_datetime(ds["time"].values)
        starts.append(t.min())
        ends.append(t.max())

    if not starts:
        raise ValueError("[temporal_align] No dataset has a usable time coordinate.")

    common_start, common_end = max(starts), min(ends)
    if common_start > common_end:
        raise ValueError(
            f"[temporal_align] Datasets do not overlap in time! "
            f"Latest common start={common_start}, earliest common end={common_end}. "
            f"VERIFY REQUIRED: check download date ranges for each source."
        )
    calendar = pd.date_range(common_start.normalize(), common_end.normalize(), freq="D")
    print(f"[temporal_align] Common daily calendar: {calendar[0].date()} -> "
          f"{calendar[-1].date()} ({len(calendar)} days)")
    return calendar


def align_to_daily(ds: xr.Dataset, calendar: pd.DatetimeIndex, name: str = "dataset") -> xr.Dataset:
    if ds is None:
        return None
    if "time" not in ds.coords:
        raise ValueError(f"[temporal_align] '{name}' has no time coordinate.")

    t = pd.to_datetime(ds["time"].values)
    spacing = _median_spacing_days(t)
    print(f"[temporal_align] '{name}' native spacing ~{spacing:.3f} days.", flush=True)

    if spacing < 0.9:
        print(f"[temporal_align] '{name}' is sub-daily -> resampling to daily mean.", flush=True)
        ds = ds.resample(time="1D").mean(skipna=True)
        aligned = ds.reindex(time=calendar)
    elif 0.9 <= spacing <= 1.1:
        # Normalize any sub-daily offsets (e.g. noon timestamps) to midnight
        # before exact reindex — otherwise all rows become NaN.
        t_mid = ds.time.dt.floor("D")
        if bool((t_mid.values != ds["time"].values).any()):
            print(f"[temporal_align] '{name}': normalizing timestamps to midnight "
                  f"(found time-of-day offset, e.g. 12:00 UTC).", flush=True)
            ds = ds.assign_coords(time=t_mid)
        print(f"[temporal_align] '{name}' is already daily -> reindexing onto common "
              f"calendar (no interpolation)...", flush=True)
        aligned = ds.reindex(time=calendar)
    else:
        print(f"[temporal_align] '{name}' is COARSER than daily (~{spacing:.1f} d). "
              f"ASSUMPTION: forward-filling onto the daily calendar. "
              f"This is a documented approximation, not real daily data.", flush=True)
        aligned = ds.reindex(time=calendar, method="ffill")

    # Cheap missing-day count (no full data scan)
    n_missing = len(calendar) - int(pd.DatetimeIndex(t.normalize().unique()).isin(calendar).sum())
    print(f"[temporal_align] '{name}' aligned: {aligned.sizes.get('time', 0)} days. "
          f"Missing days (NaN-filled): {n_missing}", flush=True)
    return aligned



def align_all(datasets: dict) -> dict:
    calendar = build_common_daily_calendar(datasets)
    aligned = {}
    for name, ds in datasets.items():
        if name == "glorys":
            continue  # GLORYS handled separately (has extra depth dim)
        aligned[name] = align_to_daily(ds, calendar, name=name) if ds is not None else None
    return aligned, calendar
