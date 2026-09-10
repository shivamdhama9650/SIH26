"""
argo_matching.py
==================
Stage: STAGE 4 — ARGO independent validation.

For each QC-passed, depth-interpolated Argo profile, find the model's
surface input at the same place/time:
  1. nearest grid (lat, lon) cell
  2. nearest available day within a configurable temporal window
  3. verify the actual (great-circle distance, time difference) is
     within configurable tolerances - reject the match otherwise
  4. verify the CNN's own 3x3-patch validity rule holds at that exact
     (time, lat, lon) - land/missing-data patches can never be matched,
     exactly like they can never be a training sample.

This is deliberately NOT "just nearest": tolerances are explicit,
configurable (argo_config.py), and every match's actual distance/time
offset is recorded for the diagnostics file - see match_profiles().
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import argo_config
from src.cnn_dataset import _OFFSETS

_EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance in km. Vectorized (numpy broadcasting)."""
    lat1r, lon1r, lat2r, lon2r = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return _EARTH_RADIUS_KM * c


def _nearest_index(value: float, grid: np.ndarray):
    idx = int(np.abs(grid - value).argmin())
    return idx, grid[idx]


def _nearest_time_within_window(target_time: pd.Timestamp, time_grid: pd.DatetimeIndex,
                                 tolerance_days: int):
    """Returns (matched_timestamp, day_offset, grid_idx) for the CLOSEST
    day within +/- tolerance_days, or (None, None, None) if none exists
    in that window (e.g. the profile predates/postdates the whole dataset)."""
    window = time_grid[(time_grid >= target_time - pd.Timedelta(days=tolerance_days)) &
                        (time_grid <= target_time + pd.Timedelta(days=tolerance_days))]
    if len(window) == 0:
        return None, None, None
    offsets = np.abs((window - target_time).days)
    best = int(np.argmin(offsets))
    matched_time = window[best]
    day_offset = int((matched_time - target_time).days)
    grid_idx = int(time_grid.get_loc(matched_time))
    return matched_time, day_offset, grid_idx


def match_profiles(argo: dict, ds: xr.Dataset,
                    temporal_tolerance_days: int = None,
                    spatial_tolerance_km: float = None) -> pd.DataFrame:
    """
    argo: output of argo_processing.process_all_profiles() (needs lat,
        lon, time, n_profiles).
    ds: the ml_ready_dataset.nc Dataset (needs lat, lon, time coords).

    Returns a DataFrame, one row per INPUT profile (matched or not),
    with columns:
        argo_idx, argo_lat, argo_lon, argo_time,
        matched, reject_reason,
        grid_lat, grid_lon, grid_time, lat_idx, lon_idx, time_idx,
        distance_km, time_offset_days

    matched=False rows keep grid_lat/grid_time etc. as NaN/NaT and
    reject_reason explains why (out of temporal window / too far
    spatially / not a valid land-free patch that day).
    """
    if temporal_tolerance_days is None:
        temporal_tolerance_days = argo_config.ARGO_TEMPORAL_TOLERANCE_DAYS
    if spatial_tolerance_km is None:
        spatial_tolerance_km = argo_config.ARGO_SPATIAL_TOLERANCE_KM

    lat_grid = ds["lat"].values
    lon_grid = ds["lon"].values
    time_grid = pd.DatetimeIndex(ds["time"].values)
    ocean_mask = ds["ocean_mask"].values  # (lat, lon) bool, static

    n_lat, n_lon = ocean_mask.shape
    half = 1  # PATCH_SIZE=3 -> half=1, locked architecture; verified again below

    rows = []
    for i in range(argo["n_profiles"]):
        a_lat = float(argo["lat"][i])
        a_lon = float(argo["lon"][i])
        a_time = pd.Timestamp(argo["time"][i])

        lat_idx, g_lat = _nearest_index(a_lat, lat_grid)
        lon_idx, g_lon = _nearest_index(a_lon, lon_grid)
        distance_km = float(haversine_km(a_lat, a_lon, g_lat, g_lon))

        matched_time, day_offset, time_idx = _nearest_time_within_window(
            a_time, time_grid, temporal_tolerance_days
        )

        row = {
            "argo_idx": i, "argo_lat": a_lat, "argo_lon": a_lon, "argo_time": a_time,
            "matched": False, "reject_reason": None,
            "grid_lat": np.nan, "grid_lon": np.nan, "grid_time": pd.NaT,
            "lat_idx": np.nan, "lon_idx": np.nan, "time_idx": np.nan,
            "distance_km": distance_km, "time_offset_days": np.nan,
        }

        if matched_time is None:
            row["reject_reason"] = (
                f"no grid day within +/-{temporal_tolerance_days}d of {a_time.date()}"
            )
            rows.append(row)
            continue

        if distance_km > spatial_tolerance_km:
            row["reject_reason"] = (
                f"nearest grid cell {distance_km:.1f} km away "
                f"(tolerance {spatial_tolerance_km} km)"
            )
            row["grid_time"] = matched_time
            row["time_offset_days"] = day_offset
            rows.append(row)
            continue

        # edge-of-grid / land / patch-validity check: the FULL 3x3
        # neighborhood must be inside the grid and ocean (static check,
        # same rule the CNN was trained under).
        if not (half <= lat_idx < n_lat - half and half <= lon_idx < n_lon - half):
            row["reject_reason"] = "matched grid cell is on the domain edge (no full 3x3 patch)"
            row["grid_time"] = matched_time
            row["time_offset_days"] = day_offset
            rows.append(row)
            continue

        patch_ocean = ocean_mask[lat_idx - half:lat_idx + half + 1,
                                  lon_idx - half:lon_idx + half + 1]
        if not patch_ocean.all():
            row["reject_reason"] = "matched grid cell's 3x3 neighborhood touches land"
            row["grid_time"] = matched_time
            row["time_offset_days"] = day_offset
            rows.append(row)
            continue

        row.update({
            "matched": True, "reject_reason": None,
            "grid_lat": g_lat, "grid_lon": g_lon, "grid_time": matched_time,
            "lat_idx": lat_idx, "lon_idx": lon_idx, "time_idx": time_idx,
            "time_offset_days": day_offset,
        })
        rows.append(row)

    df = pd.DataFrame(rows)
    n_matched = int(df["matched"].sum())
    print(f"[argo_matching] {n_matched}/{len(df)} profiles matched within "
          f"tolerances (temporal=+/-{temporal_tolerance_days}d, "
          f"spatial<={spatial_tolerance_km}km). Rejection reasons:")
    if n_matched < len(df):
        print(df.loc[~df["matched"], "reject_reason"].value_counts().to_string())
    return df


def extract_patches_for_matches(ds: xr.Dataset, channels: list, matches: pd.DataFrame,
                                 patch_size: int = 3):
    """
    For every MATCHED row, extract the exact 3x3xC input patch (from the
    same normalized ml_ready_dataset.nc the CNN trained on - so channel
    ordering/normalization/preprocessing are identical by construction)
    AND do the final time-varying finiteness check (a statically-ocean
    patch can still have a transient per-day gap).

    Returns:
        X: (n_final_valid, n_channels, patch_size, patch_size) float32
        final_valid_argo_idx: (n_final_valid,) the `argo_idx` values (into
            the ORIGINAL profile list) these patches correspond to, in order
        matches: the input DataFrame with an added "patch_valid" column
            (True only for rows that survived this final finiteness check)
    """
    half = patch_size // 2
    matches = matches.copy()
    matches["patch_valid"] = False

    matched_rows = matches[matches["matched"]]
    if len(matched_rows) == 0:
        return np.empty((0, len(channels), patch_size, patch_size), dtype="float32"), \
               np.empty((0,), dtype=int), matches

    time_idx_arr = matched_rows["time_idx"].values.astype(int)
    lat_idx_arr = matched_rows["lat_idx"].values.astype(int)
    lon_idx_arr = matched_rows["lon_idx"].values.astype(int)
    argo_idx_arr = matched_rows["argo_idx"].values.astype(int)

    n = len(matched_rows)
    X = np.empty((n, len(channels), patch_size, patch_size), dtype="float32")
    point_valid = np.ones(n, dtype=bool)

    for c, ch in enumerate(channels):
        arr = ds[ch].values  # (time, lat, lon) - normalized already
        for di, dj in _OFFSETS:
            vals = arr[time_idx_arr, lat_idx_arr + di, lon_idx_arr + dj]
            row_in_patch = di + half
            col_in_patch = dj + half
            X[:, c, row_in_patch, col_in_patch] = vals
            point_valid &= np.isfinite(vals)

    X_valid = X[point_valid]
    argo_idx_valid = argo_idx_arr[point_valid]

    matches.loc[matches["argo_idx"].isin(argo_idx_valid), "patch_valid"] = True
    n_dropped = int((~point_valid).sum())
    if n_dropped:
        print(f"[argo_matching] {n_dropped} otherwise-matched profile(s) "
              f"dropped at the final finiteness check (transient per-day "
              f"NaN in at least one channel's 3x3 patch on the matched day).")

    return X_valid, argo_idx_valid, matches
