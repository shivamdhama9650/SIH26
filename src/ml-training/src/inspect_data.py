"""
inspect_data.py
================
Stage: RAW DATA -> INSPECT

Prints a full diagnostic report for a dataset: dims, coords, data_vars,
shapes, units, min/max, time range, lat/lon range, spatial resolution,
temporal frequency, and (if present) depth levels.

Never assumes coordinate names - uses coord_utils to detect them first.
Run this BEFORE anything else so you know what you actually have.
"""

import numpy as np
import pandas as pd
import xarray as xr

from src.coord_utils import detect_coord_name


def _infer_spacing(values):
    """Median spacing of a monotonic 1D coordinate array."""
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return None
    diffs = np.diff(np.sort(values))
    return float(np.median(diffs))


def _infer_temporal_freq(time_values):
    """Median time spacing in days (float)."""
    if len(time_values) < 2:
        return None
    t = pd.to_datetime(time_values)
    diffs = np.diff(np.sort(t.values)).astype("timedelta64[s]").astype(float) / 86400.0
    return float(np.median(diffs))


def inspect_dataset(ds: xr.Dataset, name: str = "dataset") -> dict:
    """
    Prints a human-readable report and returns a dict summary that other
    stages can use programmatically (e.g. to decide temporal alignment).
    """
    print("=" * 70)
    print(f"INSPECTING: {name}")
    print("=" * 70)

    print(f"\nDimensions: {dict(ds.dims)}")
    print(f"Coordinates: {list(ds.coords)}")
    print(f"Data variables: {list(ds.data_vars)}")

    summary = {"name": name, "data_vars": list(ds.data_vars)}

    # --- variable-level detail ---
    for v in ds.data_vars:
        da = ds[v]
        units = da.attrs.get("units", "unknown")
        try:
            import dask
            is_chunked = hasattr(da, "chunks") and da.chunks
            if is_chunked:
                # Skip .min()/.max() for chunked Dask arrays — would load all data
                vmin, vmax = "(dask — skipped)", "(dask — skipped)"
            else:
                vmin = float(da.min(skipna=True).values)
                vmax = float(da.max(skipna=True).values)
        except Exception:
            vmin, vmax = None, None
        print(f"  - {v}: shape={da.shape}, dims={da.dims}, units={units}, "
              f"min={vmin}, max={vmax}")

    # --- lat ---
    lat_name = detect_coord_name(ds, "lat", required=False)
    if lat_name:
        lat_vals = ds[lat_name].values
        lat_res = _infer_spacing(lat_vals)
        print(f"\nLatitude coord: '{lat_name}' | range=[{lat_vals.min():.3f}, "
              f"{lat_vals.max():.3f}] | resolution~{lat_res}")
        summary["lat_range"] = (float(lat_vals.min()), float(lat_vals.max()))
        summary["lat_resolution"] = lat_res
    else:
        print("\nLatitude coord: NOT FOUND")

    # --- lon ---
    lon_name = detect_coord_name(ds, "lon", required=False)
    if lon_name:
        lon_vals = ds[lon_name].values
        lon_res = _infer_spacing(lon_vals)
        print(f"Longitude coord: '{lon_name}' | range=[{lon_vals.min():.3f}, "
              f"{lon_vals.max():.3f}] | resolution~{lon_res}")
        summary["lon_range"] = (float(lon_vals.min()), float(lon_vals.max()))
        summary["lon_resolution"] = lon_res
    else:
        print("Longitude coord: NOT FOUND")

    # --- time ---
    time_name = detect_coord_name(ds, "time", required=False)
    if time_name:
        time_vals = ds[time_name].values
        freq_days = _infer_temporal_freq(time_vals)
        print(f"Time coord: '{time_name}' | range=[{pd.to_datetime(time_vals.min())}, "
              f"{pd.to_datetime(time_vals.max())}] | n={len(time_vals)} | "
              f"median spacing~{freq_days} days")
        summary["time_range"] = (str(pd.to_datetime(time_vals.min())),
                                  str(pd.to_datetime(time_vals.max())))
        summary["n_time_steps"] = len(time_vals)
        summary["temporal_freq_days"] = freq_days
    else:
        print("Time coord: NOT FOUND")

    # --- depth (GLORYS only, usually) ---
    depth_name = detect_coord_name(ds, "depth", required=False)
    if depth_name:
        depth_vals = ds[depth_name].values
        print(f"Depth coord: '{depth_name}' | n_levels={len(depth_vals)} | "
              f"levels={np.round(depth_vals, 2).tolist()}")
        summary["depth_levels"] = depth_vals.tolist()
    else:
        print("Depth coord: not present (expected for surface-only products)")

    print()
    return summary


def inspect_all(datasets: dict) -> dict:
    """
    datasets: {name: xr.Dataset}
    Returns {name: summary_dict}
    """
    summaries = {}
    for name, ds in datasets.items():
        if ds is None:
            print(f"[SKIP] '{name}' is not available (see loaders.py output above).")
            continue
        summaries[name] = inspect_dataset(ds, name=name)
    return summaries
