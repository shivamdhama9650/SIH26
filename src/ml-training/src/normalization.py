"""
normalization.py
=================
Stage: NORMALIZATION (train-data-only, per-depth for targets)

Rules from the brief (non-negotiable):
  - mean/std computed from the TRAIN split ONLY, never val/test, never
    the full dataset.
  - surface inputs: one mean/std per variable.
  - GLORYS temperature target: one mean/std PER DEPTH LEVEL (15 separate
    stats), because deep water behaves very differently from the surface.
  - stats are saved to disk (JSON) so the exact same numbers can be
    reused at inference time later.
"""

import json

import numpy as np
import xarray as xr


def compute_input_stats(input_ds: xr.Dataset, train_time_index) -> dict:
    """
    input_ds: Dataset with each surface variable as a data_var, dims
    (time, lat, lon).
    Returns {var_name: {"mean": float, "std": float}}
    """
    stats = {}
    train_subset = input_ds.sel(time=train_time_index)
    for var in input_ds.data_vars:
        mean = float(train_subset[var].mean(skipna=True).values)
        std = float(train_subset[var].std(skipna=True).values)
        if std == 0 or np.isnan(std):
            print(f"[normalization] WARNING: '{var}' has std={std} on the "
                  f"training split - check for a constant/degenerate field.")
            std = std if std not in (0, np.nan) else 1.0
        stats[var] = {"mean": mean, "std": std}
        print(f"[normalization] input '{var}': train mean={mean:.4f}, "
              f"std={std:.4f}")
    return stats


def normalize_inputs(input_ds: xr.Dataset, stats: dict) -> xr.Dataset:
    normalized = input_ds.copy()
    for var, s in stats.items():
        normalized[var] = (input_ds[var] - s["mean"]) / s["std"]
    return normalized


def compute_target_stats_per_depth(target_da: xr.DataArray, train_time_index) -> dict:
    """
    target_da: (time, depth, lat, lon) temperature DataArray.
    Returns {depth_value(str): {"mean": float, "std": float}}
    """
    train_subset = target_da.sel(time=train_time_index)
    stats = {}
    for depth_val in target_da["depth"].values:
        d_slice = train_subset.sel(depth=depth_val)
        mean = float(d_slice.mean(skipna=True).values)
        std = float(d_slice.std(skipna=True).values)
        if std == 0 or np.isnan(std):
            print(f"[normalization] WARNING: depth={depth_val}m has std={std} "
                  f"on training split.")
            std = std if std not in (0, np.nan) else 1.0
        stats[str(float(depth_val))] = {"mean": mean, "std": std}
        print(f"[normalization] target depth={depth_val}m: train mean={mean:.4f}, "
              f"std={std:.4f}")
    return stats


def normalize_target_per_depth(target_da: xr.DataArray, stats: dict) -> xr.DataArray:
    normalized = target_da.copy()
    for depth_val in target_da["depth"].values:
        key = str(float(depth_val))
        s = stats[key]
        sel = dict(depth=depth_val)
        normalized.loc[sel] = (target_da.sel(depth=depth_val) - s["mean"]) / s["std"]
    return normalized


def save_stats(input_stats: dict, target_stats: dict, out_path: str):
    payload = {"input_stats": input_stats, "target_stats_per_depth": target_stats}
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[normalization] Saved normalization statistics -> {out_path}")
