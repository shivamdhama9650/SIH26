"""
assemble.py
===========
Stage: MERGE INPUTS + TARGETS -> SAVE ML-READY DATASET

Builds:
  inputs: (time, lat, lon, variable)   -- surface variables, normalized
  target: (time, depth, lat, lon)      -- GLORYS temperature, normalized per depth

Also saves: split metadata, normalization stats, grid coords, target
depths, and the config actually used, so the whole thing is reproducible.
"""

import json
from pathlib import Path

import numpy as np
import xarray as xr

from config import TARGET_DEPTHS, GRID_RESOLUTION_DEG, REGION, SPLIT_CONFIG


def build_input_stack(surface_vars: dict) -> xr.Dataset:
    """
    surface_vars: {name: xr.DataArray} each (time, lat, lon), already
    regridded/aligned/masked/normalized. Missing-and-optional variables
    (e.g. SSS=None) are simply excluded from the stack.
    channel count is therefore determined at RUNTIME, never hard-coded.
    """
    available = {k: v for k, v in surface_vars.items() if v is not None}
    print(f"[assemble] Input channels actually assembled ({len(available)}): "
          f"{list(available.keys())}")
    return xr.Dataset(available)


def build_ml_ready_dataset(input_ds: xr.Dataset, target_da: xr.DataArray,
                            ocean_mask: xr.DataArray) -> xr.Dataset:
    ml_ready = input_ds.copy()
    ml_ready["temperature_target"] = target_da
    ml_ready["ocean_mask"] = ocean_mask
    return ml_ready


def save_ml_ready(ml_ready: xr.Dataset, out_dir: Path, filename="ml_ready_dataset.nc"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    encoding = {var: {"zlib": True, "complevel": 4} for var in ml_ready.data_vars}
    ml_ready.to_netcdf(out_path, encoding=encoding)
    print(f"[assemble] Saved ML-ready dataset -> {out_path} "
          f"({out_path.stat().st_size / 1e6:.1f} MB)")
    return out_path


def save_metadata(out_dir: Path, splits: dict, channels_used: list,
                   config_snapshot: dict):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "region": REGION,
        "grid_resolution_deg": GRID_RESOLUTION_DEG,
        "target_depths": TARGET_DEPTHS,
        "channels_used": channels_used,
        "n_channels": len(channels_used),
        "split_ranges": {
            name: {"start": str(idx.min().date()), "end": str(idx.max().date()),
                   "n_days": int(len(idx))}
            for name, idx in splits.items()
        },
        "split_config_used": SPLIT_CONFIG,
    }
    meta_path = out_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"[assemble] Saved metadata -> {meta_path}")
    return meta_path
