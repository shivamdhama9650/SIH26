"""
masking.py
==========
Stage: LAND/SEA MASKING (before any 3x3 patch extraction)

Priority order:
  1. Use an explicit land/sea mask variable if one exists in any raw
     dataset (checked via coord_utils.detect_var_name on 'land_sea_mask').
  2. Otherwise, build the simplest defensible fallback mask: a grid cell
     is OCEAN if SST (surface) is finite for at least
     config.MASK_FALLBACK_MIN_VALID_FRACTION of the days in the
     TRAINING period. This assumption is printed explicitly.

The returned mask is a static 2D (lat, lon) boolean DataArray:
True = usable ocean cell, False = land/invalid.
"""

import numpy as np
import xarray as xr

from config import MASK_FALLBACK_MIN_VALID_FRACTION
from src.coord_utils import detect_var_name


def find_explicit_mask(datasets: dict):
    for name, ds in datasets.items():
        if ds is None:
            continue
        mask_var = detect_var_name(ds, "land_sea_mask", required=False)
        if mask_var is not None:
            print(f"[masking] Found explicit land/sea mask variable "
                  f"'{mask_var}' in '{name}'. Using it.")
            raw = ds[mask_var]
            # Convention varies: some masks are 1=ocean, some 1=land.
            # We assume 1=ocean/valid, 0=land, which is the most common
            # convention (e.g. CMEMS 'mask' variables). VERIFY REQUIRED
            # if your product uses the opposite convention.
            print("[masking] ASSUMPTION: mask convention is 1=ocean, 0=land. "
                  "VERIFY REQUIRED if your source uses the opposite.")
            ocean_mask = raw.astype(bool)
            if "time" in ocean_mask.dims:
                ocean_mask = ocean_mask.isel(time=0, drop=True)
            return ocean_mask
    return None


def build_fallback_mask(sst_da: xr.DataArray, train_time_slice) -> xr.DataArray:
    """
    sst_da: (time, lat, lon) DataArray, already regridded/aligned.
    train_time_slice: a slice or boolean time selector restricting to
    the training period ONLY (masking must not peek at val/test, though
    for a purely geographic land/sea mask this mostly matters for
    consistency/reproducibility rather than leakage).
    """
    print(f"[masking] No explicit mask found anywhere. Building FALLBACK mask "
          f"from SST validity over the training period. ASSUMPTION: a cell "
          f"is OCEAN if SST is non-NaN for >= "
          f"{MASK_FALLBACK_MIN_VALID_FRACTION*100:.0f}% of training days.")

    sst_train = sst_da.sel(time=train_time_slice)
    valid_fraction = sst_train.notnull().mean(dim="time")
    ocean_mask = valid_fraction >= MASK_FALLBACK_MIN_VALID_FRACTION
    n_ocean = int(ocean_mask.sum().values)
    n_total = int(ocean_mask.size)
    print(f"[masking] Fallback mask: {n_ocean}/{n_total} cells "
          f"({100*n_ocean/n_total:.1f}%) classified as ocean.")
    return ocean_mask


def apply_mask(da: xr.DataArray, ocean_mask: xr.DataArray) -> xr.DataArray:
    """Sets non-ocean cells to NaN across all time/depth for a given array."""
    return da.where(ocean_mask)


def get_land_sea_mask(datasets: dict, sst_da: xr.DataArray, train_time_slice) -> xr.DataArray:
    mask = find_explicit_mask(datasets)
    if mask is None:
        mask = build_fallback_mask(sst_da, train_time_slice)
    return mask
