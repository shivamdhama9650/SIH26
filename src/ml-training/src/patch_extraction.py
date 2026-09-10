"""
patch_extraction.py
====================
Stage: PATCH EXTRACTION (prep only - reusable function, no CNN yet)

Given the assembled input tensor (time, lat, lon, variable) and the
static land/sea mask, this finds all VALID center points: ocean cells
whose full 3x3 neighborhood is entirely ocean (no land/masked cell
anywhere in the patch, including the border). This guarantees no patch
fed to the future CNN ever touches land.
"""

import numpy as np
import xarray as xr

from config import PATCH_SIZE


def find_valid_patch_centers(ocean_mask: xr.DataArray, patch_size: int = PATCH_SIZE):
    """
    ocean_mask: static (lat, lon) boolean DataArray, True=ocean.
    Returns a boolean (lat, lon) DataArray: True where a full patch_size x
    patch_size neighborhood centered here is entirely ocean AND entirely
    within grid bounds.
    """
    if patch_size % 2 == 0:
        raise ValueError("[patch_extraction] patch_size must be odd (e.g. 3, 5, 9).")
    half = patch_size // 2

    mask_vals = ocean_mask.values.astype(bool)
    n_lat, n_lon = mask_vals.shape

    valid = np.zeros_like(mask_vals, dtype=bool)
    # Vectorized neighborhood-all-ocean check via cumulative AND over shifts.
    # Simple and clear (beginner-friendly) rather than maximally optimized.
    core_valid = np.ones((n_lat - 2 * half, n_lon - 2 * half), dtype=bool)
    for di in range(-half, half + 1):
        for dj in range(-half, half + 1):
            window = mask_vals[half + di: n_lat - half + di,
                                half + dj: n_lon - half + dj]
            core_valid &= window

    valid[half:n_lat - half, half:n_lon - half] = core_valid

    valid_da = xr.DataArray(valid, dims=ocean_mask.dims, coords=ocean_mask.coords,
                             name="valid_patch_center")
    n_valid = int(valid_da.sum().values)
    n_ocean = int(ocean_mask.sum().values)
    print(f"[patch_extraction] {n_valid} valid {patch_size}x{patch_size} patch "
          f"centers out of {n_ocean} ocean cells "
          f"({n_ocean - n_valid} ocean cells excluded because a "
          f"{patch_size}x{patch_size} neighborhood would touch land/edge).")
    return valid_da


def extract_patch(input_da: xr.DataArray, lat_idx: int, lon_idx: int,
                   patch_size: int = PATCH_SIZE) -> np.ndarray:
    """
    input_da: (time, lat, lon, variable) or (lat, lon, variable) DataArray.
    Extracts the patch_size x patch_size neighborhood centered on
    (lat_idx, lon_idx) as a numpy array. Caller should first check
    find_valid_patch_centers() to make sure this index is safe.
    """
    half = patch_size // 2
    return input_da.isel(
        lat=slice(lat_idx - half, lat_idx + half + 1),
        lon=slice(lon_idx - half, lon_idx + half + 1),
    ).values
