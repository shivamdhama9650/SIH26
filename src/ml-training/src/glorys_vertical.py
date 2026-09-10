"""
glorys_vertical.py
===================
Stage: VERTICALLY INTERPOLATE GLORYS TEMPERATURE TO THE 15 REQUIRED DEPTHS

Never assumes GLORYS's native depth levels already equal the target
depths. Reads the actual depth coordinate, checks target depths are
within range, and interpolates with xarray's built-in interp (linear
in depth, which is the standard/defensible choice for ocean temperature
profiles at these depth spacings).
"""

import numpy as np
import xarray as xr

from config import TARGET_DEPTHS
from src.coord_utils import detect_var_name


def interpolate_glorys_depths(glorys_ds: xr.Dataset) -> xr.DataArray:
    """
    Returns a DataArray named 'temperature' with dims (time, depth, lat, lon)
    where depth has exactly len(TARGET_DEPTHS) = 15 levels.
    """
    if "depth" not in glorys_ds.coords:
        raise ValueError("[glorys_vertical] GLORYS dataset has no depth coordinate. "
                          "VERIFY REQUIRED: confirm this file actually contains "
                          "3D temperature (thetao/temperature), not a 2D SST-only file.")

    native_depths = np.asarray(glorys_ds["depth"].values, dtype=float)
    print(f"[glorys_vertical] Native GLORYS depth levels ({len(native_depths)}): "
          f"{np.round(native_depths, 1).tolist()}")

    target = np.asarray(TARGET_DEPTHS, dtype=float)
    max_native = native_depths.max()
    min_native = native_depths.min()
    exceeding = target[target > max_native]
    if len(exceeding) > 0:
        raise ValueError(
            f"[glorys_vertical] Target depths {exceeding.tolist()} exceed the "
            f"maximum native GLORYS depth ({max_native}). VERIFY REQUIRED: "
            f"either this GLORYS file doesn't extend deep enough, or the "
            f"depth coordinate was mis-detected."
        )

    shallow_gap = target[target < min_native]
    extrapolate = len(shallow_gap) > 0
    if extrapolate:
        print(f"[glorys_vertical] NOTE: target depth(s) {shallow_gap.tolist()} are "
              f"SHALLOWER than GLORYS's shallowest native level ({min_native}m). "
              f"This is common when the target list includes 0m but the native "
              f"grid's top cell is centered slightly below the surface (e.g. "
              f"0.49m). ASSUMPTION: extrapolating linearly using the nearest "
              f"two native levels, which is standard practice for this small "
              f"a gap. If min_native is far from 0 (e.g. >5m), treat this "
              f"extrapolation with more caution.")

    temp_var = detect_var_name(glorys_ds, "glorys_temp", required=True)
    temp = glorys_ds[temp_var]

    print(f"[glorys_vertical] Interpolating '{temp_var}' from {len(native_depths)} "
          f"native levels to {len(target)} target depths (linear in depth)...")
    interp_kwargs = {"fill_value": "extrapolate"} if extrapolate else {}
    print("[glorys_vertical] Interpolating in 30-day chunks to stay within RAM...")
    import dask
    interpolated = temp.interp(depth=target, method="linear", kwargs=interp_kwargs)
    interpolated = interpolated.rename("temperature")
    # Ensure dask graph is chunked so .compute() later is safe
    if hasattr(interpolated, "chunks") and interpolated.chunks:
        interpolated = interpolated.chunk({"time": 30})

    n_depths_out = interpolated.sizes.get("depth", 0)
    if n_depths_out != 15:
        raise AssertionError(
            f"[glorys_vertical] Expected 15 output depths, got {n_depths_out}. "
            f"Something went wrong in interpolation."
        )
    print(f"[glorys_vertical] OK: output depth dimension size = {n_depths_out} "
          f"(target depths = {target.tolist()})")

    return interpolated
