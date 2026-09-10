"""
regrid.py
=========
Stage: PUT SURFACE VARIABLES / GLORYS ON COMMON 0.25 DEG GRID

Rule from the brief: regrid/coarsen properly, do NOT just subsample
points. xESMF (conservative or bilinear) is used if installed; otherwise
we fall back to an xarray-native block-mean coarsening (for grids finer
than the target) or linear interpolation (for grids coarser/comparable
to the target), which is the most scientifically defensible option
without extra dependencies. The fallback is clearly logged.
"""

import numpy as np
import xarray as xr

from config import REGION, GRID_RESOLUTION_DEG

try:
    import xesmf as xe
    XESMF_AVAILABLE = True
except ImportError:
    XESMF_AVAILABLE = False


def build_target_grid() -> xr.Dataset:
    """The common 0.25x0.25 grid covering the project region, cell-centered."""
    lats = np.arange(REGION["lat_min"], REGION["lat_max"] + 1e-9, GRID_RESOLUTION_DEG)
    lons = np.arange(REGION["lon_min"], REGION["lon_max"] + 1e-9, GRID_RESOLUTION_DEG)
    target = xr.Dataset({"lat": (["lat"], lats), "lon": (["lon"], lons)})
    print(f"[regrid] Target grid: {len(lats)} x {len(lons)} cells "
          f"({GRID_RESOLUTION_DEG} deg spacing), lat=[{lats[0]},{lats[-1]}], "
          f"lon=[{lons[0]},{lons[-1]}]")
    return target


def _native_resolution(ds: xr.Dataset) -> float:
    lat_res = float(np.median(np.diff(np.sort(ds["lat"].values))))
    lon_res = float(np.median(np.diff(np.sort(ds["lon"].values))))
    return (abs(lat_res) + abs(lon_res)) / 2.0


def regrid_dataset(ds: xr.Dataset, target_grid: xr.Dataset, name: str = "dataset",
                    method: str = "bilinear") -> xr.Dataset:
    """
    Regrids `ds` (must have standardized lat/lon coords) onto target_grid.
    Uses xESMF if available, otherwise a documented xarray fallback.
    """
    if ds is None:
        return None

    native_res = _native_resolution(ds)
    print(f"[regrid] '{name}' native resolution ~{native_res:.4f} deg -> "
          f"target {GRID_RESOLUTION_DEG} deg.")

    if XESMF_AVAILABLE:
        print(f"[regrid] '{name}': using xESMF ({method}).")
        regridder = xe.Regridder(ds, target_grid, method, periodic=False,
                                  ignore_degenerate=True)
        out = regridder(ds, keep_attrs=True)
        return out

    # --- Fallback: no xESMF ---
    print(f"[regrid] '{name}': xESMF NOT installed. Using xarray fallback. "
          f"LIMITATION: this is bilinear interpolation via xarray.interp "
          f"(source finer than target) or block-mean coarsening (source "
          f"finer, using coarsen()), NOT a true conservative-area regrid. "
          f"For final production numbers, install xESMF/ESMF for a proper "
          f"conservative regrid, especially near the coast where accuracy "
          f"matters most for land/sea masking.")

    if native_res < GRID_RESOLUTION_DEG * 0.7:
        # source is meaningfully finer -> coarsen (area-ish averaging)
        factor_lat = max(1, int(round(GRID_RESOLUTION_DEG / abs(
            np.median(np.diff(np.sort(ds["lat"].values)))))))
        factor_lon = max(1, int(round(GRID_RESOLUTION_DEG / abs(
            np.median(np.diff(np.sort(ds["lon"].values)))))))
        print(f"[regrid] '{name}': coarsening by factor lat={factor_lat}, "
              f"lon={factor_lon} then interpolating onto exact target grid.")
        coarsened = ds.coarsen(lat=factor_lat, lon=factor_lon, boundary="trim").mean()
        out = coarsened.interp(lat=target_grid["lat"], lon=target_grid["lon"],
                                method="linear")
    else:
        # source is comparable or coarser -> linear interpolation
        out = ds.interp(lat=target_grid["lat"], lon=target_grid["lon"], method="linear")

    return out


def regrid_all(datasets: dict, target_grid: xr.Dataset) -> dict:
    print("\n" + "=" * 70)
    print("HORIZONTAL REGRIDDING TO COMMON 0.25 DEG GRID")
    print(f"xESMF available: {XESMF_AVAILABLE}")
    print("=" * 70)
    out = {}
    for name, ds in datasets.items():
        out[name] = regrid_dataset(ds, target_grid, name=name)
    return out
