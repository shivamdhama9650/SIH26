"""
region.py
=========
Stage: CROP TO PROJECT REGION

Crops any standardized dataset (must already have 'lat'/'lon' coords,
see coord_utils.standardize_coord_names) to the project bounding box
defined in config.REGION.
"""

import xarray as xr

from config import REGION


def crop_to_region(ds: xr.Dataset, name: str = "dataset") -> xr.Dataset:
    if "lat" not in ds.coords or "lon" not in ds.coords:
        raise ValueError(f"[region] '{name}' is missing standardized lat/lon "
                          f"coords - run coord_utils.standardize_coord_names first.")

    lat_ascending = bool(ds["lat"][0] < ds["lat"][-1])
    lat_slice = (slice(REGION["lat_min"], REGION["lat_max"]) if lat_ascending
                 else slice(REGION["lat_max"], REGION["lat_min"]))

    cropped = ds.sel(lat=lat_slice, lon=slice(REGION["lon_min"], REGION["lon_max"]))

    if cropped.sizes.get("lat", 0) == 0 or cropped.sizes.get("lon", 0) == 0:
        raise ValueError(
            f"[region] Cropping '{name}' to region {REGION} produced an EMPTY "
            f"array. Original lat range was "
            f"[{float(ds['lat'].min())}, {float(ds['lat'].max())}], lon range "
            f"[{float(ds['lon'].min())}, {float(ds['lon'].max())}]. "
            f"VERIFY REQUIRED: check longitude convention (0-360 vs -180-180) "
            f"and that this dataset actually covers the North Indian Ocean."
        )

    print(f"[region] '{name}' cropped to lat=[{float(cropped['lat'].min()):.2f},"
          f"{float(cropped['lat'].max()):.2f}] lon=[{float(cropped['lon'].min()):.2f},"
          f"{float(cropped['lon'].max()):.2f}]")
    return cropped


def crop_all(datasets: dict) -> dict:
    cropped = {}
    for name, ds in datasets.items():
        if ds is None:
            cropped[name] = None
            continue
        cropped[name] = crop_to_region(ds, name=name)
    return cropped
