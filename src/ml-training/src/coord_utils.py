"""
coord_utils.py
==============
Helpers to detect coordinate and variable names in an arbitrary xarray
Dataset without assuming fixed names like 'lat'/'lon'/'time'/'depth'.

Every other module calls detect_coord_name() / detect_var_name() instead
of touching ds['lat'] etc. directly.
"""

import sys
import xarray as xr

from config import COORD_CANDIDATES, VARIABLE_CANDIDATES


class CoordinateNotFoundError(Exception):
    pass


class VariableNotFoundError(Exception):
    pass


def _search(names_available, candidates):
    """Case-insensitive exact match first, then substring match."""
    lower_map = {n.lower(): n for n in names_available}

    # exact (case-insensitive) match
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]

    # substring fallback (e.g. 'longitude_bounds' contains 'longitude')
    for cand in candidates:
        for n in names_available:
            if cand.lower() in n.lower():
                return n
    return None


def detect_coord_name(ds: xr.Dataset, coord_key: str, required: bool = True):
    """
    coord_key: one of 'lat', 'lon', 'time', 'depth' (see config.COORD_CANDIDATES)
    Returns the actual coordinate/dimension name used in `ds`, or None.
    """
    if coord_key not in COORD_CANDIDATES:
        raise ValueError(f"Unknown coord_key '{coord_key}'. "
                          f"Known keys: {list(COORD_CANDIDATES.keys())}")

    candidates = COORD_CANDIDATES[coord_key]
    all_names = list(ds.coords) + list(ds.dims)
    found = _search(all_names, candidates)

    if found is None:
        msg = (f"VERIFY REQUIRED: could not detect a '{coord_key}' coordinate "
               f"in dataset. Looked for any of {candidates} among "
               f"{all_names}. Please add the real name to "
               f"config.COORD_CANDIDATES['{coord_key}'].")
        if required:
            raise CoordinateNotFoundError(msg)
        else:
            print(msg, file=sys.stderr)
            return None
    return found


def detect_var_name(ds: xr.Dataset, var_key: str, required: bool = True):
    """
    var_key: one of the keys in config.VARIABLE_CANDIDATES (e.g. 'sst')
    Returns the actual data_var name used in `ds`, or None.
    """
    if var_key not in VARIABLE_CANDIDATES:
        raise ValueError(f"Unknown var_key '{var_key}'. "
                          f"Known keys: {list(VARIABLE_CANDIDATES.keys())}")

    candidates = VARIABLE_CANDIDATES[var_key]
    all_names = list(ds.data_vars)
    found = _search(all_names, candidates)

    if found is None:
        msg = (f"VERIFY REQUIRED: could not detect variable '{var_key}' "
               f"in dataset. Looked for any of {candidates} among "
               f"{all_names}. Please add the real name to "
               f"config.VARIABLE_CANDIDATES['{var_key}'], or this "
               f"variable is genuinely unavailable.")
        if required:
            raise VariableNotFoundError(msg)
        else:
            print(msg, file=sys.stderr)
            return None
    return found


def standardize_coord_names(ds: xr.Dataset) -> xr.Dataset:
    """
    Renames whatever lat/lon/time/depth-like coords exist to the
    canonical names 'lat', 'lon', 'time', 'depth' so downstream code
    can be written once and work for every dataset. Depth is optional
    (only GLORYS has it).
    """
    rename_map = {}
    for key in ["lat", "lon", "time", "depth"]:
        required = key != "depth"
        name = detect_coord_name(ds, key, required=required)
        if name is not None and name != key:
            rename_map[name] = key
    if rename_map:
        ds = ds.rename(rename_map)
    return ds


def normalize_longitude_convention(ds: xr.Dataset, target_min=45.0, target_max=105.0) -> xr.Dataset:
    """
    Some products use 0-360 longitude, others use -180-180.
    Our target region (45E-105E) is unambiguous in either convention,
    but we standardize to -180..180 (common convention) before cropping,
    UNLESS that would place the region outside the data's actual range,
    in which case we standardize to 0..360 instead.
    """
    if "lon" not in ds.coords:
        return ds

    lon_vals = ds["lon"].values
    lon_min, lon_max = float(lon_vals.min()), float(lon_vals.max())

    if lon_min >= 0 and lon_max > 180:
        # dataset is in 0-360 convention
        if target_min >= 0 and target_max <= 360:
            return ds  # target region already fits, no conversion needed
    elif lon_max <= 180:
        # dataset already in -180..180
        return ds

    # default: convert dataset to -180..180
    new_lon = ((ds["lon"].values + 180) % 360) - 180
    ds = ds.assign_coords(lon=new_lon)
    ds = ds.sortby("lon")
    return ds
