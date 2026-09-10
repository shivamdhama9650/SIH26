"""
argo_loader.py
================
Stage: STAGE 4 — ARGO independent validation.

Loads Argo profile data into one COMMON representation, regardless of
whether it came from a real GDAC/argopy-style file or the synthetic
test fixture, so every downstream module (argo_processing,
argo_matching, argo_validation) only ever has to know one shape.

REAL ARGO DATA FORMAT (what this loader expects)
--------------------------------------------------
Argo "core" profile files (single-cycle *_prof.nc from the GDAC, or a
multi-profile aggregate as produced by `argopy` / `argo_gdac`-style
tools) are xarray-openable NetCDF datasets shaped like:

    Dimensions: (N_PROF, N_LEVELS)
    Coordinates / variables actually used here:
        LATITUDE   (N_PROF)             float64   deg N
        LONGITUDE  (N_PROF)             float64   deg E (0-360 or -180..180
                                                    depending on source -
                                                    normalized on load)
        TIME       (N_PROF)             datetime64  (argopy already decodes
                                                    JULD -> TIME; if loading a
                                                    raw GDAC file directly, JULD
                                                    is days since
                                                    1950-01-01 and must be
                                                    decoded first - see
                                                    _decode_juld_if_needed())
        PRES       (N_PROF, N_LEVELS)   float64   dbar (~ depth in meters,
                                                    see argo_config.
                                                    ARGO_DEPTH_FROM_PRESSURE_APPROX)
        TEMP       (N_PROF, N_LEVELS)   float64   degC (in-situ temperature)
        PRES_QC    (N_PROF, N_LEVELS)   |S1 or int  Argo QC flag per level
        TEMP_QC    (N_PROF, N_LEVELS)   |S1 or int  Argo QC flag per level
        POSITION_QC(N_PROF)             |S1 or int  QC flag for the fix (optional)
        JULD_QC    (N_PROF)             |S1 or int  QC flag for the time (optional)
        PLATFORM_NUMBER (N_PROF)        optional, kept through if present

QC flags are Argo-standard single characters/ints: 1=good, 2=probably
good, 3=probably bad, 4=bad, 5=changed, 8=interpolated, 9=missing (see
argo_config.ARGO_ACCEPTED_QC_FLAGS for exactly which we accept).

WHERE TO GET REAL DATA (documented for the next step, not fetched here
- this sandbox has no network access to Argo data centers)
--------------------------------------------------------------------
- Recommended: the `argopy` Python package
  (https://argopy.readthedocs.io), e.g.:
      from argopy import DataFetcher
      ds = DataFetcher().region(
          [45, 105, 5, 30, 0, 1000, '2018-01-01', '2018-06-01']
      ).to_xarray()
  This already returns (close to) the shape documented above.
- Direct GDAC mirrors (ifremer / usgodae) also work, but need the
  per-float index file first (e.g. `ar_index_global_prof.txt`) to know
  which profile files fall in the region/time window before
  downloading - do NOT bulk-download the global archive.
- Restrict to the project region (5N-30N, 45E-105E) and the ml_ready
  dataset's actual time coverage (see metadata.json) - there is no
  reason to fetch more than that.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import argo_config


class ArgoLoadError(Exception):
    pass


REQUIRED_VARS = ["LATITUDE", "LONGITUDE", "PRES", "TEMP"]


def _decode_juld_if_needed(ds: xr.Dataset) -> xr.Dataset:
    """
    argopy output already has a decoded TIME variable. A raw GDAC profile
    file instead has JULD (days since 1950-01-01, Argo's reference epoch).
    If TIME is missing but JULD is present, decode it here rather than
    silently leaving profiles without a usable timestamp.
    """
    if "TIME" in ds.variables:
        return ds
    if "JULD" not in ds.variables:
        raise ArgoLoadError(
            "[argo_loader] Neither 'TIME' nor 'JULD' found in the Argo "
            "file - cannot determine profile timestamps. This file does "
            "not look like a standard Argo profile product."
        )
    epoch = pd.Timestamp("1950-01-01")
    juld_days = ds["JULD"].values.astype("float64")
    time_vals = pd.to_datetime(juld_days, unit="D", origin=epoch)
    ds = ds.assign(TIME=("N_PROF", time_vals))
    print("[argo_loader] Decoded JULD (days since 1950-01-01) -> TIME.")
    return ds


def _normalize_longitude(lon: np.ndarray) -> np.ndarray:
    """Standardize Argo longitudes to -180..180, matching Day 1's
    normalize_longitude_convention() default target so lat/lon comparisons
    against the model grid are apples-to-apples. Our region (45E-105E) is
    positive/unambiguous in either convention, so this only matters if a
    source ever hands back e.g. -300 style wraparounds."""
    return ((lon + 180) % 360) - 180


def _extract_qc_array(ds: xr.Dataset, name: str, shape) -> np.ndarray:
    """
    Argo QC flags are sometimes stored as byte strings (b'1'), sometimes
    as raw integers, sometimes absent entirely. Returns an int array of
    the given shape; if the variable is absent, returns all-9 ("missing")
    rather than assuming everything passed QC - never optimistic on
    missing QC.
    """
    if name not in ds.variables:
        print(f"[argo_loader] WARNING: '{name}' not present in file - "
              f"treating all flags as 9 (missing/unknown), so nothing "
              f"passes QC on this field.")
        return np.full(shape, 9, dtype=int)

    raw = ds[name].values
    out = np.empty(raw.shape, dtype=int)
    flat_raw = raw.reshape(-1)
    flat_out = out.reshape(-1)
    for i, v in enumerate(flat_raw):
        if isinstance(v, (bytes, np.bytes_)):
            v = v.decode("utf-8", errors="ignore").strip()
        if v in (None, "", " "):
            flat_out[i] = 9
            continue
        try:
            flat_out[i] = int(v)
        except (ValueError, TypeError):
            flat_out[i] = 9
    return out


def load_argo_dataset(path_or_glob) -> dict:
    """
    Loads one or more Argo profile NetCDF file(s) (real or synthetic
    fixture - same code path, since the fixture is built to the same
    schema) into a common in-memory representation:

    {
        "n_profiles": int,
        "n_levels": int,
        "lat": (n_profiles,) float64,
        "lon": (n_profiles,) float64,           # standardized -180..180
        "time": (n_profiles,) datetime64[ns],
        "pres": (n_profiles, n_levels) float64, # dbar, NaN-padded
        "temp": (n_profiles, n_levels) float64, # degC, NaN-padded
        "pres_qc": (n_profiles, n_levels) int,
        "temp_qc": (n_profiles, n_levels) int,
        "position_qc": (n_profiles,) int,
        "juld_qc": (n_profiles,) int,
        "platform_number": (n_profiles,) object or None,
        "source_files": [str, ...],
    }

    Raises ArgoLoadError with a clear message (naming the missing
    variable) rather than guessing, if the file doesn't look like a
    standard Argo profile product.
    """
    path_or_glob = Path(path_or_glob)
    if path_or_glob.is_dir():
        files = sorted(path_or_glob.glob("*.nc"))
    else:
        files = sorted(path_or_glob.parent.glob(path_or_glob.name))

    if not files:
        raise ArgoLoadError(
            f"[argo_loader] No Argo files found matching {path_or_glob}. "
            f"See the docstring at the top of this module (or the "
            f"README's Stage 4 section) for where/how to obtain real "
            f"Argo data for this region."
        )

    print(f"[argo_loader] Found {len(files)} Argo file(s): "
          f"{[f.name for f in files]}")

    datasets = []
    for f in files:
        ds = xr.open_dataset(f)
        missing = [v for v in REQUIRED_VARS if v not in ds.variables]
        if missing:
            raise ArgoLoadError(
                f"[argo_loader] '{f}' is missing required variable(s) "
                f"{missing}. This does not look like a standard Argo "
                f"profile file (expects LATITUDE, LONGITUDE, PRES, TEMP "
                f"on an N_PROF x N_LEVELS grid - see this module's "
                f"docstring)."
            )
        ds = _decode_juld_if_needed(ds)
        datasets.append(ds)

    # Concatenate along N_PROF if multiple files (pad N_LEVELS to the max
    # seen, matching Argo's own convention of ragged profile lengths).
    if len(datasets) == 1:
        combined = datasets[0]
    else:
        combined = xr.concat(datasets, dim="N_PROF", join="outer")

    n_profiles = combined.sizes["N_PROF"]
    n_levels = combined.sizes["N_LEVELS"]

    lat = combined["LATITUDE"].values.astype("float64")
    lon = _normalize_longitude(combined["LONGITUDE"].values.astype("float64"))
    time = pd.to_datetime(combined["TIME"].values)
    pres = combined["PRES"].values.astype("float64")
    temp = combined["TEMP"].values.astype("float64")

    pres_qc = _extract_qc_array(combined, "PRES_QC", pres.shape)
    temp_qc = _extract_qc_array(combined, "TEMP_QC", temp.shape)
    position_qc = _extract_qc_array(combined, "POSITION_QC", (n_profiles,))
    juld_qc = _extract_qc_array(combined, "JULD_QC", (n_profiles,))

    platform_number = None
    if "PLATFORM_NUMBER" in combined.variables:
        platform_number = np.asarray(combined["PLATFORM_NUMBER"].values)

    print(f"[argo_loader] Loaded {n_profiles} profiles, up to {n_levels} "
          f"levels each, from {len(files)} file(s).")

    return {
        "n_profiles": n_profiles,
        "n_levels": n_levels,
        "lat": lat,
        "lon": lon,
        "time": time,
        "pres": pres,
        "temp": temp,
        "pres_qc": pres_qc,
        "temp_qc": temp_qc,
        "position_qc": position_qc,
        "juld_qc": juld_qc,
        "platform_number": platform_number,
        "source_files": [str(f) for f in files],
    }


def load_real_argo(glob_path=None) -> dict:
    """Real-data entry point. Fails loudly (ArgoLoadError) with a helpful
    message if no files are present yet - never fabricates data."""
    glob_path = glob_path or argo_config.ARGO_REAL_FILES_GLOB
    return load_argo_dataset(glob_path)


def load_synthetic_argo(path=None) -> dict:
    """Synthetic-fixture entry point (Step 9 of the Stage 4 brief).
    Same schema, clearly NOT real observations - callers/reports must
    label results from this source as synthetic-fixture-only."""
    path = path or argo_config.ARGO_SYNTHETIC_FIXTURE_PATH
    return load_argo_dataset(path)
