"""
config.py
=========
Single place to change paths / variable names / region / depths.
Nothing in src/*.py should hard-code any of these values - they all
read from this file so you can adapt to whatever files you actually
downloaded without touching the processing logic.

VERIFY REQUIRED markers below mean: "I could not know this without
looking at your actual files - please fill this in / confirm it."
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# 1. PATHS
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = Path('/content/processed')
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# VERIFY REQUIRED: point these at your actual downloaded files.
# Glob patterns are fine (e.g. "sst_*.nc") if a variable is split across
# multiple daily/monthly files - loaders.py opens them with
# xr.open_mfdataset when a pattern matches more than one file.
RAW_FILES = {
    "sst":   RAW_DIR / "sst" / "*.nc",
    "sss":   RAW_DIR / "sss" / "*.nc",       # optional - checked at runtime
    "ssh":   RAW_DIR / "ssh" / "*.nc",
    "currents": RAW_DIR / "currents" / "*.nc",  # product not assumed - see below
    "winds": RAW_DIR / "winds" / "*.nc",
    "glorys": RAW_DIR / "glorys" / "*.nc",
}

# ---------------------------------------------------------------------------
# 2. VARIABLE NAME CANDIDATES
# ---------------------------------------------------------------------------
# Different products name the same physical quantity differently.
# coord_utils.detect_var() searches a dataset's data_vars for the first
# candidate (case-insensitive, substring match as a last resort) so we
# never assume a single fixed name.
VARIABLE_CANDIDATES = {
    "sst": ["sst", "analysed_sst", "sea_surface_temperature", "thetao", "temperature"],
    "sss": ["sss", "sea_surface_salinity", "so", "salinity"],
    "ssh": ["ssh", "sla", "adt", "sea_surface_height", "zos"],
    "u_current": ["ugos", "uo", "u_current", "eastward_sea_water_velocity", "u"],
    "v_current": ["vgos", "vo", "v_current", "northward_sea_water_velocity", "v"],
    "u_wind": ["u10", "uwnd", "eastward_wind", "u_wind"],
    "v_wind": ["v10", "vwnd", "northward_wind", "v_wind"],
    "glorys_temp": ["thetao", "temperature", "water_temp", "temp"],
    "land_sea_mask": ["mask", "land_sea_mask", "lsm", "sea_mask"],
}

# ---------------------------------------------------------------------------
# 3. COORDINATE NAME CANDIDATES
# ---------------------------------------------------------------------------
COORD_CANDIDATES = {
    "lat": ["lat", "latitude", "y", "nav_lat"],
    "lon": ["lon", "longitude", "long", "x", "nav_lon"],
    "time": ["time", "date", "valid_time", "forecast_reference_time"],
    "depth": ["depth", "lev", "z", "level", "deptht"],
}

# ---------------------------------------------------------------------------
# 4. REGION AND GRID
# ---------------------------------------------------------------------------
REGION = {
    "lat_min": 5.0,
    "lat_max": 30.0,
    "lon_min": 45.0,
    "lon_max": 105.0,
}
GRID_RESOLUTION_DEG = 0.25

# ---------------------------------------------------------------------------
# 5. TARGET DEPTHS (meters) - GLORYS is interpolated onto exactly these
# ---------------------------------------------------------------------------
TARGET_DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]

# ---------------------------------------------------------------------------
# 6. TEMPORAL ALIGNMENT
# ---------------------------------------------------------------------------
TARGET_TEMPORAL_FREQ = "1D"  # daily - all inputs are resampled/aligned to this

# ---------------------------------------------------------------------------
# 7. TRAIN / VALIDATION / TEST SPLIT
# ---------------------------------------------------------------------------
# VERIFY REQUIRED: once dataset_inspection.py tells you how many distinct
# calendar years are available, confirm/edit this split.
SPLIT_CONFIG = {
    "mode": "auto",          # "auto" | "whole_year" | "seasonal_block"
    # used only if mode == "whole_year": explicit year lists
    "train_years": None,     # e.g. [2015, 2016, 2017, 2018]
    "val_years": None,       # e.g. [2019]
    "test_years": None,      # e.g. [2020]
    # used only if mode == "seasonal_block" (single-year pilot fallback):
    # month numbers (1-12) assigned to each split, contiguous blocks.
    "seasonal_block": {
        "train_months": [1, 2, 3, 4, 5, 6, 7, 8],
        "val_months": [9, 10],
        "test_months": [11, 12],
    },
}

# ---------------------------------------------------------------------------
# 8. MASKING
# ---------------------------------------------------------------------------
# If no explicit land/sea mask variable is found in any dataset, we fall
# back to: a grid cell is "ocean" if SST (or GLORYS surface temperature)
# is finite (not NaN/fill) for >= this fraction of training-period days.
MASK_FALLBACK_MIN_VALID_FRACTION = 0.9

# ---------------------------------------------------------------------------
# 9. PATCH EXTRACTION
# ---------------------------------------------------------------------------
PATCH_SIZE = 3  # 3x3, locked architecture

# ---------------------------------------------------------------------------
# 10. MISC
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
NAN_FRACTION_WARN_THRESHOLD = 0.5  # warn loudly if a variable is >50% NaN

# Products that are surface-only and should NOT trigger a depth-coord warning.
SURFACE_ONLY_PRODUCTS = {"ssh", "sla", "adt", "sss", "winds", "u10", "v10"}
