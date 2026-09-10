"""
argo_config.py
================
Stage: STAGE 4 — ARGO independent validation.

Deliberately a SEPARATE file from config.py. config.py drives the
training pipeline (Day 1-3); nothing in Stage 4 is allowed to feed
back into training, so its settings live here instead of being mixed
into the same file. Everything below is imported READ-ONLY by the
Stage 4 modules.

VERIFY REQUIRED markers mean the same thing they mean in config.py:
"I could not know this without your actual data/environment - please
confirm or edit."
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# 1. PATHS
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
ARGO_RAW_DIR = PROJECT_ROOT / "data" / "raw_argo"       # kept OUT of data/raw
                                                          # (Day 1's RAW_DIR) on
                                                          # purpose, see leakage note below.
ARGO_RESULTS_DIR = PROJECT_ROOT / "results" / "argo"

# VERIFY REQUIRED: point this at your real downloaded ARGO file(s) once you
# have them (see README's "Stage 4 - ARGO validation" section for exactly
# what to download and from where). Glob pattern - can match multiple files.
ARGO_REAL_FILES_GLOB = ARGO_RAW_DIR / "real" / "*.nc"

# Synthetic fixture used for the mandatory pre-real-data pipeline test
# (Step 9 of the Stage 4 brief). Not real observations - see
# make_synthetic_argo_fixture.py.
ARGO_SYNTHETIC_FIXTURE_PATH = ARGO_RAW_DIR / "synthetic" / "argo_synthetic_fixture.nc"

# ---------------------------------------------------------------------------
# 2. QUALITY CONTROL
# ---------------------------------------------------------------------------
# Argo QC flags (standard Argo convention, same for PRES_QC / TEMP_QC /
# POSITION_QC / JULD_QC):
#   1 = good, 2 = probably good, 3 = probably bad, 4 = bad, 5 = changed,
#   6-7 = not used, 8 = interpolated, 9 = missing.
# We accept ONLY 1 and 2 ("good" / "probably good") - the standard
# conservative choice for scientific validation use. Anything else is
# treated as missing (never silently used).
ARGO_ACCEPTED_QC_FLAGS = {1, 2}

# A profile-level position/time fix must also be good, or we cannot trust
# the match itself regardless of how good the temperature QC is.
ARGO_ACCEPTED_POSITION_QC_FLAGS = {1, 2}
ARGO_ACCEPTED_JULD_QC_FLAGS = {1, 2}

# ---------------------------------------------------------------------------
# 3. DEPTH INTERPOLATION
# ---------------------------------------------------------------------------
# Minimum number of QC-passed observations a profile must have before we
# attempt ANY interpolation onto the 15 target depths (a 1-point "profile"
# cannot be meaningfully interpolated).
ARGO_MIN_VALID_OBS_FOR_INTERP = 2

# We NEVER extrapolate: a target depth outside [min_observed, max_observed]
# pressure for that profile is left as NaN, not filled. This is enforced in
# src/argo_processing.py regardless of this config value - listed here only
# for documentation/visibility.
ARGO_ALLOW_EXTRAPOLATION = False

# Pressure (dbar) is used directly as depth (m) without a TEOS-10
# pressure-to-depth conversion (gsw.z_from_p). At these depths (<=1000 m)
# and this region's latitudes, the difference between pressure-in-dbar and
# true depth-in-meters is a few meters at most at 1000 dbar - small
# relative to our target-depth spacing, but it IS a known approximation.
# VERIFY REQUIRED / TODO if higher precision is needed: convert via gsw.
ARGO_DEPTH_FROM_PRESSURE_APPROX = True

# ---------------------------------------------------------------------------
# 4. SPATIAL + TEMPORAL MATCHING TOLERANCES (configurable, documented)
# ---------------------------------------------------------------------------
# TEMPORAL: the model consumes DAILY-resampled surface fields (see Day 1's
# TARGET_TEMPORAL_FREQ). An Argo profile's ascent takes several hours, so
# matching it to the SAME calendar day is the natural choice. We allow a
# small configurable slack (+/- N days) for cases where the nearest daily
# grid day has no valid patch (e.g. a satellite data gap that day), so a
# profile isn't discarded purely because of a same-day sensor gap. Matching
# always picks the CLOSEST available day within this window, not just any
# day within it.
ARGO_TEMPORAL_TOLERANCE_DAYS = 1

# SPATIAL: the grid resolution is 0.25 deg (~27.75 km at the equator, less
# at higher latitude as cos(lat)). We match each Argo profile to its
# NEAREST grid cell, then require the actual great-circle distance between
# the profile's true position and that grid cell's center to be within
# this tolerance - i.e. at most about half the grid diagonal, so a profile
# is never silently matched to a grid cell that is actually a full cell
# width away. Configurable because a coarser or finer product should use a
# different value.
ARGO_SPATIAL_TOLERANCE_KM = 20.0

# ---------------------------------------------------------------------------
# 5. REGIONAL CASE STUDY DEFINITIONS (Step 7)
# ---------------------------------------------------------------------------
# Explicit, documented, ADMITTEDLY simplified rectangular split. The real
# Arabian-Sea / Bay-of-Bengal boundary follows the Indian coastline and the
# Sri Lanka / Palk Strait area, not a straight meridian - but a straight,
# explicit, reproducible split is preferable here to an implicit or
# undocumented one. 77 deg E is the conventional approximate longitude of
# the southern Indian coastline/Kanyakumari.
ARABIAN_SEA_LON_RANGE = (45.0, 77.0)
BAY_OF_BENGAL_LON_RANGE = (77.0, 105.0)

# Do not force the regional comparison (Step 7) if a region has fewer
# matched profiles than this.
REGIONAL_MIN_MATCHED_PROFILES = 5

# ---------------------------------------------------------------------------
# 6. MODEL / CHECKPOINT (read-only references - Stage 4 never retrains)
# ---------------------------------------------------------------------------
CNN_CHECKPOINT_PATH = PROJECT_ROOT / "models" / "cnn_best.pt"
ML_READY_DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "ml_ready_dataset.nc"
NORMALIZATION_STATS_PATH = PROJECT_ROOT / "data" / "processed" / "normalization_stats.json"
METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "metadata.json"

RANDOM_SEED = 42
