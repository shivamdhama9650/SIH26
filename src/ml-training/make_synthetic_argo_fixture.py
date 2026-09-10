"""
make_synthetic_argo_fixture.py
=================================
Stage: STAGE 4, Step 9 — build a small synthetic ARGO fixture to test
the complete ARGO pipeline BEFORE any real ARGO data is used.

NOT real observations. Schema-matches what src/argo_loader.py expects
from a real Argo profile product (LATITUDE, LONGITUDE, TIME, PRES,
TEMP, PRES_QC, TEMP_QC, POSITION_QC, JULD_QC on an N_PROF x N_LEVELS
grid).

Reads the ACTUAL Day 1 output (data/processed/ml_ready_dataset.nc) to
find real grid/time coverage, so the fixture's "should match" profiles
are guaranteed to land on real ocean grid cells within the real time
range, and its "should NOT match" profiles are guaranteed to fall
outside tolerance - deliberately, not by accident.

Run AFTER run_day1/2/3 have produced their outputs:
    python make_synthetic_argo_fixture.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import config
import argo_config

RNG = np.random.default_rng(argo_config.RANDOM_SEED)


def _build_profile(lat, lon, time, depths, temps, pres_qc=None, temp_qc=None):
    n_levels = len(depths)
    pres_qc = pres_qc if pres_qc is not None else np.ones(n_levels, dtype=int)
    temp_qc = temp_qc if temp_qc is not None else np.ones(n_levels, dtype=int)
    return {
        "lat": lat, "lon": lon, "time": time,
        "pres": np.asarray(depths, dtype="float64"),
        "temp": np.asarray(temps, dtype="float64"),
        "pres_qc": np.asarray(pres_qc, dtype=int),
        "temp_qc": np.asarray(temp_qc, dtype=int),
        "position_qc": 1, "juld_qc": 1,
    }


def _synthetic_temp_profile(depths):
    """Same decay shape as make_synthetic_data*.py's fake GLORYS field,
    so the CNN's predictions and the fixture's 'truth' are in a
    comparable physical range (not required to match exactly - Argo IS
    an independent source - but keeping them in a realistic range makes
    the smoke-test metrics interpretable rather than nonsensical)."""
    surf = 28.0 + RNG.normal(0, 0.5)
    return 4 + (surf - 4) * np.exp(-np.asarray(depths) / 300.0)


def build_fixture(out_path: Path):
    ml_ready_path = config.PROCESSED_DIR / "ml_ready_dataset.nc"
    if not ml_ready_path.exists():
        raise FileNotFoundError(
            f"[make_synthetic_argo_fixture] {ml_ready_path} not found - "
            f"run run_pipeline.py (Day 1) first so this script can anchor "
            f"the fixture to real grid/time coverage."
        )
    ds = xr.open_dataset(ml_ready_path)
    lat_grid = ds["lat"].values
    lon_grid = ds["lon"].values
    time_grid = pd.DatetimeIndex(ds["time"].values)
    ocean_mask = ds["ocean_mask"].values

    # find a handful of interior (non-edge) ocean cells to anchor "should
    # match" profiles to
    n_lat, n_lon = ocean_mask.shape
    interior_ocean = np.zeros_like(ocean_mask)
    interior_ocean[1:-1, 1:-1] = ocean_mask[1:-1, 1:-1]
    # require the FULL 3x3 neighborhood to be ocean too (same rule as training)
    full_valid = np.ones((n_lat - 2, n_lon - 2), dtype=bool)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            full_valid &= ocean_mask[1 + di:n_lat - 1 + di, 1 + dj:n_lon - 1 + dj]
    candidate_lat_idx, candidate_lon_idx = np.where(full_valid)
    candidate_lat_idx += 1
    candidate_lon_idx += 1
    if len(candidate_lat_idx) < 4:
        raise RuntimeError("[make_synthetic_argo_fixture] Not enough valid "
                            "interior ocean cells in this dataset to build "
                            "a meaningful fixture.")
    pick = RNG.choice(len(candidate_lat_idx), size=4, replace=False)

    mid_time = time_grid[len(time_grid) // 2]

    profiles = []

    # ---- 1. "SHOULD MATCH" profiles: near an interior ocean grid cell,
    #         well within the time range, full depth coverage, all-good QC.
    for k in pick[:2]:
        lat_idx, lon_idx = candidate_lat_idx[k], candidate_lon_idx[k]
        lat0, lon0 = lat_grid[lat_idx], lon_grid[lon_idx]
        # small jitter well within spatial tolerance (a few km)
        lat = lat0 + RNG.uniform(-0.03, 0.03)
        lon = lon0 + RNG.uniform(-0.03, 0.03)
        t = mid_time + pd.Timedelta(hours=int(RNG.uniform(-6, 6)))
        depths = [2, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500, 700, 1000]
        temps = _synthetic_temp_profile(depths)
        profiles.append(_build_profile(lat, lon, t, depths, temps))

    # ---- 2. "SHOULD MATCH BUT WITH MISSING DEEP MEASUREMENTS": a
    #         shallow-only float, only reaches 150 m.
    lat_idx, lon_idx = candidate_lat_idx[pick[2]], candidate_lon_idx[pick[2]]
    lat0, lon0 = lat_grid[lat_idx], lon_grid[lon_idx]
    depths = [2, 10, 25, 50, 75, 100, 150]
    temps = _synthetic_temp_profile(depths)
    profiles.append(_build_profile(lat0 + 0.01, lon0 - 0.02, mid_time, depths, temps))

    # ---- 3. "SHOULD MATCH BUT SOME LEVELS FAIL QC": mix of good/bad flags.
    lat_idx, lon_idx = candidate_lat_idx[pick[3]], candidate_lon_idx[pick[3]]
    lat0, lon0 = lat_grid[lat_idx], lon_grid[lon_idx]
    depths = [2, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500, 700, 1000]
    temps = _synthetic_temp_profile(depths)
    temp_qc = np.array([1, 1, 4, 1, 1, 3, 1, 1, 1, 4, 1, 1, 1])  # a few bad levels
    profiles.append(_build_profile(lat0, lon0, mid_time, depths, temps, temp_qc=temp_qc))

    # ---- 4. "SHOULD NOT MATCH — spatially far". NOTE: on a regular
    #         0.25deg grid, the farthest any point INSIDE the domain can
    #         be from its nearest grid cell is ~half the grid diagonal
    #         (~19.6 km) - already under most reasonable spatial
    #         tolerances, so a true "too far" case has to sit OUTSIDE the
    #         grid's actual coverage (a real, realistic scenario: a float
    #         just outside the study region). Placed just past lat_max.
    far_lat = float(lat_grid.max() + 2.0)
    far_lon = float(lon_grid.mean())
    depths = [2, 10, 20, 50, 100]
    temps = _synthetic_temp_profile(depths)
    profiles.append(_build_profile(far_lat, far_lon, mid_time, depths, temps))

    # ---- 5. "SHOULD NOT MATCH — temporally far" (60 days outside the
    #         whole dataset's time range).
    lat_idx, lon_idx = candidate_lat_idx[pick[0]], candidate_lon_idx[pick[0]]
    lat0, lon0 = lat_grid[lat_idx], lon_grid[lon_idx]
    far_time = time_grid.max() + pd.Timedelta(days=60)
    depths = [2, 10, 20, 50, 100]
    temps = _synthetic_temp_profile(depths)
    profiles.append(_build_profile(lat0, lon0, far_time, depths, temps))

    # ---- 6. "SHOULD NOT MATCH — lands on/near the synthetic land mass"
    #         (only meaningful if a land cell actually exists in this grid;
    #         skipped otherwise rather than fabricating a fake one).
    land_idx = np.where(~ocean_mask)
    if len(land_idx[0]) > 0:
        li = land_idx[0][0], land_idx[1][0]
        lat0, lon0 = lat_grid[li[0]], lon_grid[li[1]]
        depths = [2, 10, 20, 50, 100]
        temps = _synthetic_temp_profile(depths)
        profiles.append(_build_profile(lat0, lon0, mid_time, depths, temps))

    # ---- 7. "SHOULD NOT MATCH — position QC fails" (fix not trustworthy,
    #         even though it looks close in space/time).
    lat_idx, lon_idx = candidate_lat_idx[pick[1]], candidate_lon_idx[pick[1]]
    lat0, lon0 = lat_grid[lat_idx], lon_grid[lon_idx]
    depths = [2, 10, 20, 50, 100]
    temps = _synthetic_temp_profile(depths)
    prof = _build_profile(lat0, lon0, mid_time, depths, temps)
    prof["position_qc"] = 4  # bad fix
    profiles.append(prof)

    # ---- pack into the N_PROF x N_LEVELS Argo schema (NaN-pad ragged depths) ----
    n_prof = len(profiles)
    n_levels = max(len(p["pres"]) for p in profiles)

    lat_arr = np.array([p["lat"] for p in profiles], dtype="float64")
    lon_arr = np.array([p["lon"] for p in profiles], dtype="float64")
    time_arr = np.array([p["time"] for p in profiles], dtype="datetime64[ns]")
    pres_arr = np.full((n_prof, n_levels), np.nan, dtype="float64")
    temp_arr = np.full((n_prof, n_levels), np.nan, dtype="float64")
    pres_qc_arr = np.full((n_prof, n_levels), 9, dtype="int32")
    temp_qc_arr = np.full((n_prof, n_levels), 9, dtype="int32")
    position_qc_arr = np.array([p["position_qc"] for p in profiles], dtype="int32")
    juld_qc_arr = np.array([p["juld_qc"] for p in profiles], dtype="int32")

    for i, p in enumerate(profiles):
        n = len(p["pres"])
        pres_arr[i, :n] = p["pres"]
        temp_arr[i, :n] = p["temp"]
        pres_qc_arr[i, :n] = p["pres_qc"]
        temp_qc_arr[i, :n] = p["temp_qc"]

    juld = (time_arr - np.datetime64("1950-01-01")) / np.timedelta64(1, "D")

    ds_out = xr.Dataset(
        {
            "LATITUDE": ("N_PROF", lat_arr),
            "LONGITUDE": ("N_PROF", lon_arr),
            "TIME": ("N_PROF", time_arr),
            "JULD": ("N_PROF", juld.astype("float64")),
            "PRES": (("N_PROF", "N_LEVELS"), pres_arr),
            "TEMP": (("N_PROF", "N_LEVELS"), temp_arr),
            "PRES_QC": (("N_PROF", "N_LEVELS"), pres_qc_arr),
            "TEMP_QC": (("N_PROF", "N_LEVELS"), temp_qc_arr),
            "POSITION_QC": ("N_PROF", position_qc_arr),
            "JULD_QC": ("N_PROF", juld_qc_arr),
        },
        attrs={"title": "SYNTHETIC ARGO FIXTURE - NOT REAL OBSERVATIONS",
               "purpose": "Stage 4 pipeline smoke test only"},
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds_out.to_netcdf(out_path)
    print(f"[make_synthetic_argo_fixture] Wrote {n_prof} synthetic profiles "
          f"(up to {n_levels} levels each) -> {out_path}")
    print("  Expected outcomes when matched:")
    print("    profiles 0-1: MATCH (full depth, all-good QC)")
    print("    profile  2:   MATCH but missing-deep (only to 150m)")
    print("    profile  3:   MATCH but some levels QC-rejected")
    print("    profile  4:   NO MATCH (too far spatially)")
    print("    profile  5:   NO MATCH (too far temporally)")
    if len(land_idx[0]) > 0:
        print("    profile  6:   NO MATCH (lands on/near masked land)")
        print("    profile  7:   NO MATCH (bad POSITION_QC)")
    else:
        print("    profile  6:   NO MATCH (bad POSITION_QC)")
        print("    (no land cell existed in this grid to build a land-reject case)")


if __name__ == "__main__":
    build_fixture(argo_config.ARGO_SYNTHETIC_FIXTURE_PATH)
