"""
argo_processing.py
====================
Stage: STAGE 4 — ARGO independent validation.

Two jobs:
  1. Apply quality control (per-level PRES_QC/TEMP_QC + profile-level
     POSITION_QC/JULD_QC) - bad measurements become NaN, never silently
     kept.
  2. Interpolate each QC-passed profile onto the SAME 15 target depths
     the CNN was trained to predict (config.TARGET_DEPTHS), WITHOUT
     extrapolating beyond what that profile actually observed.

Depth vs. pressure: PRES (dbar) is used directly as depth (m) - see
argo_config.ARGO_DEPTH_FROM_PRESSURE_APPROX for the documented caveat.
"""

import numpy as np

import argo_config


def apply_qc(argo: dict) -> dict:
    """
    Returns a COPY of `argo` (as loaded by argo_loader) with:
      - temp/pres values failing per-level QC replaced by NaN
      - profiles failing profile-level POSITION_QC or JULD_QC entirely
        excluded (position/time can't be trusted -> can't be matched or
        reported at all)

    Never mutates temperature/pressure values that passed QC - only
    ever replaces FAILED values with NaN or drops whole profiles whose
    fix/time is untrustworthy.
    """
    accepted = argo_config.ARGO_ACCEPTED_QC_FLAGS
    accepted_pos = argo_config.ARGO_ACCEPTED_POSITION_QC_FLAGS
    accepted_juld = argo_config.ARGO_ACCEPTED_JULD_QC_FLAGS

    pres = argo["pres"].copy()
    temp = argo["temp"].copy()

    level_bad = ~np.isin(argo["pres_qc"], list(accepted)) | \
                ~np.isin(argo["temp_qc"], list(accepted))
    n_bad_levels = int(level_bad.sum())
    pres[level_bad] = np.nan
    temp[level_bad] = np.nan

    profile_ok = np.isin(argo["position_qc"], list(accepted_pos)) & \
                 np.isin(argo["juld_qc"], list(accepted_juld))
    n_dropped_profiles = int((~profile_ok).sum())

    print(f"[argo_processing] QC: {n_bad_levels} individual level "
          f"observations rejected (bad PRES_QC/TEMP_QC); "
          f"{n_dropped_profiles}/{argo['n_profiles']} whole profiles "
          f"dropped (bad POSITION_QC/JULD_QC - fix or time not trustworthy).")

    out = dict(argo)
    out["pres"] = pres[profile_ok]
    out["temp"] = temp[profile_ok]
    out["lat"] = argo["lat"][profile_ok]
    out["lon"] = argo["lon"][profile_ok]
    out["time"] = argo["time"][profile_ok]
    out["pres_qc"] = argo["pres_qc"][profile_ok]
    out["temp_qc"] = argo["temp_qc"][profile_ok]
    out["position_qc"] = argo["position_qc"][profile_ok]
    out["juld_qc"] = argo["juld_qc"][profile_ok]
    if argo["platform_number"] is not None:
        out["platform_number"] = argo["platform_number"][profile_ok]
    out["n_profiles"] = int(profile_ok.sum())
    out["n_qc_rejected_levels"] = n_bad_levels
    out["n_qc_dropped_profiles"] = n_dropped_profiles
    return out


def interpolate_profile_to_target_depths(pres_row: np.ndarray, temp_row: np.ndarray,
                                          target_depths: list,
                                          min_valid_obs: int = None) -> np.ndarray:
    """
    One profile at a time. Returns (len(target_depths),) array.

    - Only levels where BOTH pres and temp are finite are used.
    - If fewer than `min_valid_obs` valid levels exist, returns all-NaN
      (cannot responsibly interpolate at all).
    - For each target depth: if it falls OUTSIDE [min_observed_depth,
      max_observed_depth] for THIS profile, it stays NaN - we never
      extrapolate. Only depths within the observed range are linearly
      interpolated (np.interp, which itself never extrapolates when we
      restrict the query to the observed range).
    """
    if min_valid_obs is None:
        min_valid_obs = argo_config.ARGO_MIN_VALID_OBS_FOR_INTERP

    valid = np.isfinite(pres_row) & np.isfinite(temp_row)
    n_valid = int(valid.sum())
    out = np.full(len(target_depths), np.nan, dtype="float64")
    if n_valid < min_valid_obs:
        return out

    p = pres_row[valid]
    t = temp_row[valid]
    order = np.argsort(p)
    p, t = p[order], t[order]

    # de-duplicate identical pressures (np.interp requires strictly
    # increasing x for well-defined behavior) by averaging temp at ties
    p_unique, inv = np.unique(p, return_inverse=True)
    if len(p_unique) < len(p):
        t_unique = np.array([t[inv == i].mean() for i in range(len(p_unique))])
        p, t = p_unique, t_unique

    if len(p) < min_valid_obs:
        return out

    p_min, p_max = p.min(), p.max()
    for d_idx, depth in enumerate(target_depths):
        if depth < p_min or depth > p_max:
            continue  # outside observed range -> stays NaN, NEVER extrapolated
        out[d_idx] = float(np.interp(depth, p, t))
    return out


def process_all_profiles(argo_qc: dict, target_depths: list) -> dict:
    """
    argo_qc: output of apply_qc().
    Returns argo_qc plus:
        "temp_on_target_depths": (n_profiles, len(target_depths)) float64
        "n_valid_obs_per_profile": (n_profiles,) int - QC-passed levels
        "max_observed_depth_per_profile": (n_profiles,) float64 (NaN if 0 valid obs)
    """
    n_profiles = argo_qc["n_profiles"]
    n_depths = len(target_depths)
    temp_on_target = np.full((n_profiles, n_depths), np.nan, dtype="float64")
    n_valid_obs = np.zeros(n_profiles, dtype=int)
    max_obs_depth = np.full(n_profiles, np.nan, dtype="float64")

    for i in range(n_profiles):
        pres_row = argo_qc["pres"][i]
        temp_row = argo_qc["temp"][i]
        valid = np.isfinite(pres_row) & np.isfinite(temp_row)
        n_valid_obs[i] = int(valid.sum())
        if n_valid_obs[i] > 0:
            max_obs_depth[i] = float(pres_row[valid].max())
        temp_on_target[i] = interpolate_profile_to_target_depths(
            pres_row, temp_row, target_depths
        )

    n_depths_filled = int(np.isfinite(temp_on_target).sum())
    n_depths_total = n_profiles * n_depths
    print(f"[argo_processing] Interpolated {n_profiles} QC-passed profiles "
          f"onto {n_depths} target depths: {n_depths_filled}/{n_depths_total} "
          f"(profile x depth) cells filled "
          f"({100 * n_depths_filled / max(n_depths_total, 1):.1f}%). "
          f"Remaining are NaN because that profile did not observe that "
          f"depth (never extrapolated, never filled).")

    out = dict(argo_qc)
    out["temp_on_target_depths"] = temp_on_target
    out["n_valid_obs_per_profile"] = n_valid_obs
    out["max_observed_depth_per_profile"] = max_obs_depth
    out["target_depths"] = list(target_depths)
    return out
