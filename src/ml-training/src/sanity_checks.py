"""
sanity_checks.py
=================
Final verification stage. Prints all the checks required by the brief
and produces 3 simple diagnostic plots (not presentation figures).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xarray as xr

from config import TARGET_DEPTHS, GRID_RESOLUTION_DEG


def run_sanity_checks(ml_ready: xr.Dataset, splits: dict, channels_used: list,
                       ocean_mask: xr.DataArray, valid_patch_centers: xr.DataArray,
                       input_stats: dict, out_dir):
    print("\n" + "=" * 70)
    print("SANITY CHECKS")
    print("=" * 70)

    lat = ml_ready["lat"].values
    lon = ml_ready["lon"].values
    print(f"Final latitude range: [{lat.min():.3f}, {lat.max():.3f}]")
    print(f"Final longitude range: [{lon.min():.3f}, {lon.max():.3f}]")

    lat_spacing = float(np.median(np.diff(np.sort(lat))))
    lon_spacing = float(np.median(np.diff(np.sort(lon))))
    print(f"Grid spacing: lat~{lat_spacing:.4f} deg, lon~{lon_spacing:.4f} deg "
          f"(target={GRID_RESOLUTION_DEG})")
    assert abs(lat_spacing - GRID_RESOLUTION_DEG) < 0.05, "Grid spacing check FAILED"
    assert abs(lon_spacing - GRID_RESOLUTION_DEG) < 0.05, "Grid spacing check FAILED"
    print("  -> OK, grid spacing matches target.")

    time_vals = ml_ready["time"].values
    time_diffs = np.diff(np.sort(time_vals)).astype("timedelta64[D]").astype(int)
    print(f"Daily time spacing: median={np.median(time_diffs)} day(s), "
          f"max gap={time_diffs.max() if len(time_diffs) else 'n/a'} day(s)")

    n_depths = ml_ready["depth"].sizes.get("depth", 0) if "depth" in ml_ready.dims else \
        ml_ready["temperature_target"].sizes.get("depth", 0)
    print(f"Number of target depths: {n_depths} (expected 15)")
    assert n_depths == 15, "Target depth count check FAILED"
    print(f"Target depth values: {TARGET_DEPTHS}")

    print(f"Number of input channels actually available: {len(channels_used)} "
          f"-> {channels_used}")

    for name, idx in splits.items():
        print(f"{name.upper()} date range: {idx.min().date()} -> {idx.max().date()} "
              f"({len(idx)} days)")

    for ch in channels_used:
        nan_frac = float(ml_ready[ch].isnull().mean().values)
        print(f"NaN fraction in '{ch}': {nan_frac*100:.2f}%")

    target_nan_frac = float(ml_ready["temperature_target"].isnull().mean().values)
    print(f"NaN fraction in temperature_target: {target_nan_frac*100:.2f}%")

    print("\nNormalized training means/stds (should be ~0 / ~1 on train split):")
    for var, s in input_stats.items():
        print(f"  {var}: mean={s['mean']:.4f} (pre-normalization train mean), "
              f"std={s['std']:.4f} (pre-normalization train std)")

    for ch in channels_used:
        print(f"Shape of X['{ch}']: {ml_ready[ch].shape} (time, lat, lon)")
    print(f"Shape of Y (temperature_target): {ml_ready['temperature_target'].shape} "
          f"(time, depth, lat, lon)")

    n_valid_centers = int(valid_patch_centers.sum().values)
    print(f"\nValid 3x3 patch centers (no land in any patch): {n_valid_centers}")
    assert n_valid_centers > 0, "No valid patch centers found - check masking!"
    print("  -> Confirmed: only these center points may be used for patch extraction.")

    _make_plots(ml_ready, channels_used, out_dir)


def _make_plots(ml_ready: xr.Dataset, channels_used: list, out_dir):
    out_dir = out_dir
    fig_dir = out_dir / "sanity_plots"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. one surface variable map (last available time step)
    if channels_used:
        var = channels_used[0]
        da = ml_ready[var].isel(time=-1)
        plt.figure(figsize=(6, 5))
        da.plot()
        plt.title(f"Surface variable check: {var} (last time step)")
        plt.savefig(fig_dir / "01_surface_variable_map.png", dpi=100, bbox_inches="tight")
        plt.close()

    # 2. one GLORYS target depth map
    target = ml_ready["temperature_target"].isel(time=-1)
    if "depth" in target.dims:
        mid_depth = target["depth"].values[len(target["depth"]) // 2]
        plt.figure(figsize=(6, 5))
        target.sel(depth=mid_depth).plot()
        plt.title(f"GLORYS target check: temperature @ {mid_depth}m (last time step)")
        plt.savefig(fig_dir / "02_glorys_target_map.png", dpi=100, bbox_inches="tight")
        plt.close()

    # 3. one example 15-depth temperature profile at a random ocean point
    mask = ml_ready["ocean_mask"].values
    ocean_idx = np.argwhere(mask)
    if len(ocean_idx) > 0:
        i, j = ocean_idx[len(ocean_idx) // 2]
        profile = ml_ready["temperature_target"].isel(time=-1, lat=i, lon=j)
        plt.figure(figsize=(5, 6))
        plt.plot(profile.values, profile["depth"].values, marker="o")
        plt.gca().invert_yaxis()
        plt.xlabel("Normalized temperature")
        plt.ylabel("Depth (m)")
        plt.title("Example 15-depth temperature profile (normalized)")
        plt.savefig(fig_dir / "03_example_depth_profile.png", dpi=100, bbox_inches="tight")
        plt.close()

    print(f"\n[sanity_checks] Saved 3 diagnostic plots -> {fig_dir}")
