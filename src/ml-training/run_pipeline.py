"""
run_pipeline.py
================
DAY 1 MAIN ENTRY POINT.

Run:
    cd oceanxray_project
    python run_pipeline.py

This chains every stage exactly in the order described in the brief:

RAW -> load -> inspect -> crop -> temporal align -> horizontal regrid
     -> GLORYS vertical interp -> land/sea mask -> split -> normalize
     -> merge -> save ML-ready dataset + sanity checks

Every stage prints what it did. If something can't be determined from
your actual files, you'll see a line starting with "VERIFY REQUIRED".
Fix config.py accordingly and re-run.
"""

import sys
import traceback
from pathlib import Path

import config
from src import loaders
from src.inspect_data import inspect_all
from src.region import crop_all
from src.temporal_align import align_all
from src.regrid import build_target_grid, regrid_all
from src.glorys_vertical import interpolate_glorys_depths
from src.masking import get_land_sea_mask, apply_mask
from src.splitting import compute_split
from src.normalization import (
    compute_input_stats, normalize_inputs,
    compute_target_stats_per_depth, normalize_target_per_depth, save_stats,
)
from src.assemble import build_input_stack, build_ml_ready_dataset, save_ml_ready, save_metadata
from src.patch_extraction import find_valid_patch_centers
from src.sanity_checks import run_sanity_checks


def main():
    out_dir = config.PROCESSED_DIR

    # ---- 1. LOAD ----
    raw = loaders.load_all()

    # ---- 2. INSPECT ----
    inspect_all(raw)

    # ---- 3. CROP TO REGION ----
    print("\n" + "=" * 70)
    print("CROPPING TO PROJECT REGION")
    print("=" * 70)
    cropped = crop_all(raw)

    # ---- 4. TEMPORAL ALIGNMENT (surface variables only; GLORYS handled after) ----
    print("\n" + "=" * 70)
    print("TEMPORAL ALIGNMENT")
    print("=" * 70)
    surface_only = {k: v for k, v in cropped.items() if k != "glorys"}
    aligned_surface, calendar = align_all(surface_only)

    glorys_cropped = cropped["glorys"]

    # ---- 5. HORIZONTAL REGRIDDING (surface vars + GLORYS) ----
    target_grid = build_target_grid()
    to_regrid = dict(aligned_surface)
    to_regrid["glorys"] = glorys_cropped
    regridded = regrid_all(to_regrid, target_grid)

    glorys_regridded = regridded.pop("glorys")

    # ---- 6. GLORYS VERTICAL INTERPOLATION (native depths -> 15 target depths) ----
    print("\n" + "=" * 70)
    print("GLORYS VERTICAL INTERPOLATION")
    print("=" * 70)
    temperature_target = interpolate_glorys_depths(glorys_regridded)

    # align GLORYS time onto the same daily calendar (exact reindex, no interp)
    temperature_target = temperature_target.reindex(time=calendar)

    # ---- 7. EXTRACT VARIABLE NAMES FROM EACH REGRIDDED SURFACE DATASET ----
    # (each raw dataset may hold >1 variable; we pull out exactly the ones
    #  the project needs, using coord_utils detection done at load time is
    #  not enough for data_vars, so we re-detect here on the regridded ds)
    from src.coord_utils import detect_var_name

    surface_vars = {}

    def _extract(ds, var_key, out_name):
        if ds is None:
            print(f"[assemble] '{out_name}' unavailable (dataset missing).")
            surface_vars[out_name] = None
            return
        vname = detect_var_name(ds, var_key, required=False)
        if vname is None:
            print(f"[assemble] '{out_name}' unavailable (variable not found "
                  f"in dataset - see VERIFY REQUIRED above).")
            surface_vars[out_name] = None
            return
        surface_vars[out_name] = ds[vname].rename(out_name)

    _extract(regridded.get("sst"), "sst", "sst")
    _extract(regridded.get("sss"), "sss", "sss")
    _extract(regridded.get("ssh"), "ssh", "ssh")
    _extract(regridded.get("currents"), "u_current", "u_current")
    _extract(regridded.get("currents"), "v_current", "v_current")
    _extract(regridded.get("winds"), "u_wind", "u_wind")
    _extract(regridded.get("winds"), "v_wind", "v_wind")

    if surface_vars.get("sst") is None:
        raise RuntimeError("[run_pipeline] SST is required and unavailable - "
                            "cannot continue. Fix config.RAW_FILES['sst'].")

    # ---- 8. LAND/SEA MASK (before any split-specific processing/patching) ----
    print("\n" + "=" * 70)
    print("LAND/SEA MASKING")
    print("=" * 70)
    # need a preliminary split just to know the training time range for the
    # fallback mask; compute the real split next and reuse it
    prelim_splits = compute_split(calendar)
    ocean_mask = get_land_sea_mask(raw, surface_vars["sst"], prelim_splits["train"])

    for name in list(surface_vars.keys()):
        if surface_vars[name] is not None:
            surface_vars[name] = apply_mask(surface_vars[name], ocean_mask)
    temperature_target = apply_mask(temperature_target, ocean_mask)

    # ---- 9. TRAIN / VAL / TEST SPLIT ----
    print("\n" + "=" * 70)
    print("TRAIN / VALIDATION / TEST SPLIT")
    print("=" * 70)
    splits = prelim_splits  # already computed and printed above

    # ---- 10. NORMALIZATION (train-only, per-depth for targets) ----
    print("\n" + "=" * 70)
    print("NORMALIZATION (training data only)")
    print("=" * 70)
    input_ds = build_input_stack(surface_vars)
    input_stats = compute_input_stats(input_ds, splits["train"])
    input_ds_norm = normalize_inputs(input_ds, input_stats)

    target_stats = compute_target_stats_per_depth(temperature_target, splits["train"])
    target_norm = normalize_target_per_depth(temperature_target, target_stats)

    save_stats(input_stats, target_stats, out_dir / "normalization_stats.json")

    # ---- 11. MERGE + SAVE ML-READY DATASET ----
    print("\n" + "=" * 70)
    print("ASSEMBLING + SAVING ML-READY DATASET")
    print("=" * 70)
    ml_ready = build_ml_ready_dataset(input_ds_norm, target_norm, ocean_mask)
    save_ml_ready(ml_ready, out_dir)

    channels_used = list(input_ds.data_vars)
    save_metadata(out_dir, splits, channels_used, config_snapshot={})

    # ---- 12. PATCH EXTRACTION PREP (function ready, not run into a CNN) ----
    print("\n" + "=" * 70)
    print("PATCH EXTRACTION PREP")
    print("=" * 70)
    valid_centers = find_valid_patch_centers(ocean_mask, patch_size=config.PATCH_SIZE)

    # ---- 13. SANITY CHECKS ----
    run_sanity_checks(ml_ready, splits, channels_used, ocean_mask, valid_centers,
                       input_stats, out_dir)

    print("\n" + "=" * 70)
    print("DAY 1 PIPELINE COMPLETE")
    print("=" * 70)
    print(f"ML-ready dataset + metadata + stats saved in: {out_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n" + "!" * 70)
        print(f"PIPELINE FAILED: {e}")
        print("!" * 70)
        traceback.print_exc()
        sys.exit(1)
