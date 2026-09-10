"""
run_argo_validation.py
=========================
STAGE 4 MAIN ENTRY POINT — independent ARGO validation.

Run (after Day 1-3 have already produced their outputs):
    python run_pipeline.py     # Day 1, if not already done
    python run_day2.py         # Day 2, if not already done
    python run_day3.py         # Day 3 (produces models/cnn_best.pt,
                                #  the FINAL LOCKED CNN this script uses)
    python make_synthetic_argo_fixture.py   # first time only, for the
                                             # mandatory pipeline smoke test
    python run_argo_validation.py

ARGO is used ONLY for post-hoc evaluation here. It is never used for
training, normalization, hyperparameter tuning, model selection,
checkpoint selection, or early stopping - see
src.argo_validation.verify_no_argo_leakage() (Step 8), which is
checked first, before anything else runs.

Data source: tries REAL Argo data first (argo_config.ARGO_REAL_FILES_GLOB).
If none is found, falls back to the SYNTHETIC fixture and labels every
output accordingly - never fabricates real-looking results.
"""

import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr

import config
import argo_config
from src.argo_loader import load_real_argo, load_synthetic_argo, ArgoLoadError
from src.argo_processing import apply_qc, process_all_profiles
from src.argo_matching import match_profiles, extract_patches_for_matches
from src.argo_validation import (
    verify_no_argo_leakage, load_final_cnn_checkpoint, predict_cnn,
    compute_masked_metrics, regional_split, plot_depthwise_metric,
    plot_matched_locations, plot_example_profiles,
)
from src.metrics import inverse_transform_target


def main():
    out_dir = argo_config.ARGO_RESULTS_DIR
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # ---- STEP 8 (checked FIRST, not last): no data leakage guard ----
    print("\n" + "=" * 70)
    print("STEP 8: DATA-LEAKAGE GUARD (checked before anything else runs)")
    print("=" * 70)
    verify_no_argo_leakage(config.PROJECT_ROOT)

    # ---- load Day 1-3 outputs (read-only) ----
    for p in [argo_config.ML_READY_DATASET_PATH, argo_config.NORMALIZATION_STATS_PATH,
              argo_config.METADATA_PATH]:
        if not p.exists():
            raise FileNotFoundError(f"[run_argo_validation] Missing Day 1 output: {p}. "
                                     f"Run run_pipeline.py first.")
    if not argo_config.CNN_CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"[run_argo_validation] Missing Day 3 CNN checkpoint: "
            f"{argo_config.CNN_CHECKPOINT_PATH}. Run run_day3.py first."
        )

    ds = xr.open_dataset(argo_config.ML_READY_DATASET_PATH)
    with open(argo_config.NORMALIZATION_STATS_PATH) as f:
        norm_stats = json.load(f)
    with open(argo_config.METADATA_PATH) as f:
        metadata = json.load(f)

    channels = metadata["channels_used"]
    target_depths = metadata["target_depths"]
    target_stats_per_depth = norm_stats["target_stats_per_depth"]
    print(f"[run_argo_validation] Channels (from Day 1 metadata, never "
          f"hard-coded): {channels}")
    print(f"[run_argo_validation] Target depths: {target_depths}")
    print(f"[run_argo_validation] ml_ready_dataset time coverage: "
          f"{pd.Timestamp(ds.time.values.min()).date()} -> "
          f"{pd.Timestamp(ds.time.values.max()).date()}")

    # ---- STEP 1/9/10: ARGO DATA DISCOVERY - real first, else synthetic ----
    print("\n" + "=" * 70)
    print("STEP 1 / 9 / 10: ARGO DATA DISCOVERY")
    print("=" * 70)
    using_synthetic = False
    try:
        argo_raw = load_real_argo()
        print("[run_argo_validation] Using REAL ARGO data.")
    except ArgoLoadError as e:
        print(f"[run_argo_validation] No real ARGO data found ({e}).")
        print("[run_argo_validation] Falling back to the SYNTHETIC fixture "
              "for a pipeline smoke test. Results below are NOT real "
              "validation results - see 'data_source' in "
              "validation_summary.json.")
        if not argo_config.ARGO_SYNTHETIC_FIXTURE_PATH.exists():
            raise FileNotFoundError(
                f"[run_argo_validation] No real ARGO data AND no synthetic "
                f"fixture at {argo_config.ARGO_SYNTHETIC_FIXTURE_PATH}. Run "
                f"make_synthetic_argo_fixture.py first (needs Day 1's "
                f"ml_ready_dataset.nc to already exist)."
            )
        argo_raw = load_synthetic_argo()
        using_synthetic = True

    # ---- STEP 2: QC + depth interpolation ----
    print("\n" + "=" * 70)
    print("STEP 2: ARGO PROFILE PROCESSING (QC + depth interpolation)")
    print("=" * 70)
    argo_qc = apply_qc(argo_raw)
    argo_processed = process_all_profiles(argo_qc, target_depths)

    # ---- STEP 3: SPATIAL + TEMPORAL MATCHING ----
    print("\n" + "=" * 70)
    print("STEP 3: SPATIAL + TEMPORAL MATCHING")
    print("=" * 70)
    matches = match_profiles(argo_processed, ds)
    matches.to_csv(out_dir / "matched_profiles.csv", index=False)
    print(f"[run_argo_validation] Saved matching diagnostics -> "
          f"{out_dir / 'matched_profiles.csv'}")
    plot_matched_locations(matches, plots_dir)

    # ---- STEP 4: FINAL CNN PREDICTION ----
    print("\n" + "=" * 70)
    print("STEP 4: FINAL CNN PREDICTION (checkpoint reused, never retrained)")
    print("=" * 70)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn_model = load_final_cnn_checkpoint(argo_config.CNN_CHECKPOINT_PATH, device, channels)

    X, matched_argo_idx, matches = extract_patches_for_matches(ds, channels, matches)
    matches.to_csv(out_dir / "matched_profiles.csv", index=False)  # re-save with patch_valid col

    n_final = len(matched_argo_idx)
    print(f"[run_argo_validation] {n_final} profile(s) have a usable, "
          f"finite 3x3 input patch and will be scored.")

    summary = {
        "data_source": "synthetic_fixture" if using_synthetic else "real_argo",
        "profiles_available": int(argo_raw["n_profiles"]),
        "profiles_after_qc": int(argo_processed["n_profiles"]),
        "profiles_matched_space_time": int(matches["matched"].sum()),
        "profiles_final_scored": int(n_final),
    }

    if n_final == 0:
        print("\n[run_argo_validation] No profiles survived matching + patch "
              "validity - nothing to score. Writing summary-only output.")
        with open(out_dir / "validation_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(json.dumps(summary, indent=2, default=str))
        return

    cnn_pred_norm = predict_cnn(cnn_model, X, device)
    cnn_pred_degc = inverse_transform_target(cnn_pred_norm, target_depths, target_stats_per_depth)

    # ---- STEP 5: COMPARE AGAINST ARGO ----
    print("\n" + "=" * 70)
    print("STEP 5: COMPARE AGAINST ARGO (independent observations)")
    print("=" * 70)
    argo_temp_degc = argo_processed["temp_on_target_depths"][matched_argo_idx]
    overall_metrics = compute_masked_metrics(cnn_pred_degc, argo_temp_degc, target_depths)
    print(f"[run_argo_validation] Overall (ARGO-independent): "
          f"RMSE={overall_metrics['overall']['rmse']:.4f} degC, "
          f"MAE={overall_metrics['overall']['mae']:.4f} degC, "
          f"Bias={overall_metrics['overall']['bias']:.4f} degC, "
          f"Corr={overall_metrics['overall']['correlation']:.4f}, "
          f"n_valid_obs={overall_metrics['overall']['n_valid_obs']}")

    with open(out_dir / "argo_metrics.json", "w") as f:
        json.dump({
            "note": "CNN vs ARGO (independent observations) - NOT GLORYS. "
                     "For CNN vs GLORYS (the training/reference target), "
                     "see results/comparison_3way_test.json from run_day3.py.",
            "data_source": summary["data_source"],
            "overall": overall_metrics["overall"],
            "per_depth": overall_metrics["per_depth"],
        }, f, indent=2)

    depth_rows = []
    for d in target_depths:
        m = overall_metrics["per_depth"][str(float(d))]
        depth_rows.append({"depth_m": d, **m})
    pd.DataFrame(depth_rows).to_csv(out_dir / "argo_depth_metrics.csv", index=False)

    plot_depthwise_metric(overall_metrics, target_depths, plots_dir, "rmse",
                           "argo_depthwise_rmse.png", "Depth-wise RMSE: CNN vs ARGO")
    plot_depthwise_metric(overall_metrics, target_depths, plots_dir, "mae",
                           "argo_depthwise_mae.png", "Depth-wise MAE: CNN vs ARGO")
    plot_example_profiles(cnn_pred_degc, argo_temp_degc, target_depths, plots_dir)

    # ---- STEP 7: REGIONAL CASE STUDY ----
    print("\n" + "=" * 70)
    print("STEP 7: REGIONAL CASE STUDY (Arabian Sea vs Bay of Bengal)")
    print("=" * 70)
    scored_lons = matches.loc[matches["patch_valid"], "argo_lon"].values
    # matches rows are in argo_idx order but only patch_valid ones were
    # scored - reorder to match matched_argo_idx exactly
    lon_by_argo_idx = dict(zip(matches["argo_idx"], matches["argo_lon"]))
    scored_lons = np.array([lon_by_argo_idx[i] for i in matched_argo_idx])

    regions = regional_split(scored_lons)
    regional_metrics = {}
    for region_name, mask in regions.items():
        n_region = int(mask.sum())
        if n_region < argo_config.REGIONAL_MIN_MATCHED_PROFILES:
            print(f"[run_argo_validation] '{region_name}': only {n_region} "
                  f"matched profile(s) (< "
                  f"{argo_config.REGIONAL_MIN_MATCHED_PROFILES}) - skipping "
                  f"regional comparison for this region rather than forcing it.")
            regional_metrics[region_name] = {"n_profiles": n_region, "skipped": True}
            continue
        m = compute_masked_metrics(cnn_pred_degc[mask], argo_temp_degc[mask], target_depths)
        regional_metrics[region_name] = {"n_profiles": n_region, "skipped": False, **m}
        print(f"[run_argo_validation] '{region_name}' (n={n_region}): "
              f"RMSE={m['overall']['rmse']:.4f}, MAE={m['overall']['mae']:.4f}, "
              f"Bias={m['overall']['bias']:.4f}, Corr={m['overall']['correlation']:.4f}")

    with open(out_dir / "argo_regional_metrics.json", "w") as f:
        json.dump(regional_metrics, f, indent=2, default=str)

    # ---- STEP 6: SCIENTIFIC REPORT / validation_summary.json ----
    print("\n" + "=" * 70)
    print("STEP 6: SCIENTIFIC REPORT")
    print("=" * 70)
    summary.update({
        "overall_metrics_degC": overall_metrics["overall"],
        "depth_wise_n_valid_obs": {
            d: overall_metrics["per_depth"][str(float(d))]["n_valid_obs"]
            for d in target_depths
        },
        "regional": {k: {"n_profiles": v["n_profiles"], "skipped": v["skipped"],
                          **({"rmse": v["overall"]["rmse"], "mae": v["overall"]["mae"],
                              "bias": v["overall"]["bias"],
                              "correlation": v["overall"]["correlation"]}
                             if not v["skipped"] else {})}
                     for k, v in regional_metrics.items()},
        "matching_tolerances": {
            "temporal_days": argo_config.ARGO_TEMPORAL_TOLERANCE_DAYS,
            "spatial_km": argo_config.ARGO_SPATIAL_TOLERANCE_KM,
        },
        "qc_rule": f"accepted flags {sorted(argo_config.ARGO_ACCEPTED_QC_FLAGS)}",
        "distinguishing_note": "This is CNN vs ARGO (independent observations), "
                                "separate from CNN vs GLORYS (training/reference "
                                "target) reported in results/comparison_3way_test.json.",
    })
    with open(out_dir / "validation_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n" + "=" * 40)
    print("STAGE 4 — ARGO VALIDATION")
    print("=" * 40)
    print(f"Data source: {summary['data_source']}"
          f"{'  <-- SYNTHETIC, NOT REAL RESULTS' if using_synthetic else ''}")
    print(f"\nProfiles available: {summary['profiles_available']}")
    print(f"Profiles matched:   {summary['profiles_matched_space_time']}")
    print(f"Profiles scored:    {summary['profiles_final_scored']}")
    print("\nValid observations by depth:")
    for d in target_depths:
        print(f"  {d:>6.0f} m: {summary['depth_wise_n_valid_obs'][d]}")
    print("\nOverall:")
    print(f"RMSE:        {overall_metrics['overall']['rmse']:.4f} degC")
    print(f"MAE:         {overall_metrics['overall']['mae']:.4f} degC")
    print(f"Bias:        {overall_metrics['overall']['bias']:.4f} degC")
    print(f"Correlation: {overall_metrics['overall']['correlation']:.4f}")
    print("\nArabian Sea vs Bay of Bengal:")
    for region_name, m in regional_metrics.items():
        if m["skipped"]:
            print(f"  {region_name}: skipped (only {m['n_profiles']} matched profile(s))")
        else:
            print(f"  {region_name} (n={m['n_profiles']}): RMSE={m['overall']['rmse']:.4f}, "
                  f"MAE={m['overall']['mae']:.4f}, Bias={m['overall']['bias']:.4f}, "
                  f"Corr={m['overall']['correlation']:.4f}")
    print("=" * 40)
    print(f"\nResults saved in: {out_dir.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n" + "!" * 70)
        print(f"STAGE 4 (ARGO VALIDATION) FAILED: {e}")
        print("!" * 70)
        traceback.print_exc()
        sys.exit(1)
