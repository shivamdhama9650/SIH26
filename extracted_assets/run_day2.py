"""
run_day2.py
============
DAY 2 MAIN ENTRY POINT.

Run (after Day 1's run_pipeline.py has produced data/processed/ml_ready_dataset.nc):
    cd oceanxray_project
    python run_day2.py

Chains:
  load Day 1 output -> reconstruct EXACT Day 1 split -> fit climatology
  (train data only) -> build MLP point samples (train/val/test) -> train
  MLP (early stopping on val) -> evaluate climatology + MLP on the TEST
  split (both normalized and degC space) -> save models/results/plots
  -> print the DAY 2 GATE.

CNN is NOT implemented here - Day 2 stops at climatology + MLP only.
"""

import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import config
from src.splitting import compute_split
from src.climatology import fit_climatology
from src.mlp_dataset import find_valid_point_samples, gather_point_samples
from src.train_mlp import train_mlp
from src.evaluation import evaluate_all, plot_training_curve, print_gate


DAY2_CONFIG = {
    "seed": 42,
    "hidden_dim": 64,
    "batch_size": 256,
    "epochs": 50,
    "lr": 1e-3,
    "patience": 8,
}


def main():
    processed_dir = config.PROCESSED_DIR
    ml_ready_path = processed_dir / "ml_ready_dataset.nc"
    stats_path = processed_dir / "normalization_stats.json"
    metadata_path = processed_dir / "metadata.json"

    for p in [ml_ready_path, stats_path, metadata_path]:
        if not p.exists():
            raise FileNotFoundError(
                f"[run_day2] Missing Day 1 output: {p}. Run run_pipeline.py "
                f"(Day 1) first."
            )

    print("\n" + "=" * 70)
    print("LOADING DAY 1 OUTPUT")
    print("=" * 70)
    ds = xr.open_dataset(ml_ready_path, chunks={'time': 50})
    with open(stats_path) as f:
        norm_stats = json.load(f)
    with open(metadata_path) as f:
        metadata = json.load(f)

    channels = [ch for ch in metadata["channels_used"] if ch not in ["temperature_target", "ocean_mask", "u_wind"]]
    target_depths = metadata["target_depths"]
    target_stats_per_depth = norm_stats["target_stats_per_depth"]

    print(f"[run_day2] Channels: {channels} ({len(channels)} channels)")
    print(f"[run_day2] Target depths: {target_depths} ({len(target_depths)} depths)")
    print(f"[run_day2] Grid: lat={ds.sizes['lat']}, lon={ds.sizes['lon']}, "
          f"time={ds.sizes['time']}")

    # ---- 1. RECONSTRUCT THE EXACT DAY 1 SPLIT ----
    # We recompute via the SAME splitting.compute_split() function Day 1
    # used, applied to this dataset's actual time coordinate. This is more
    # robust than parsing metadata.json's start/end strings, since
    # whole_year mode with non-contiguous years wouldn't round-trip
    # through a simple date range.
    print("\n" + "=" * 70)
    print("RECONSTRUCTING DAY 1 TRAIN/VAL/TEST SPLIT")
    print("=" * 70)
    calendar = pd.DatetimeIndex(ds["time"].values)
    splits = compute_split(calendar)

    # ---- 2. CLIMATOLOGY (Stage 1) — TRAIN DATA ONLY ----
    print("\n" + "=" * 70)
    print("STAGE 1: FITTING CLIMATOLOGY BASELINE (training data only)")
    print("=" * 70)
    climatology = fit_climatology(ds["temperature_target"], splits["train"])

    # ---- 3. BUILD POINT SAMPLES FOR MLP (train/val/test) ----
    print("\n" + "=" * 70)
    print("STAGE 2: BUILDING POINT SAMPLES FOR MLP")
    print("=" * 70)

    def build_split_samples(split_name):
        time_index = splits[split_name]
        valid_mask = find_valid_point_samples(ds, time_index, channels)
        X, Y, idx, sample_times = gather_point_samples(ds, time_index, channels, valid_mask)
        print(f"[run_day2] '{split_name}': X={X.shape}, Y={Y.shape}")
        return X, Y, idx, sample_times

    X_train, Y_train, idx_train, times_train = build_split_samples("train")
    X_val, Y_val, idx_val, times_val = build_split_samples("val")
    X_test, Y_test, idx_test, times_test = build_split_samples("test")

    # ---- 4. TRAIN THE MLP (early stopping on val) ----
    print("\n" + "=" * 70)
    print("STAGE 2: TRAINING POINT-BASED MLP")
    print("=" * 70)
    models_dir = Path("models")
    mlp_model, history, device = train_mlp(
        X_train, Y_train, X_val, Y_val,
        channels=channels, n_output_depths=len(target_depths),
        config=DAY2_CONFIG, out_dir=models_dir,
    )

    results_dir = Path("results")
    plot_training_curve(history, Path("."))

    # save the training config + history alongside the model for reproducibility
    with open(models_dir / "training_config.json", "w") as f:
        json.dump({"day2_config": DAY2_CONFIG, "channels": channels,
                   "target_depths": target_depths,
                   "best_epoch": history["best_epoch"],
                   "best_val_loss_normalized_mse": history["best_val_loss"]}, f, indent=2)

    # ---- 5. EVALUATE ON VALIDATION SPLIT -> THIS IS WHAT THE GATE USES ----
    # Per the locked architecture (Section 5, hard gate): "MLP measurably
    # beats climatology on the VALIDATION split." Validation is for
    # model/approach-selection decisions (like whether to proceed to
    # CNN); TEST must stay untouched by any decision-making and is only
    # used once, below, for a final unbiased report.
    print("\n" + "=" * 70)
    print("EVALUATING CLIMATOLOGY + MLP ON VALIDATION SPLIT (drives the gate)")
    print("=" * 70)
    clim_metrics_val, mlp_metrics_val, comparison_val = evaluate_all(
        climatology, mlp_model, device,
        X_val, Y_val, times_val, idx_val,
        target_depths, target_stats_per_depth, out_dir=Path("."),
        split_label="val", make_plots=True,
    )

    print(f"\n[run_day2] Climatology overall (degC, VAL): RMSE={clim_metrics_val['overall']['rmse']:.4f}, "
          f"MAE={clim_metrics_val['overall']['mae']:.4f}, Bias={clim_metrics_val['overall']['bias']:.4f}, "
          f"Corr={clim_metrics_val['overall']['correlation']:.4f}")
    print(f"[run_day2] MLP overall (degC, VAL):         RMSE={mlp_metrics_val['overall']['rmse']:.4f}, "
          f"MAE={mlp_metrics_val['overall']['mae']:.4f}, Bias={mlp_metrics_val['overall']['bias']:.4f}, "
          f"Corr={mlp_metrics_val['overall']['correlation']:.4f}")

    # ---- 6. FINAL, ONE-TIME TEST EVALUATION (reporting only, no decisions) ----
    print("\n" + "=" * 70)
    print("FINAL TEST-SPLIT EVALUATION (reporting only — not used for the gate)")
    print("=" * 70)
    clim_metrics_test, mlp_metrics_test, comparison_test = evaluate_all(
        climatology, mlp_model, device,
        X_test, Y_test, times_test, idx_test,
        target_depths, target_stats_per_depth, out_dir=Path("."),
        split_label="test", make_plots=False,
    )
    print(f"[run_day2] Climatology overall (degC, TEST): RMSE={clim_metrics_test['overall']['rmse']:.4f}")
    print(f"[run_day2] MLP overall (degC, TEST):         RMSE={mlp_metrics_test['overall']['rmse']:.4f}")

    # ---- 7. FINAL GATE (based on VALIDATION, per the locked architecture) ----
    print_gate(comparison_val)

    print(f"\nModels saved in:  {models_dir.resolve()}")
    print(f"Results saved in: {results_dir.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n" + "!" * 70)
        print(f"DAY 2 PIPELINE FAILED: {e}")
        print("!" * 70)
        traceback.print_exc()
        sys.exit(1)
