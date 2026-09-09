"""
run_day3.py
============
DAY 3 / STAGE 3 MAIN ENTRY POINT.

Run (after Day 1's run_pipeline.py AND Day 2's run_day2.py have both
already produced their outputs):
    cd oceanxray_project
    python run_pipeline.py   # Day 1, if not already done
    python run_day2.py       # Day 2, if not already done (trains the MLP
                              # checkpoint this script REUSES rather than
                              # retraining)
    python run_day3.py       # Day 3 / Stage 3

Chains:
  load Day 1 output -> reconstruct EXACT Day 1/2 split -> fit climatology
  (train data only, reused from Day 2's src.climatology) -> load the
  ALREADY-TRAINED Day 2 MLP checkpoint (not retrained) -> build 3x3
  patch samples (train/val/test) -> sanity-check tensor shapes -> train
  CNN (early stopping on val) -> evaluate climatology vs MLP vs CNN on
  VALIDATION (drives the gate) -> evaluate once more on TEST (final
  report only) -> extract + save CNN embeddings -> print STAGE 3 RESULT.
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
from src.splitting import compute_split
from src.climatology import fit_climatology
from src.cnn_dataset import find_valid_patch_samples, gather_patch_samples
from src.train_cnn import train_cnn
from src.cnn_evaluation import (
    load_mlp_checkpoint, evaluate_three_way, extract_embeddings,
    save_embeddings, plot_embedding_pca, plot_cnn_training_curve,
    print_stage3_result,
)


CNN_CONFIG = {
    "seed": 42,
    "num_filters": 16,
    "embedding_dim": 32,
    "hidden_dim": 64,
    "activation": "relu",
    "batch_size": 256,
    "epochs": 50,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "patience": 8,
    "optimizer": "adam",
}


def main():
    processed_dir = config.PROCESSED_DIR
    ml_ready_path = processed_dir / "ml_ready_dataset.nc"
    stats_path = processed_dir / "normalization_stats.json"
    metadata_path = processed_dir / "metadata.json"
    mlp_checkpoint_path = Path("models") / "mlp_best.pt"

    for p in [ml_ready_path, stats_path, metadata_path]:
        if not p.exists():
            raise FileNotFoundError(
                f"[run_day3] Missing Day 1 output: {p}. Run run_pipeline.py "
                f"(Day 1) first."
            )
    if not mlp_checkpoint_path.exists():
        raise FileNotFoundError(
            f"[run_day3] Missing Day 2 MLP checkpoint: {mlp_checkpoint_path}. "
            f"Run run_day2.py first — Stage 3 REUSES the trained Day 2 MLP "
            f"for a fair 3-way comparison rather than retraining it."
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

    print(f"[run_day3] Channels: {channels} ({len(channels)} channels) — "
          f"read from Day 1 metadata, never hard-coded.")
    print(f"[run_day3] Target depths: {target_depths} ({len(target_depths)} depths)")
    print(f"[run_day3] Grid: lat={ds.sizes['lat']}, lon={ds.sizes['lon']}, "
          f"time={ds.sizes['time']}")

    # ---- 1. RECONSTRUCT THE EXACT DAY 1/2 SPLIT ----
    print("\n" + "=" * 70)
    print("RECONSTRUCTING DAY 1/2 TRAIN/VAL/TEST SPLIT (reused, not re-invented)")
    print("=" * 70)
    calendar = pd.DatetimeIndex(ds["time"].values)
    splits = compute_split(calendar)

    # ---- 2. CLIMATOLOGY (reused from Day 2, refit is cheap/deterministic) ----
    print("\n" + "=" * 70)
    print("FITTING CLIMATOLOGY BASELINE (training data only, reused from Day 2)")
    print("=" * 70)
    climatology = fit_climatology(ds["temperature_target"], splits["train"])

    # ---- 3. LOAD THE ALREADY-TRAINED DAY 2 MLP (not retrained) ----
    print("\n" + "=" * 70)
    print("LOADING DAY 2 POINT-MLP CHECKPOINT (reused, not retrained)")
    print("=" * 70)
    mlp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mlp_model = load_mlp_checkpoint(mlp_checkpoint_path, mlp_device, channels)

    # ---- 4. BUILD 3x3 PATCH SAMPLES FOR THE CNN (train/val/test) ----
    print("\n" + "=" * 70)
    print("STAGE 3: BUILDING 3x3 PATCH SAMPLES FOR CNN")
    print("=" * 70)

    def build_split_patches(split_name):
        time_index = splits[split_name]
        valid_mask = find_valid_patch_samples(ds, time_index, channels)
        X, Y, idx, sample_times = gather_patch_samples(ds, time_index, channels, valid_mask)
        print(f"[run_day3] '{split_name}': X={X.shape}, Y={Y.shape}")
        return X, Y, idx, sample_times

    X_train, Y_train, idx_train, times_train = build_split_patches("train")
    X_val, Y_val, idx_val, times_val = build_split_patches("val")
    X_test, Y_test, idx_test, times_test = build_split_patches("test")

    # ---- 4b. SANITY-CHECK TENSOR SHAPES (explicit, printed once) ----
    print("\n" + "=" * 70)
    print("TENSOR SHAPE SANITY CHECK")
    print("=" * 70)
    patch_size = X_train.shape[-1]
    half = patch_size // 2
    print(f"Raw patch (one channel):      ({patch_size}, {patch_size})")
    print(f"Multi-channel input (1 sample): {X_train[0].shape}  (channels, {patch_size}, {patch_size})")
    print(f"Batch of patches (train):     {X_train.shape}  (n_samples, channels, {patch_size}, {patch_size})")
    print(f"Target (train):               {Y_train.shape}  (n_samples, 15)")
    assert X_train.shape[1] == len(channels), "Channel dim mismatch!"
    assert X_train.shape[2] == patch_size and X_train.shape[3] == patch_size, "Patch dims mismatch!"
    assert Y_train.shape[1] == len(target_depths), "Depth dim mismatch!"
    print("Shapes verified against channel count / patch size / depth count.")

    # ---- 5. TRAIN THE CNN (early stopping on val) ----
    print("\n" + "=" * 70)
    print("STAGE 3: TRAINING CNN (encoder -> embedding -> MLP decoder)")
    print("=" * 70)
    models_dir = Path("models")
    cnn_model, history, cnn_device = train_cnn(
        X_train, Y_train, X_val, Y_val,
        channels=channels, n_output_depths=len(target_depths),
        config=CNN_CONFIG, out_dir=models_dir,
    )

    print(f"\nCNN embedding shape check: encode() on one batch ->")
    with torch.no_grad():
        sample_batch = torch.from_numpy(X_val[:8]).to(cnn_device)
        emb_check = cnn_model.encode(sample_batch)
        out_check = cnn_model(sample_batch)
        print(f"  input batch:  {tuple(sample_batch.shape)}")
        print(f"  embedding:    {tuple(emb_check.shape)}  (batch, embedding_dim)")
        print(f"  output:       {tuple(out_check.shape)}  (batch, 15)")
        assert emb_check.shape[1] == CNN_CONFIG["embedding_dim"]
        assert out_check.shape[1] == len(target_depths)

    plot_cnn_training_curve(history, Path("results") / "plots")

    with open(models_dir / "cnn_training_config.json", "w") as f:
        json.dump({"day3_config": CNN_CONFIG, "channels": channels,
                   "target_depths": target_depths, "patch_size": patch_size,
                   "best_epoch": history["best_epoch"],
                   "best_val_loss_normalized_mse": history["best_val_loss"]}, f, indent=2)

    n_params = sum(p.numel() for p in cnn_model.parameters())

    # ---- 6. 3-WAY EVALUATION ON VALIDATION -> DRIVES THE GATE ----
    # Per the Day 2 audit fix, model-selection decisions use VALIDATION;
    # this applies identically here.
    print("\n" + "=" * 70)
    print("3-WAY EVALUATION ON VALIDATION SPLIT (drives model-selection decisions)")
    print("=" * 70)
    clim_metrics_val, mlp_metrics_val, cnn_metrics_val, comparison_val = evaluate_three_way(
        climatology, mlp_model, mlp_device, cnn_model, cnn_device,
        X_val, Y_val, times_val, idx_val,
        target_depths, target_stats_per_depth, out_dir=Path("."),
        split_label="val", make_plots=True,
    )

    # ---- 7. FINAL, ONE-TIME TEST EVALUATION (reporting only) ----
    print("\n" + "=" * 70)
    print("FINAL TEST-SPLIT 3-WAY EVALUATION (reporting only — not used for any decision)")
    print("=" * 70)
    clim_metrics_test, mlp_metrics_test, cnn_metrics_test, comparison_test = evaluate_three_way(
        climatology, mlp_model, mlp_device, cnn_model, cnn_device,
        X_test, Y_test, times_test, idx_test,
        target_depths, target_stats_per_depth, out_dir=Path("."),
        split_label="test", make_plots=False,
    )

    # ---- 8. EMBEDDING EXTRACTION (metadata for later PCA/t-SNE/UMAP) ----
    print("\n" + "=" * 70)
    print("EXTRACTING CNN EMBEDDINGS")
    print("=" * 70)
    lat_coords = ds["lat"].values
    lon_coords = ds["lon"].values
    # extract for train+val+test combined so later analysis has full coverage
    all_embeddings = []
    for split_name, X_s, idx_s, times_s in [
        ("train", X_train, idx_train, times_train),
        ("val", X_val, idx_val, times_val),
        ("test", X_test, idx_test, times_test),
    ]:
        emb = extract_embeddings(cnn_model, X_s, cnn_device, times_s, idx_s,
                                  lat_coords=lat_coords, lon_coords=lon_coords)
        emb["split"] = np.array([split_name] * len(times_s))
        all_embeddings.append(emb)

    combined = {}
    for key in all_embeddings[0]:
        combined[key] = np.concatenate([e[key] for e in all_embeddings], axis=0)
    save_embeddings(combined, Path("results") / "embeddings.npz")
    plot_embedding_pca(combined, Path("results") / "plots")

    # ---- 9. FINAL RESULT ----
    print_stage3_result(comparison_val, comparison_test)

    print(f"\nCNN checkpoint:     {(models_dir / 'cnn_best.pt').resolve()}")
    print(f"Embeddings:         {(Path('results') / 'embeddings.npz').resolve()}")
    print(f"Model parameters:   {n_params}")
    print(f"Results saved in:   {Path('results').resolve()}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n" + "!" * 70)
        print(f"DAY 3 PIPELINE FAILED: {e}")
        print("!" * 70)
        traceback.print_exc()
        sys.exit(1)
