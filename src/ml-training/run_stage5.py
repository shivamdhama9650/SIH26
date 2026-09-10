"""
run_stage5.py
===============
STAGE 5 MAIN ENTRY POINT — deeper embedding analysis, regional case
study, and the spatial-context ablation.

Run (after Day 1-3 have already produced their outputs; Stage 4 is
optional but its ARGO regional numbers are included if present):
    python run_pipeline.py          # Day 1, if not already done
    python run_day2.py              # Day 2, if not already done
    python run_day3.py              # Day 3 (also produces
                                     #  results/embeddings.npz, used by
                                     #  Part A)
    python run_argo_validation.py   # Stage 4, optional - if present,
                                     #  its ARGO regional numbers are
                                     #  surfaced in Part B
    python run_stage5.py

Does NOT rebuild anything that already works: the Day 3 CNN checkpoint
is loaded and reused as-is (never retrained); Stage 4's ARGO regional
metrics are read as-is (never recomputed). The only NEW model trained
here is the ablation's Center-MLP (Part C), because a fair "spatial
context vs. none" comparison requires a baseline trained on the exact
same sample set the CNN used (see src/ablation.py's docstring for why
Day 2's own MLP checkpoint can't be reused for this specific
comparison).
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
from src.cnn_dataset import find_valid_patch_samples, gather_patch_samples
from src.cnn_model import PatchCNN
from src.metrics import compute_metrics, inverse_transform_target
from src.train_ablation import train_center_mlp
from src.ablation import (
    extract_center_pixel, predict_cnn_degc, predict_center_mlp_degc,
    build_ablation_table, build_depthwise_table, plot_ablation_comparison,
    plot_depthwise_comparison,
)
from src.regional_analysis import (
    compute_glorys_regional_metrics, load_argo_regional_metrics,
    plot_region_rmse_bar, plot_region_depthwise_rmse, plot_representative_profiles,
)
from src.embedding_analysis import (
    load_embeddings, attach_region_and_season, attach_surface_sst, subsample,
    run_pca, run_tsne_optional, make_all_pca_plots,
)
from src.argo_validation import verify_no_argo_leakage

EMBEDDING_MAX_POINTS_FOR_PLOTS = 20000
EMBEDDING_MAX_POINTS_FOR_TSNE = 8000


def verify_no_stage5_test_leakage():
    """Part G, structural (not just documentation): train_center_mlp()'s
    signature only accepts train/val arrays - there is no test argument
    it could even be tuned against. Confirmed here at runtime by
    introspection rather than just asserted in a docstring."""
    import inspect
    sig = inspect.signature(train_center_mlp)
    param_names = list(sig.parameters.keys())
    test_like = [p for p in param_names if "test" in p.lower()]
    assert not test_like, (
        f"[run_stage5] LEAKAGE GUARD TRIPPED: train_center_mlp() accepts "
        f"test-like argument(s) {test_like} - it must only be able to see "
        f"train/val data."
    )
    print("[run_stage5] Leakage guard OK: train_center_mlp() cannot accept "
          "a test-split argument even by mistake (checked by signature "
          "introspection, not just by convention).")


def main():
    print("\n" + "=" * 70)
    print("PART G (checked first): NO DATA LEAKAGE")
    print("=" * 70)
    verify_no_argo_leakage(config.PROJECT_ROOT)  # still applies - Stage 5 must
                                                   # not reintroduce ARGO leakage either
    verify_no_stage5_test_leakage()

    # ---- PART E: real-data-first check ----
    print("\n" + "=" * 70)
    print("PART E: REAL-DATA-FIRST CHECK")
    print("=" * 70)
    ml_ready_path = config.PROCESSED_DIR / "ml_ready_dataset.nc"
    for p in [ml_ready_path, config.PROCESSED_DIR / "normalization_stats.json",
              config.PROCESSED_DIR / "metadata.json"]:
        if not p.exists():
            raise FileNotFoundError(f"[run_stage5] Missing Day 1 output: {p}. "
                                     f"Run run_pipeline.py first.")
    cnn_checkpoint_path = Path("models") / "cnn_best.pt"
    if not cnn_checkpoint_path.exists():
        raise FileNotFoundError(f"[run_stage5] Missing Day 3 CNN checkpoint: "
                                 f"{cnn_checkpoint_path}. Run run_day3.py first.")

    with open(config.PROCESSED_DIR / "metadata.json") as f:
        metadata = json.load(f)
    # NOTE: this repo has no field in metadata.json that self-declares
    # "real" vs "synthetic" data provenance (Day 1 processes whatever is
    # in data/raw/ regardless of its origin). We rely on a printed
    # reminder instead of an inferred flag: if the raw source files were
    # produced by make_synthetic_data.py / make_synthetic_data_smoketest.py,
    # this project's README/handoff docs say so. Whoever runs this for
    # real should confirm their data/raw/ actually holds downloaded
    # GLORYS/satellite files, not synthetic ones, before treating these
    # numbers as science.
    print("[run_stage5] REMINDER: this script cannot itself verify whether "
          "data/raw/ holds real downloaded data or synthetic test data - "
          "check your own data/raw/ provenance. All console/report output "
          "below should be labeled accordingly by whoever runs this.")

    ds = xr.open_dataset(ml_ready_path, chunks={'time': 50})
    with open(config.PROCESSED_DIR / "normalization_stats.json") as f:
        norm_stats = json.load(f)
    channels = [ch for ch in metadata["channels_used"] if ch not in ["temperature_target", "ocean_mask", "u_wind"]]
    target_depths = metadata["target_depths"]
    target_stats_per_depth = norm_stats["target_stats_per_depth"]
    lat_coords = ds["lat"].values
    lon_coords = ds["lon"].values

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =========================================================================
    # PART C: ABLATION - build the CNN's own valid sample set (reused split)
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART C: ABLATION — building samples (same split/samples as Day 3's CNN)")
    print("=" * 70)
    calendar = pd.DatetimeIndex(ds["time"].values)
    splits = compute_split(calendar)

    def build_split_patches(split_name):
        time_index = splits[split_name]
        valid_mask = find_valid_patch_samples(ds, time_index, channels)
        X, Y, idx, sample_times = gather_patch_samples(ds, time_index, channels, valid_mask)
        return X, Y, idx, sample_times

    X_train, Y_train, idx_train, times_train = build_split_patches("train")
    X_val, Y_val, idx_val, times_val = build_split_patches("val")
    X_test, Y_test, idx_test, times_test = build_split_patches("test")
    print(f"[run_stage5] train={X_train.shape[0]}, val={X_val.shape[0]}, "
          f"test={X_test.shape[0]} samples (identical to Day 3's CNN samples).")

    Xc_train = extract_center_pixel(X_train)
    Xc_val = extract_center_pixel(X_val)
    Xc_test = extract_center_pixel(X_test)

    # reuse the EXACT CNN training config (epochs/patience/batch_size/lr)
    # Day 3 used, for a fair comparison, not a re-guessed one.
    with open("models/cnn_training_config.json") as f:
        cnn_train_cfg = json.load(f)["day3_config"]
    ablation_mlp_config = {
        "seed": cnn_train_cfg["seed"], "hidden_dim": cnn_train_cfg["hidden_dim"],
        "batch_size": cnn_train_cfg["batch_size"], "epochs": cnn_train_cfg["epochs"],
        "lr": cnn_train_cfg["lr"], "patience": cnn_train_cfg["patience"],
    }
    center_mlp_model, center_mlp_history, center_mlp_device = train_center_mlp(
        Xc_train, Y_train, Xc_val, Y_val, channels=channels,
        n_output_depths=len(target_depths), config=ablation_mlp_config,
        out_dir=Path("models") / "ablation",
    )

    # load the CNN, reused as-is (never retrained here)
    cnn_state = torch.load(cnn_checkpoint_path, map_location=device, weights_only=False)
    if cnn_state["channels"] != channels:
        raise ValueError(f"[run_stage5] CNN checkpoint channels {cnn_state['channels']} "
                          f"!= current Day 1 channels {channels}.")
    cnn_model = PatchCNN(
        n_input_channels=cnn_state["n_input_channels"], patch_size=cnn_state["patch_size"],
        num_filters=cnn_state["num_filters"], embedding_dim=cnn_state["embedding_dim"],
        hidden_dim=cnn_state["hidden_dim"], n_output_depths=cnn_state["n_output_depths"],
        activation=cnn_state["activation"],
    ).to(device)
    cnn_model.load_state_dict(cnn_state["model_state_dict"])
    cnn_model.eval()
    print(f"[run_stage5] Loaded Day 3 CNN checkpoint (reused, not retrained): "
          f"epoch={cnn_state['epoch']}, val_loss={cnn_state['val_loss']:.5f}")

    Y_val_degc = inverse_transform_target(Y_val, target_depths, target_stats_per_depth)
    Y_test_degc = inverse_transform_target(Y_test, target_depths, target_stats_per_depth)

    center_mlp_val_pred = predict_center_mlp_degc(center_mlp_model, Xc_val, center_mlp_device,
                                                    target_depths, target_stats_per_depth)
    center_mlp_test_pred = predict_center_mlp_degc(center_mlp_model, Xc_test, center_mlp_device,
                                                     target_depths, target_stats_per_depth)
    cnn_val_pred = predict_cnn_degc(cnn_model, X_val, device, target_depths, target_stats_per_depth)
    cnn_test_pred = predict_cnn_degc(cnn_model, X_test, device, target_depths, target_stats_per_depth)

    center_mlp_val_metrics = compute_metrics(center_mlp_val_pred, Y_val_degc, target_depths)
    center_mlp_test_metrics = compute_metrics(center_mlp_test_pred, Y_test_degc, target_depths)
    cnn_val_metrics = compute_metrics(cnn_val_pred, Y_val_degc, target_depths)
    cnn_test_metrics = compute_metrics(cnn_test_pred, Y_test_degc, target_depths)

    center_mlp_n_params = sum(p.numel() for p in center_mlp_model.parameters())
    cnn_n_params = sum(p.numel() for p in cnn_model.parameters())

    ablation_table = build_ablation_table(
        center_mlp_val_metrics, center_mlp_test_metrics, cnn_val_metrics, cnn_test_metrics,
        center_mlp_n_params, cnn_n_params,
    )
    depthwise_table = build_depthwise_table(center_mlp_test_metrics, cnn_test_metrics, target_depths)

    ablation_dir = Path("results") / "ablation"
    ablation_dir.mkdir(parents=True, exist_ok=True)
    ablation_table.to_csv(ablation_dir / "ablation_table.csv", index=False)
    depthwise_table.to_csv(ablation_dir / "ablation_depthwise.csv", index=False)
    with open(ablation_dir / "ablation_summary.json", "w") as f:
        json.dump({
            "center_mlp": {"val": center_mlp_val_metrics, "test": center_mlp_test_metrics,
                            "n_params": center_mlp_n_params, "best_epoch": center_mlp_history["best_epoch"]},
            "cnn": {"val": cnn_val_metrics, "test": cnn_test_metrics, "n_params": cnn_n_params},
            "note": "Val drives model selection (already baked into each checkpoint's "
                    "early stopping); test is final-report-only, per project convention.",
        }, f, indent=2)
    plot_ablation_comparison(ablation_table, ablation_dir / "plots")
    plot_depthwise_comparison(depthwise_table, ablation_dir / "plots")
    print("\n[run_stage5] ABLATION TABLE:")
    print(ablation_table.to_string(index=False))

    # =========================================================================
    # PART B: REGIONAL CASE STUDY (reuses the same CNN test predictions)
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART B: REGIONAL CASE STUDY (Arabian Sea vs Bay of Bengal)")
    print("=" * 70)
    test_lons = lon_coords[idx_test["lon_idx"]]
    glorys_regional = compute_glorys_regional_metrics(cnn_test_pred, Y_test_degc, test_lons, target_depths)

    regional_dir = Path("results") / "regional"
    regional_dir.mkdir(parents=True, exist_ok=True)
    with open(regional_dir / "glorys_regional_metrics.json", "w") as f:
        json.dump(glorys_regional, f, indent=2)
    plot_region_rmse_bar(glorys_regional, regional_dir / "plots")
    plot_region_depthwise_rmse(glorys_regional, target_depths, regional_dir / "plots")
    plot_representative_profiles(cnn_test_pred, Y_test_degc, test_lons, target_depths,
                                  regional_dir / "plots")

    argo_regional = load_argo_regional_metrics(Path("results") / "argo" / "argo_regional_metrics.json")
    with open(regional_dir / "regional_summary.json", "w") as f:
        json.dump({
            "glorys_based_regional_analysis": {
                k: {"n_samples": v.get("n_samples"), "skipped": v.get("skipped"),
                    **({"rmse": v["overall"]["rmse"], "mae": v["overall"]["mae"],
                        "bias": v["overall"]["bias"], "correlation": v["overall"]["correlation"]}
                       if not v.get("skipped", True) else {})}
                for k, v in glorys_regional.items()
            },
            "argo_independent_regional_validation": argo_regional,
            "note": "These two sections are DIFFERENT things: the first is CNN vs "
                    "GLORYS (training/reference target) on the test split; the second "
                    "(if present) is CNN vs independent ARGO observations from Stage 4 "
                    "- read as-is, not recomputed here.",
        }, f, indent=2, default=str)

    # =========================================================================
    # PART A: EMBEDDING ANALYSIS
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART A: DEEPER EMBEDDING ANALYSIS")
    print("=" * 70)
    emb_dir = Path("results") / "embedding"
    emb_dir.mkdir(parents=True, exist_ok=True)

    emb_full = load_embeddings(Path("results") / "embeddings.npz")
    emb_full = attach_region_and_season(emb_full)
    emb_full = attach_surface_sst(emb_full, ds)

    emb = subsample(emb_full, max_points=EMBEDDING_MAX_POINTS_FOR_PLOTS)

    pca_coords, explained_var = run_pca(emb["embeddings"], n_components=2)
    make_all_pca_plots(emb, pca_coords, explained_var, emb_dir / "plots", method_label="PCA")

    tsne_coords, tsne_sel = run_tsne_optional(emb["embeddings"],
                                               max_points_for_tsne=EMBEDDING_MAX_POINTS_FOR_TSNE)
    emb_tsne = {k: v[tsne_sel] for k, v in emb.items()}
    make_all_pca_plots(emb_tsne, tsne_coords, np.array([np.nan, np.nan]),
                        emb_dir / "plots", method_label="tSNE")

    np.savez_compressed(
        emb_dir / "pca_coordinates.npz",
        pca_coords=pca_coords, explained_variance_ratio=explained_var,
        region=emb["region"], season=emb["season"], split=emb["split"],
        lat=emb["lat"], lon=emb["lon"], month=emb["month"],
        sst_normalized=emb["sst_normalized"],
    )
    metadata_df = pd.DataFrame({
        "time": emb["time"], "lat": emb["lat"], "lon": emb["lon"],
        "region": emb["region"], "season": emb["season"], "split": emb["split"],
        "sst_normalized": emb["sst_normalized"],
        "pc1": pca_coords[:, 0], "pc2": pca_coords[:, 1],
    })
    metadata_df.to_csv(emb_dir / "embedding_metadata.csv", index=False)
    with open(emb_dir / "explained_variance.json", "w") as f:
        json.dump({
            "n_points_used_for_pca": int(len(emb["embeddings"])),
            "n_points_total": int(len(emb_full["embeddings"])),
            "explained_variance_ratio": explained_var.tolist(),
            "cumulative_variance_2pc": float(np.sum(explained_var)),
        }, f, indent=2)

    print(f"[run_stage5] PCA explained variance (2 PCs): "
          f"{explained_var[0]*100:.1f}% + {explained_var[1]*100:.1f}% = "
          f"{np.sum(explained_var)*100:.1f}% total.")

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "=" * 40)
    print("STAGE 5 — SUMMARY")
    print("=" * 40)
    print("\n[Ablation]")
    print(ablation_table.to_string(index=False))
    print("\n[Regional - GLORYS-based, test split]")
    for name in ["arabian_sea", "bay_of_bengal"]:
        r = glorys_regional.get(name, {})
        if r.get("skipped", True):
            print(f"  {name}: skipped (0 samples)")
        else:
            print(f"  {name} (n={r['n_samples']}): RMSE={r['overall']['rmse']:.4f}, "
                  f"MAE={r['overall']['mae']:.4f}, Bias={r['overall']['bias']:.4f}, "
                  f"Corr={r['overall']['correlation']:.4f}")
    print("\n[Regional - ARGO independent validation]")
    if argo_regional is None:
        print("  Not available (Stage 4 not yet run).")
    else:
        for name, r in argo_regional.items():
            if r.get("skipped", True):
                print(f"  {name}: skipped ({r.get('n_profiles', 0)} matched profile(s))")
            else:
                print(f"  {name} (n={r['n_profiles']}): RMSE={r['overall']['rmse']:.4f}")
    print(f"\n[Embedding] PCA explained variance: "
          f"{np.sum(explained_var)*100:.1f}% (2 components), "
          f"{len(emb['embeddings'])}/{len(emb_full['embeddings'])} points plotted.")
    print(f"\nResults saved in: {ablation_dir.resolve()}, {regional_dir.resolve()}, "
          f"{emb_dir.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n" + "!" * 70)
        print(f"STAGE 5 FAILED: {e}")
        print("!" * 70)
        traceback.print_exc()
        sys.exit(1)
