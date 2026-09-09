"""
cnn_evaluation.py
==================
Stage: DAY 3 (Stage 3) — ties everything together for the 3-way
comparison: Climatology vs Point MLP vs CNN.

FAIR COMPARISON NOTE (important): the CNN can only use a point as a
sample if its FULL 3x3 neighborhood is valid (see cnn_dataset.py) —
this is a STRICTER requirement than the point-MLP's own validity rule
from Day 2 (which only needs the center point itself to be valid). So
evaluating each model on ITS OWN separately-computed valid-sample set
would compare them on different points, which is not a fair comparison.

Instead, this module always evaluates all three models on the CNN's
(stricter) valid sample set. The point-MLP's single-pixel input for
these exact points is derived directly from the CNN patch's own center
pixel (X_cnn[:, :, half, half]) — since cnn_dataset.py already asserts
the patch center equals the point value at that (time, lat, lon), this
is mathematically identical to what Day 2's own gather would have
produced, with zero extra data access and guaranteed alignment.

Reuses (does not duplicate) src.metrics.compute_metrics and
src.metrics.inverse_transform_target from Day 2.
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.metrics import compute_metrics, inverse_transform_target
from src.mlp_model import PointMLP


def load_mlp_checkpoint(checkpoint_path: Path, device, expected_channels: list):
    """
    Loads the ALREADY-TRAINED Day 2 point-MLP checkpoint rather than
    retraining it (reuse, not duplication). Fails loudly if the
    checkpoint's channel list doesn't match what Day 1/3 report now —
    a silent mismatch here would misalign every input feature.
    """
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if state["channels"] != expected_channels:
        raise ValueError(
            f"[cnn_evaluation] Day 2 MLP checkpoint was trained with channels "
            f"{state['channels']} but the current Day 1 dataset reports "
            f"channels {expected_channels}. Retrain the Day 2 MLP (run_day2.py) "
            f"against the current data before running Day 3."
        )
    model = PointMLP(n_input_channels=state["n_input_channels"],
                      n_output_depths=state["n_output_depths"],
                      hidden_dim=state["hidden_dim"]).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    print(f"[cnn_evaluation] Loaded Day 2 MLP checkpoint from {checkpoint_path} "
          f"(trained to epoch {state['epoch']}, val_loss={state['val_loss']:.5f}).")
    return model


def _batched_predict(model, X: np.ndarray, device, batch_size: int = 2048) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i:i + batch_size]).to(device)
            preds.append(model(xb).cpu().numpy())
    return np.concatenate(preds, axis=0) if preds else np.empty((0,))


def evaluate_three_way(climatology, mlp_model, mlp_device, cnn_model, cnn_device,
                        X_cnn_eval, Y_eval_norm, sample_times, idx,
                        target_depths, target_stats_per_depth, out_dir: Path,
                        split_label: str, make_plots: bool = True):
    """
    X_cnn_eval: (n_samples, n_channels, patch_size, patch_size) — the CNN's
        own valid patch samples for this split. This determines the common
        evaluation point set for ALL THREE models.
    Y_eval_norm: (n_samples, 15) normalized target at those same points.
    sample_times, idx: from cnn_dataset.gather_patch_samples for this split.

    Returns (clim_metrics_degc, mlp_metrics_degc, cnn_metrics_degc, comparison).
    """
    out_dir = Path(out_dir)
    results_dir = out_dir / "results"
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    half = X_cnn_eval.shape[-1] // 2
    X_mlp_eval = X_cnn_eval[:, :, half, half]  # derive point features from patch center

    # ---- predictions (normalized space) ----
    clim_pred_norm = climatology.predict(sample_times, idx["lat_idx"], idx["lon_idx"])
    mlp_pred_norm = _batched_predict(mlp_model, X_mlp_eval, mlp_device)
    cnn_pred_norm = _batched_predict(cnn_model, X_cnn_eval, cnn_device)

    # ---- metrics in normalized space (debugging only) ----
    clim_metrics_norm = compute_metrics(clim_pred_norm, Y_eval_norm, target_depths)
    mlp_metrics_norm = compute_metrics(mlp_pred_norm, Y_eval_norm, target_depths)
    cnn_metrics_norm = compute_metrics(cnn_pred_norm, Y_eval_norm, target_depths)

    # ---- inverse-transform to degC (the real scientific numbers) ----
    clim_pred_degc = inverse_transform_target(clim_pred_norm, target_depths, target_stats_per_depth)
    mlp_pred_degc = inverse_transform_target(mlp_pred_norm, target_depths, target_stats_per_depth)
    cnn_pred_degc = inverse_transform_target(cnn_pred_norm, target_depths, target_stats_per_depth)
    target_degc = inverse_transform_target(Y_eval_norm, target_depths, target_stats_per_depth)

    clim_metrics_degc = compute_metrics(clim_pred_degc, target_degc, target_depths)
    mlp_metrics_degc = compute_metrics(mlp_pred_degc, target_degc, target_depths)
    cnn_metrics_degc = compute_metrics(cnn_pred_degc, target_degc, target_depths)

    # ---- save JSON (prefixed by split) ----
    with open(results_dir / f"climatology_metrics_{split_label}.json", "w") as f:
        json.dump({"normalized_space": clim_metrics_norm, "degC_space": clim_metrics_degc}, f, indent=2)
    with open(results_dir / f"mlp_metrics_{split_label}.json", "w") as f:
        json.dump({"normalized_space": mlp_metrics_norm, "degC_space": mlp_metrics_degc}, f, indent=2)
    with open(results_dir / f"cnn_metrics_{split_label}.json", "w") as f:
        json.dump({"normalized_space": cnn_metrics_norm, "degC_space": cnn_metrics_degc}, f, indent=2)

    cnn_beats_mlp = cnn_metrics_degc["overall"]["rmse"] < mlp_metrics_degc["overall"]["rmse"]
    cnn_beats_climatology = cnn_metrics_degc["overall"]["rmse"] < clim_metrics_degc["overall"]["rmse"]
    comparison = {
        "split_used": split_label,
        "n_eval_samples": int(X_cnn_eval.shape[0]),
        "climatology_rmse_degC": clim_metrics_degc["overall"]["rmse"],
        "mlp_rmse_degC": mlp_metrics_degc["overall"]["rmse"],
        "cnn_rmse_degC": cnn_metrics_degc["overall"]["rmse"],
        "climatology_mae_degC": clim_metrics_degc["overall"]["mae"],
        "mlp_mae_degC": mlp_metrics_degc["overall"]["mae"],
        "cnn_mae_degC": cnn_metrics_degc["overall"]["mae"],
        "cnn_beats_mlp": bool(cnn_beats_mlp),
        "cnn_beats_climatology": bool(cnn_beats_climatology),
    }
    with open(results_dir / f"comparison_3way_{split_label}.json", "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"[cnn_evaluation] Saved '{split_label}' 3-way metrics + comparison -> {results_dir}")

    if make_plots:
        _plot_rmse_bar(clim_metrics_degc, mlp_metrics_degc, cnn_metrics_degc, plots_dir)
        _plot_depthwise(clim_metrics_degc, mlp_metrics_degc, cnn_metrics_degc,
                         target_depths, plots_dir, metric_key="rmse",
                         filename="02_depthwise_rmse_3way.png")
        _plot_depthwise(clim_metrics_degc, mlp_metrics_degc, cnn_metrics_degc,
                         target_depths, plots_dir, metric_key="mae",
                         filename="03_depthwise_mae_3way.png")
        _plot_example_profile(clim_pred_degc, mlp_pred_degc, cnn_pred_degc,
                               target_degc, target_depths, plots_dir)
        _plot_scatter(cnn_pred_degc, target_degc, plots_dir)

    return clim_metrics_degc, mlp_metrics_degc, cnn_metrics_degc, comparison


def extract_embeddings(cnn_model, X_cnn: np.ndarray, device, sample_times,
                        idx: dict, lat_coords=None, lon_coords=None,
                        batch_size: int = 2048) -> dict:
    """
    Reusable embedding extractor: model.encode(x) for every sample in
    X_cnn, plus enough metadata (time, lat, lon) to later color a
    PCA/t-SNE/UMAP plot by region/season/surface-state.

    Returns a dict of numpy/plain-python arrays, ready to np.savez(...).
    """
    cnn_model.eval()
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(X_cnn), batch_size):
            xb = torch.from_numpy(X_cnn[i:i + batch_size]).to(device)
            emb = cnn_model.encode(xb).cpu().numpy()
            embeddings.append(emb)
    embeddings = np.concatenate(embeddings, axis=0) if embeddings else \
        np.empty((0, cnn_model.embedding_dim))

    out = {
        "embeddings": embeddings,
        "time": np.array([str(t.date()) for t in sample_times]),
        "month": np.array([t.month for t in sample_times]),
        "lat_idx": idx["lat_idx"],
        "lon_idx": idx["lon_idx"],
    }
    if lat_coords is not None:
        out["lat"] = np.asarray(lat_coords)[idx["lat_idx"]]
    if lon_coords is not None:
        out["lon"] = np.asarray(lon_coords)[idx["lon_idx"]]
    return out


def save_embeddings(embedding_dict: dict, out_path: Path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **embedding_dict)
    print(f"[cnn_evaluation] Saved embeddings + metadata -> {out_path} "
          f"({embedding_dict['embeddings'].shape[0]} samples, "
          f"{embedding_dict['embeddings'].shape[1]}-D embedding)")


def plot_embedding_pca(embedding_dict: dict, plots_dir: Path, max_points: int = 20000):
    """
    Optional bonus (brief section 16.7): a quick 2D PCA scatter of the
    CNN embeddings, colored by month, just to sanity-check the embedding
    space looks structured rather than random. NOT the full PCA/t-SNE/
    UMAP analysis promised for later - that comes with real data and
    dedicated region/season labels.
    """
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        print("[cnn_evaluation] scikit-learn not available - skipping the "
              "optional embedding PCA plot (not required for Stage 3).")
        return

    emb = embedding_dict["embeddings"]
    months = embedding_dict["month"]
    n = emb.shape[0]
    if n < 3:
        print("[cnn_evaluation] Too few embedding samples for a PCA plot - skipping.")
        return

    if n > max_points:
        rng = np.random.default_rng(42)
        sel = rng.choice(n, size=max_points, replace=False)
        emb, months = emb[sel], months[sel]

    pca = PCA(n_components=2)
    coords = pca.fit_transform(emb)

    plt.figure(figsize=(6, 5))
    sc = plt.scatter(coords[:, 0], coords[:, 1], c=months, cmap="twilight", s=4, alpha=0.5)
    plt.colorbar(sc, label="month")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    plt.title("CNN embedding — PCA (colored by month, quick sanity check)")
    plt.savefig(Path(plots_dir) / "07_embedding_pca.png", dpi=100, bbox_inches="tight")
    plt.close()
    print(f"[cnn_evaluation] Saved embedding PCA sanity plot -> {plots_dir}/07_embedding_pca.png")


def plot_cnn_training_curve(history: dict, plots_dir: Path):
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(history["train_loss"], label="train loss")
    plt.plot(history["val_loss"], label="val loss")
    plt.axvline(history["best_epoch"] - 1, color="gray", linestyle="--",
                label=f"best epoch ({history['best_epoch']})")
    plt.xlabel("epoch")
    plt.ylabel("MSE loss (normalized space)")
    plt.title("CNN training curve")
    plt.legend()
    plt.savefig(plots_dir / "05_cnn_training_curve.png", dpi=100, bbox_inches="tight")
    plt.close()


def _plot_rmse_bar(clim_metrics, mlp_metrics, cnn_metrics, plots_dir: Path):
    plt.figure(figsize=(5, 4))
    names = ["Climatology", "MLP", "CNN"]
    vals = [clim_metrics["overall"]["rmse"], mlp_metrics["overall"]["rmse"],
            cnn_metrics["overall"]["rmse"]]
    plt.bar(names, vals, color=["gray", "steelblue", "darkorange"])
    plt.ylabel("Overall RMSE (degC)")
    plt.title("Climatology vs MLP vs CNN — overall RMSE")
    for i, v in enumerate(vals):
        plt.text(i, v, f"{v:.3f}", ha="center", va="bottom")
    plt.savefig(plots_dir / "01_rmse_comparison_3way.png", dpi=100, bbox_inches="tight")
    plt.close()


def _plot_depthwise(clim_metrics, mlp_metrics, cnn_metrics, target_depths,
                     plots_dir: Path, metric_key: str, filename: str):
    clim_vals = [clim_metrics["per_depth"][str(float(d))][metric_key] for d in target_depths]
    mlp_vals = [mlp_metrics["per_depth"][str(float(d))][metric_key] for d in target_depths]
    cnn_vals = [cnn_metrics["per_depth"][str(float(d))][metric_key] for d in target_depths]

    plt.figure(figsize=(6, 6))
    plt.plot(clim_vals, target_depths, marker="o", label="Climatology", color="gray")
    plt.plot(mlp_vals, target_depths, marker="o", label="MLP", color="steelblue")
    plt.plot(cnn_vals, target_depths, marker="o", label="CNN", color="darkorange")
    plt.gca().invert_yaxis()
    plt.xlabel(f"{metric_key.upper()} (degC)")
    plt.ylabel("Depth (m)")
    plt.title(f"Depth-wise {metric_key.upper()}: Climatology vs MLP vs CNN")
    plt.legend()
    plt.savefig(plots_dir / filename, dpi=100, bbox_inches="tight")
    plt.close()


def _plot_example_profile(clim_pred_degc, mlp_pred_degc, cnn_pred_degc,
                           target_degc, target_depths, plots_dir: Path):
    i = len(target_degc) // 2
    plt.figure(figsize=(5, 6))
    plt.plot(target_degc[i], target_depths, marker="o", label="GLORYS target", color="black")
    plt.plot(clim_pred_degc[i], target_depths, marker="s", label="Climatology", color="gray")
    plt.plot(mlp_pred_degc[i], target_depths, marker="^", label="MLP", color="steelblue")
    plt.plot(cnn_pred_degc[i], target_depths, marker="d", label="CNN", color="darkorange")
    plt.gca().invert_yaxis()
    plt.xlabel("Temperature (degC)")
    plt.ylabel("Depth (m)")
    plt.title("Example 15-depth profile: prediction vs target")
    plt.legend()
    plt.savefig(plots_dir / "04_example_profile_3way.png", dpi=100, bbox_inches="tight")
    plt.close()


def _plot_scatter(cnn_pred_degc, target_degc, plots_dir: Path, max_points: int = 50000):
    # Subsample for rendering speed / file size - a real evaluation split
    # can have millions of (sample x depth) points, which is both slow
    # to render and produces a needlessly huge PNG. The underlying
    # metrics (RMSE/MAE/etc.) are computed on the FULL data elsewhere;
    # this cap only affects the scatter plot's visual sample.
    pred_flat = cnn_pred_degc.flatten()
    target_flat = target_degc.flatten()
    if len(pred_flat) > max_points:
        rng = np.random.default_rng(42)
        sel = rng.choice(len(pred_flat), size=max_points, replace=False)
        pred_flat, target_flat = pred_flat[sel], target_flat[sel]

    plt.figure(figsize=(5, 5))
    plt.scatter(target_flat, pred_flat, s=2, alpha=0.2, color="darkorange")
    lims = [min(target_flat.min(), pred_flat.min()),
            max(target_flat.max(), pred_flat.max())]
    plt.plot(lims, lims, "r--", linewidth=1)
    plt.xlabel("GLORYS target (degC)")
    plt.ylabel("CNN predicted (degC)")
    plt.title("CNN predicted vs actual temperature")
    plt.savefig(plots_dir / "06_scatter_cnn.png", dpi=100, bbox_inches="tight")
    plt.close()


def print_stage3_result(comparison_val: dict, comparison_test: dict):
    print("\n" + "=" * 40)
    print("STAGE 3 CNN RESULT")
    print("=" * 40)
    print("\nValidation:")
    print(f"Climatology RMSE: {comparison_val['climatology_rmse_degC']:.4f} degC")
    print(f"MLP RMSE:         {comparison_val['mlp_rmse_degC']:.4f} degC")
    print(f"CNN RMSE:         {comparison_val['cnn_rmse_degC']:.4f} degC")
    print()
    print(f"CNN beats MLP: {'YES' if comparison_val['cnn_beats_mlp'] else 'NO'}")
    print(f"CNN beats climatology: {'YES' if comparison_val['cnn_beats_climatology'] else 'NO'}")

    print("\nTest:")
    print(f"Climatology RMSE: {comparison_test['climatology_rmse_degC']:.4f} degC")
    print(f"MLP RMSE:         {comparison_test['mlp_rmse_degC']:.4f} degC")
    print(f"CNN RMSE:         {comparison_test['cnn_rmse_degC']:.4f} degC")
    print("=" * 40)
