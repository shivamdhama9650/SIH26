"""
evaluation.py
==============
Stage: DAY 2 — ties everything together: runs both models on the SAME
evaluation split, computes metrics in normalized AND degC space, saves
JSON results, makes comparison plots, and prints the final DAY 2 GATE.
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.metrics import compute_metrics, inverse_transform_target


def predict_mlp(model, X: np.ndarray, device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        xb = torch.from_numpy(X).to(device)
        pred = model(xb).cpu().numpy()
    return pred


def evaluate_all(climatology, mlp_model, device,
                  X_eval, Y_eval_norm, sample_times, idx,
                  target_depths, target_stats_per_depth, out_dir: Path,
                  split_label: str = "test", make_plots: bool = True):
    """
    Runs climatology + MLP on the SAME evaluation samples, computes
    metrics in both normalized and degC space, saves everything, and
    (optionally) makes plots. Returns (clim_metrics_degc, mlp_metrics_degc,
    comparison).

    split_label: "val" or "test" — used as a filename prefix so val-based
    and test-based results never overwrite each other. Per the locked
    architecture's hard-gate rule, the "MLP beats climatology" GATE
    DECISION must be computed on VALIDATION data (validation is for
    model/approach selection; test is reserved for a final, untouched,
    one-time check) — see run_day2.py for how the two calls are used.
    """
    out_dir = Path(out_dir)
    results_dir = out_dir / "results"
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # ---- predictions (normalized space) ----
    clim_pred_norm = climatology.predict(sample_times, idx["lat_idx"], idx["lon_idx"])
    mlp_pred_norm = predict_mlp(mlp_model, X_eval, device)

    # ---- metrics in normalized space (debugging) ----
    clim_metrics_norm = compute_metrics(clim_pred_norm, Y_eval_norm, target_depths)
    mlp_metrics_norm = compute_metrics(mlp_pred_norm, Y_eval_norm, target_depths)

    # ---- inverse-transform to degC (the real scientific numbers) ----
    clim_pred_degc = inverse_transform_target(clim_pred_norm, target_depths, target_stats_per_depth)
    mlp_pred_degc = inverse_transform_target(mlp_pred_norm, target_depths, target_stats_per_depth)
    target_degc = inverse_transform_target(Y_eval_norm, target_depths, target_stats_per_depth)

    clim_metrics_degc = compute_metrics(clim_pred_degc, target_degc, target_depths)
    mlp_metrics_degc = compute_metrics(mlp_pred_degc, target_degc, target_depths)

    # ---- save JSON (prefixed by split so val/test never collide) ----
    with open(results_dir / f"climatology_metrics_{split_label}.json", "w") as f:
        json.dump({"normalized_space": clim_metrics_norm, "degC_space": clim_metrics_degc}, f, indent=2)
    with open(results_dir / f"mlp_metrics_{split_label}.json", "w") as f:
        json.dump({"normalized_space": mlp_metrics_norm, "degC_space": mlp_metrics_degc}, f, indent=2)

    mlp_beats_climatology = mlp_metrics_degc["overall"]["rmse"] < clim_metrics_degc["overall"]["rmse"]
    comparison = {
        "split_used": split_label,
        "climatology_rmse_degC": clim_metrics_degc["overall"]["rmse"],
        "mlp_rmse_degC": mlp_metrics_degc["overall"]["rmse"],
        "climatology_mae_degC": clim_metrics_degc["overall"]["mae"],
        "mlp_mae_degC": mlp_metrics_degc["overall"]["mae"],
        "mlp_beats_climatology": bool(mlp_beats_climatology),
        "n_eval_samples": int(X_eval.shape[0]),
    }
    with open(results_dir / f"comparison_{split_label}.json", "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"[evaluation] Saved '{split_label}' metrics + comparison -> {results_dir}")

    # ---- plots (only made once, from the split the caller designates —
    # run_day2.py makes plots from the val-based gate call to avoid
    # writing the same 5 filenames twice) ----
    if make_plots:
        _plot_rmse_comparison(clim_metrics_degc, mlp_metrics_degc, plots_dir)
        _plot_depthwise_rmse(clim_metrics_degc, mlp_metrics_degc, target_depths, plots_dir)
        _plot_example_profile(clim_pred_degc, mlp_pred_degc, target_degc, target_depths, plots_dir)
        _plot_scatter(mlp_pred_degc, target_degc, plots_dir)

    return clim_metrics_degc, mlp_metrics_degc, comparison


def plot_training_curve(history: dict, out_dir: Path):
    out_dir = Path(out_dir) / "results" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(history["train_loss"], label="train loss")
    plt.plot(history["val_loss"], label="val loss")
    plt.axvline(history["best_epoch"] - 1, color="gray", linestyle="--",
                label=f"best epoch ({history['best_epoch']})")
    plt.xlabel("epoch")
    plt.ylabel("MSE loss (normalized space)")
    plt.title("MLP training curve")
    plt.legend()
    plt.savefig(out_dir / "04_training_curve.png", dpi=100, bbox_inches="tight")
    plt.close()


def _plot_rmse_comparison(clim_metrics, mlp_metrics, plots_dir: Path):
    plt.figure(figsize=(4, 4))
    names = ["Climatology", "MLP"]
    vals = [clim_metrics["overall"]["rmse"], mlp_metrics["overall"]["rmse"]]
    plt.bar(names, vals, color=["gray", "steelblue"])
    plt.ylabel("Overall RMSE (degC)")
    plt.title("Climatology vs MLP — overall RMSE")
    for i, v in enumerate(vals):
        plt.text(i, v, f"{v:.3f}", ha="center", va="bottom")
    plt.savefig(plots_dir / "01_rmse_comparison.png", dpi=100, bbox_inches="tight")
    plt.close()


def _plot_depthwise_rmse(clim_metrics, mlp_metrics, target_depths, plots_dir: Path):
    clim_rmse = [clim_metrics["per_depth"][str(float(d))]["rmse"] for d in target_depths]
    mlp_rmse = [mlp_metrics["per_depth"][str(float(d))]["rmse"] for d in target_depths]

    plt.figure(figsize=(6, 6))
    plt.plot(clim_rmse, target_depths, marker="o", label="Climatology", color="gray")
    plt.plot(mlp_rmse, target_depths, marker="o", label="MLP", color="steelblue")
    plt.gca().invert_yaxis()
    plt.xlabel("RMSE (degC)")
    plt.ylabel("Depth (m)")
    plt.title("Depth-wise RMSE: Climatology vs MLP")
    plt.legend()
    plt.savefig(plots_dir / "02_depthwise_rmse.png", dpi=100, bbox_inches="tight")
    plt.close()


def _plot_example_profile(clim_pred_degc, mlp_pred_degc, target_degc, target_depths, plots_dir: Path):
    i = len(target_degc) // 2  # just pick a sample from the middle
    plt.figure(figsize=(5, 6))
    plt.plot(target_degc[i], target_depths, marker="o", label="GLORYS target", color="black")
    plt.plot(clim_pred_degc[i], target_depths, marker="s", label="Climatology", color="gray")
    plt.plot(mlp_pred_degc[i], target_depths, marker="^", label="MLP", color="steelblue")
    plt.gca().invert_yaxis()
    plt.xlabel("Temperature (degC)")
    plt.ylabel("Depth (m)")
    plt.title("Example 15-depth profile: prediction vs target")
    plt.legend()
    plt.savefig(plots_dir / "03_example_profile.png", dpi=100, bbox_inches="tight")
    plt.close()


def _plot_scatter(mlp_pred_degc, target_degc, plots_dir: Path, max_points: int = 50000):
    # Subsample for rendering speed / file size - see cnn_evaluation.py's
    # identical rationale. Metrics elsewhere are computed on full data.
    pred_flat = mlp_pred_degc.flatten()
    target_flat = target_degc.flatten()
    if len(pred_flat) > max_points:
        rng = np.random.default_rng(42)
        sel = rng.choice(len(pred_flat), size=max_points, replace=False)
        pred_flat, target_flat = pred_flat[sel], target_flat[sel]

    plt.figure(figsize=(5, 5))
    plt.scatter(target_flat, pred_flat, s=2, alpha=0.2)
    lims = [min(target_flat.min(), pred_flat.min()),
            max(target_flat.max(), pred_flat.max())]
    plt.plot(lims, lims, "r--", linewidth=1)
    plt.xlabel("GLORYS target (degC)")
    plt.ylabel("MLP predicted (degC)")
    plt.title("MLP predicted vs actual temperature")
    plt.savefig(plots_dir / "05_scatter_mlp.png", dpi=100, bbox_inches="tight")
    plt.close()


def print_gate(comparison: dict):
    print("\n" + "=" * 40)
    print("DAY 2 GATE")
    print("=" * 40)
    print(f"(decision based on the {comparison.get('split_used', 'val')} split, "
          f"per the locked architecture's hard-gate rule)")
    print(f"Climatology RMSE: {comparison['climatology_rmse_degC']:.4f} degC")
    print(f"MLP RMSE:         {comparison['mlp_rmse_degC']:.4f} degC")
    print()
    if comparison["mlp_beats_climatology"]:
        print("MLP beats climatology: YES")
        print()
        print("DAY 2 PASSED — CNN can proceed.")
    else:
        print("MLP beats climatology: NO")
        print()
        print("DAY 2 FAILED — DO NOT proceed to CNN.")
        print("Investigate data/model/training before adding complexity.")
    print("=" * 40)
