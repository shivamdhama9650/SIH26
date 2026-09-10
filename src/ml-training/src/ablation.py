"""
ablation.py
=============
Stage: STAGE 5, PART C/D — spatial-context ablation.

Model A ("Center MLP", 1x1 context): the existing src.mlp_model.PointMLP
    architecture (reused, not reinvented), trained FRESH on the center
    pixel of the CNN's own valid 3x3 samples (see below for why "fresh"
    matters).
Model B ("CNN", 3x3 context): the ALREADY-TRAINED models/cnn_best.pt
    from Day 3 — reused as-is, NOT retrained here.

WHY MODEL A IS TRAINED FRESH (not just reusing Day 2's mlp_best.pt):
Day 2's point-MLP was trained on ITS OWN, looser valid-sample set
(src.mlp_dataset.find_valid_point_samples only requires the center
pixel itself to be valid). The CNN requires a stricter condition (the
full 3x3 neighborhood must be valid). Comparing Day 2's MLP against
the CNN would then be confounded by two different sample sets, not
just by "spatial context vs. no spatial context" - exactly the kind of
unfair comparison the brief calls out. So this module builds a NEW
Center-MLP, trained on the CNN's own (stricter) valid-sample set, with
identical channels/target/normalization/splits/evaluation metrics and
the same validation-based checkpoint-selection rule as the CNN. This
isolates exactly one variable: 1x1 input vs 3x3 input, same everything
else.

Both models are evaluated on the CNN's exact valid-sample set (same
samples Day 3 always used for the "fair 3-way" comparison too), so
this is consistent with the project's established fairness convention.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.metrics import compute_metrics, inverse_transform_target


def extract_center_pixel(X_cnn: np.ndarray) -> np.ndarray:
    """X_cnn: (n_samples, n_channels, patch_size, patch_size).
    Returns (n_samples, n_channels) - the exact center pixel, i.e.
    identical to what a 1x1/point-only model would have seen at that
    same (time, lat, lon). cnn_dataset.py's own alignment assertion
    already guarantees this equals the raw point value."""
    patch_size = X_cnn.shape[-1]
    half = patch_size // 2
    return X_cnn[:, :, half, half].copy()


def predict_cnn_degc(cnn_model, X_cnn: np.ndarray, device, target_depths, target_stats_per_depth,
                      batch_size: int = 2048) -> np.ndarray:
    cnn_model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_cnn), batch_size):
            xb = torch.from_numpy(X_cnn[i:i + batch_size]).to(device)
            preds.append(cnn_model(xb).cpu().numpy())
    pred_norm = np.concatenate(preds, axis=0) if preds else np.empty((0, len(target_depths)))
    return inverse_transform_target(pred_norm, target_depths, target_stats_per_depth)


def predict_center_mlp_degc(mlp_model, X_center: np.ndarray, device, target_depths,
                             target_stats_per_depth, batch_size: int = 2048) -> np.ndarray:
    mlp_model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_center), batch_size):
            xb = torch.from_numpy(X_center[i:i + batch_size]).to(device)
            preds.append(mlp_model(xb).cpu().numpy())
    pred_norm = np.concatenate(preds, axis=0) if preds else np.empty((0, len(target_depths)))
    return inverse_transform_target(pred_norm, target_depths, target_stats_per_depth)


def build_ablation_table(center_mlp_val, center_mlp_test, cnn_val, cnn_test,
                          center_mlp_n_params: int, cnn_n_params: int) -> pd.DataFrame:
    """Returns the exact table shape requested in the brief (Part D),
    plus n_params for transparency about "reasonably comparable" size,
    and both val (model-selection evidence) and test (final report)
    RMSE so nothing here implies test was used to pick a model."""
    rows = [
        {"Model": "Center MLP", "Context": "1x1", "N_Params": center_mlp_n_params,
         "Val_RMSE": center_mlp_val["overall"]["rmse"], "Test_RMSE": center_mlp_test["overall"]["rmse"],
         "Test_MAE": center_mlp_test["overall"]["mae"], "Test_Bias": center_mlp_test["overall"]["bias"],
         "Test_Correlation": center_mlp_test["overall"]["correlation"]},
        {"Model": "CNN", "Context": "3x3", "N_Params": cnn_n_params,
         "Val_RMSE": cnn_val["overall"]["rmse"], "Test_RMSE": cnn_test["overall"]["rmse"],
         "Test_MAE": cnn_test["overall"]["mae"], "Test_Bias": cnn_test["overall"]["bias"],
         "Test_Correlation": cnn_test["overall"]["correlation"]},
    ]
    return pd.DataFrame(rows)


def build_depthwise_table(center_mlp_test: dict, cnn_test: dict, target_depths: list) -> pd.DataFrame:
    rows = []
    for d in target_depths:
        key = str(float(d))
        rows.append({
            "depth_m": d,
            "center_mlp_rmse": center_mlp_test["per_depth"][key]["rmse"],
            "cnn_rmse": cnn_test["per_depth"][key]["rmse"],
            "rmse_improvement": center_mlp_test["per_depth"][key]["rmse"] - cnn_test["per_depth"][key]["rmse"],
            "center_mlp_mae": center_mlp_test["per_depth"][key]["mae"],
            "cnn_mae": cnn_test["per_depth"][key]["mae"],
        })
    return pd.DataFrame(rows)


def plot_ablation_comparison(table: pd.DataFrame, plots_dir: Path):
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 4))
    plt.bar(table["Model"], table["Test_RMSE"], color=["#718096", "#dd6b20"])
    plt.ylabel("Test RMSE (degC)")
    plt.title("Ablation: spatial context (3x3 CNN) vs center-only (1x1 MLP)")
    plt.savefig(plots_dir / "ablation_rmse_bar.png", dpi=110, bbox_inches="tight")
    plt.close()


def plot_depthwise_comparison(depthwise_table: pd.DataFrame, plots_dir: Path):
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 6))
    plt.plot(depthwise_table["center_mlp_rmse"], depthwise_table["depth_m"],
              marker="o", color="#718096", label="Center MLP (1x1)")
    plt.plot(depthwise_table["cnn_rmse"], depthwise_table["depth_m"],
              marker="d", color="#dd6b20", label="CNN (3x3)")
    plt.gca().invert_yaxis()
    plt.xlabel("RMSE (degC)")
    plt.ylabel("Depth (m)")
    plt.title("Ablation — depth-wise RMSE (test split)")
    plt.legend()
    plt.savefig(plots_dir / "ablation_depthwise_rmse.png", dpi=110, bbox_inches="tight")
    plt.close()
