"""
argo_validation.py
=====================
Stage: STAGE 4 — ARGO independent validation.

Loads the FINAL LOCKED CNN checkpoint (never retrains, never tunes),
runs it on the matched Argo patches, compares against the Argo
depth-interpolated temperatures (NOT against GLORYS), and produces the
depth-wise / regional / overall metrics and report files.

Metrics here are NaN-aware (a real Argo profile usually does NOT reach
all 15 depths), unlike src.metrics.compute_metrics which assumes dense
arrays - that function is reused where it fits (Day 1-3 GLORYS
evaluation) but cannot be reused as-is here, so a masked variant is
defined below instead.
"""

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import argo_config
from src.cnn_model import PatchCNN
from src.metrics import inverse_transform_target


# ---------------------------------------------------------------------------
# STEP 8: NO DATA LEAKAGE — static guards
# ---------------------------------------------------------------------------
TRAINING_ENTRY_SCRIPTS = ["run_pipeline.py", "run_day2.py", "run_day3.py"]
TRAINING_SRC_MODULES = [
    "loaders.py", "regrid.py", "glorys_vertical.py", "masking.py",
    "splitting.py", "normalization.py", "assemble.py", "climatology.py",
    "mlp_dataset.py", "mlp_model.py", "train_mlp.py", "cnn_dataset.py",
    "cnn_model.py", "train_cnn.py",
]


def verify_no_argo_leakage(project_root: Path):
    """
    Static source-level guard: scans every Day 1-3 training entry point
    and src module for the substring 'argo' (case-insensitive). If any
    of them mention ARGO in any way, that is a strong signal ARGO data
    could be reaching the training/normalization/tuning path, so we
    fail loudly rather than silently trust the file layout.

    This is deliberately conservative (a false positive just means a
    comment mentions ARGO) - the point is that Stage 4 must NEVER be
    the thing that lets ARGO quietly leak backward into Day 1-3 code.
    """
    project_root = Path(project_root)
    offending = []
    for fname in TRAINING_ENTRY_SCRIPTS:
        fpath = project_root / fname
        if fpath.exists() and "argo" in fpath.read_text().lower():
            offending.append(str(fpath))
    for fname in TRAINING_SRC_MODULES:
        fpath = project_root / "src" / fname
        if fpath.exists() and "argo" in fpath.read_text().lower():
            offending.append(str(fpath))

    if offending:
        raise RuntimeError(
            f"[argo_validation] LEAKAGE GUARD TRIPPED: the word 'argo' "
            f"appears in training-pipeline file(s) {offending}. Stage 4 "
            f"(ARGO validation) must never be referenced from Day 1-3 "
            f"training/normalization/tuning code. Refusing to proceed "
            f"until this is resolved."
        )
    print(f"[argo_validation] Leakage guard OK: no Day 1-3 training file "
          f"references ARGO (checked {len(TRAINING_ENTRY_SCRIPTS)} entry "
          f"scripts + {len(TRAINING_SRC_MODULES)} src modules).")

    # Also assert this module itself never IMPORTS the training-only stat
    # functions that would recompute normalization from data. Checked as
    # an import statement (not a bare substring search) so this guard
    # doesn't trip on its own list of forbidden names below.
    forbidden_calls = ["compute_input_stats", "compute_target_stats_per_depth"]
    this_module_src = Path(__file__).read_text()
    used = [f for f in forbidden_calls
            if f"import {f}" in this_module_src or f", {f}" in this_module_src
            or f"normalization.{f}" in this_module_src]
    assert not used, (
        f"[argo_validation] {used} must never be imported into Stage 4 - "
        f"normalization stats are LOADED from the existing "
        f"normalization_stats.json, never recomputed here."
    )


# ---------------------------------------------------------------------------
# CNN checkpoint loading + inference (reuse the checkpoint, never retrain)
# ---------------------------------------------------------------------------
def load_final_cnn_checkpoint(checkpoint_path: Path, device, expected_channels: list):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"[argo_validation] No final CNN checkpoint at {checkpoint_path}. "
            f"Run run_day3.py first - Stage 4 loads the checkpoint Day 3 "
            f"already trained and NEVER trains its own."
        )
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if state["channels"] != expected_channels:
        raise ValueError(
            f"[argo_validation] CNN checkpoint was trained with channels "
            f"{state['channels']} but the current Day 1 metadata reports "
            f"{expected_channels}. Refusing to run ARGO validation against "
            f"a mismatched checkpoint - re-run Day 1-3 or point at the "
            f"correct checkpoint."
        )
    model = PatchCNN(
        n_input_channels=state["n_input_channels"], patch_size=state["patch_size"],
        num_filters=state["num_filters"], embedding_dim=state["embedding_dim"],
        hidden_dim=state["hidden_dim"], n_output_depths=state["n_output_depths"],
        activation=state["activation"],
    ).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    print(f"[argo_validation] Loaded FINAL LOCKED CNN checkpoint from "
          f"{checkpoint_path} (trained to epoch {state['epoch']}, "
          f"val_loss={state['val_loss']:.5f}). This checkpoint is used "
          f"as-is - no gradient updates happen in Stage 4.")
    return model


def predict_cnn(model, X: np.ndarray, device, batch_size: int = 2048) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i:i + batch_size]).to(device)
            preds.append(model(xb).cpu().numpy())
    return np.concatenate(preds, axis=0) if preds else np.empty((0,))


# ---------------------------------------------------------------------------
# NaN-aware metrics (Argo profiles rarely reach all 15 depths)
# ---------------------------------------------------------------------------
def _safe_corr(pred: np.ndarray, target: np.ndarray):
    valid = np.isfinite(pred) & np.isfinite(target)
    if valid.sum() < 2 or np.std(pred[valid]) == 0 or np.std(target[valid]) == 0:
        return float("nan")
    return float(np.corrcoef(pred[valid], target[valid])[0, 1])


def compute_masked_metrics(pred: np.ndarray, target: np.ndarray, target_depths: list) -> dict:
    """
    pred, target: (n_profiles, n_depth) degC. target may contain NaN
    (Argo did not observe that depth for that profile) - those cells are
    EXCLUDED from every statistic, never treated as zero/bias-free.

    Returns the same shape of dict as src.metrics.compute_metrics, plus
    an explicit "n_valid_obs" count per depth/overall (since it varies).
    """
    assert pred.shape == target.shape
    valid = np.isfinite(pred) & np.isfinite(target)
    error = np.where(valid, pred - target, np.nan)

    overall_valid = valid
    overall = {
        "rmse": float(np.sqrt(np.nanmean(error[overall_valid] ** 2))) if overall_valid.any() else float("nan"),
        "mae": float(np.nanmean(np.abs(error[overall_valid]))) if overall_valid.any() else float("nan"),
        "bias": float(np.nanmean(error[overall_valid])) if overall_valid.any() else float("nan"),
        "correlation": _safe_corr(pred.flatten(), target.flatten()),
        "n_valid_obs": int(overall_valid.sum()),
    }

    per_depth = {}
    for d_idx, depth_val in enumerate(target_depths):
        v = valid[:, d_idx]
        n = int(v.sum())
        if n == 0:
            per_depth[str(float(depth_val))] = {
                "rmse": float("nan"), "mae": float("nan"), "bias": float("nan"),
                "correlation": float("nan"), "n_valid_obs": 0,
            }
            continue
        e = error[v, d_idx]
        per_depth[str(float(depth_val))] = {
            "rmse": float(np.sqrt(np.mean(e ** 2))),
            "mae": float(np.mean(np.abs(e))),
            "bias": float(np.mean(e)),
            "correlation": _safe_corr(pred[v, d_idx], target[v, d_idx]),
            "n_valid_obs": n,
        }

    return {"overall": overall, "per_depth": per_depth,
            "n_profiles": int(pred.shape[0])}


# ---------------------------------------------------------------------------
# Regional case study (Step 7)
# ---------------------------------------------------------------------------
def regional_split(lons: np.ndarray) -> dict:
    """Returns {'arabian_sea': bool_mask, 'bay_of_bengal': bool_mask}
    using the explicit rectangular lon ranges in argo_config.py."""
    a_lo, a_hi = argo_config.ARABIAN_SEA_LON_RANGE
    b_lo, b_hi = argo_config.BAY_OF_BENGAL_LON_RANGE
    return {
        "arabian_sea": (lons >= a_lo) & (lons < a_hi),
        "bay_of_bengal": (lons >= b_lo) & (lons <= b_hi),
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_depthwise_metric(metrics: dict, target_depths: list, plots_dir: Path,
                           metric_key: str, filename: str, title: str):
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    vals = [metrics["per_depth"][str(float(d))][metric_key] for d in target_depths]
    plt.figure(figsize=(5, 6))
    plt.plot(vals, target_depths, marker="o", color="darkorange")
    plt.gca().invert_yaxis()
    plt.xlabel(f"{metric_key.upper()} (degC)")
    plt.ylabel("Depth (m)")
    plt.title(title)
    plt.savefig(plots_dir / filename, dpi=100, bbox_inches="tight")
    plt.close()


def plot_matched_locations(matches: pd.DataFrame, plots_dir: Path):
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 6))
    unmatched = matches[~matches["matched"]]
    matched = matches[matches["matched"]]
    plt.scatter(unmatched["argo_lon"], unmatched["argo_lat"], s=15, c="lightgray",
                label=f"unmatched ({len(unmatched)})", marker="x")
    plt.scatter(matched["argo_lon"], matched["argo_lat"], s=20, c="darkorange",
                label=f"matched ({len(matched)})")
    plt.xlabel("Longitude (deg E)")
    plt.ylabel("Latitude (deg N)")
    plt.title("Matched ARGO profile locations")
    plt.legend()
    plt.gca().set_aspect("equal", adjustable="box")
    plt.savefig(plots_dir / "argo_matched_locations.png", dpi=100, bbox_inches="tight")
    plt.close()


def plot_example_profiles(cnn_pred_degc: np.ndarray, argo_temp_degc: np.ndarray,
                           target_depths: list, plots_dir: Path, n_examples: int = 3):
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    n = min(n_examples, cnn_pred_degc.shape[0])
    if n == 0:
        return
    idxs = np.linspace(0, cnn_pred_degc.shape[0] - 1, n).astype(int)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 6), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, i in zip(axes, idxs):
        obs_valid = np.isfinite(argo_temp_degc[i])
        ax.plot(argo_temp_degc[i][obs_valid], np.array(target_depths)[obs_valid],
                marker="o", color="black", label="ARGO observed")
        ax.plot(cnn_pred_degc[i], target_depths, marker="d", color="darkorange",
                label="CNN predicted")
        ax.invert_yaxis()
        ax.set_xlabel("Temp (degC)")
        ax.set_title(f"profile {i}")
    axes[0].set_ylabel("Depth (m)")
    axes[0].legend()
    plt.suptitle("Example ARGO vs CNN profiles")
    plt.savefig(plots_dir / "argo_example_profiles.png", dpi=100, bbox_inches="tight")
    plt.close()
