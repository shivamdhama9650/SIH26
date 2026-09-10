"""
regional_analysis.py
=======================
Stage: STAGE 5, PART B — Arabian Sea vs Bay of Bengal case study.

Uses the SAME regional definition as Stage 4 (77E split,
src.argo_validation.regional_split) - never redefines it.

Two genuinely different things are reported here, kept explicitly
separate per the brief:
  1. GLORYS-based regional analysis: CNN predictions vs GLORYS
     (the training/reference target) on the TEST split, newly computed
     here (Day 3's evaluation was global-only, never split by region).
  2. ARGO regional validation: NOT recomputed here - this module reads
     Stage 4's already-produced results/argo/argo_regional_metrics.json
     as-is (reuse, not duplication) and reports it alongside, clearly
     labeled as independent-observation validation, separate from (1).
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.argo_validation import regional_split
from src.metrics import compute_metrics


def compute_glorys_regional_metrics(cnn_pred_degc: np.ndarray, target_degc: np.ndarray,
                                     lons: np.ndarray, target_depths: list) -> dict:
    """
    cnn_pred_degc, target_degc: (n_samples, n_depth) degC, TEST split only
        (final-report split, same rule as every other stage).
    lons: (n_samples,) the longitude of each sample (from idx/ds lookup).

    Returns {"arabian_sea": {...}, "bay_of_bengal": {...}, "other": {...}}
    each holding compute_metrics()'s usual overall/per_depth/n_samples,
    or {"n_samples": 0, "skipped": True} if a region has zero samples.
    """
    masks = regional_split(lons)
    # 'other' catches any sample outside both defined lon ranges (mostly
    # won't happen inside 45-105E, but a region split should never
    # silently drop points).
    covered = masks["arabian_sea"] | masks["bay_of_bengal"]
    masks["other"] = ~covered

    out = {}
    for region_name, mask in masks.items():
        n = int(mask.sum())
        if n == 0:
            out[region_name] = {"n_samples": 0, "skipped": True}
            continue
        m = compute_metrics(cnn_pred_degc[mask], target_degc[mask], target_depths)
        m["skipped"] = False
        out[region_name] = m
        print(f"[regional_analysis] GLORYS-based, '{region_name}' (n={n}): "
              f"RMSE={m['overall']['rmse']:.4f}, MAE={m['overall']['mae']:.4f}, "
              f"Bias={m['overall']['bias']:.4f}, Corr={m['overall']['correlation']:.4f}")
    return out


def load_argo_regional_metrics(path: Path):
    """Reads Stage 4's own output as-is. Returns None (with a clear
    printed message) if Stage 4 hasn't been run yet - never fabricates
    ARGO regional numbers."""
    path = Path(path)
    if not path.exists():
        print(f"[regional_analysis] {path} not found - Stage 4 "
              f"(run_argo_validation.py) hasn't been run, so no ARGO "
              f"regional numbers are available. GLORYS-based regional "
              f"analysis below is unaffected.")
        return None
    with open(path) as f:
        data = json.load(f)
    print(f"[regional_analysis] Loaded existing ARGO regional metrics from "
          f"{path} (produced by Stage 4, NOT recomputed here).")
    return data


def plot_region_rmse_bar(glorys_regional: dict, plots_dir: Path):
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    names = [n for n in ["arabian_sea", "bay_of_bengal"]
             if not glorys_regional.get(n, {}).get("skipped", True)]
    vals = [glorys_regional[n]["overall"]["rmse"] for n in names]
    plt.figure(figsize=(5, 4))
    plt.bar(names, vals, color=["#2b6cb0", "#dd6b20"])
    plt.ylabel("RMSE (degC)")
    plt.title("CNN vs GLORYS — overall RMSE by region (test split)")
    plt.savefig(plots_dir / "regional_rmse_bar.png", dpi=110, bbox_inches="tight")
    plt.close()


def plot_region_depthwise_rmse(glorys_regional: dict, target_depths: list, plots_dir: Path):
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 6))
    colors = {"arabian_sea": "#2b6cb0", "bay_of_bengal": "#dd6b20"}
    for region_name, color in colors.items():
        r = glorys_regional.get(region_name, {})
        if r.get("skipped", True):
            continue
        vals = [r["per_depth"][str(float(d))]["rmse"] for d in target_depths]
        plt.plot(vals, target_depths, marker="o", color=color, label=region_name)
    plt.gca().invert_yaxis()
    plt.xlabel("RMSE (degC)")
    plt.ylabel("Depth (m)")
    plt.title("CNN vs GLORYS — depth-wise RMSE by region (test split)")
    plt.legend()
    plt.savefig(plots_dir / "regional_depthwise_rmse.png", dpi=110, bbox_inches="tight")
    plt.close()


def plot_representative_profiles(cnn_pred_degc: np.ndarray, target_degc: np.ndarray,
                                  lons: np.ndarray, target_depths: list, plots_dir: Path,
                                  seed: int = 42):
    """One representative CNN-vs-GLORYS profile per region (nearest to
    that region's mean prediction, for a stable, non-cherry-picked
    pick), so a reader can see actual shapes, not just summary numbers."""
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    masks = regional_split(lons)
    fig, axes = plt.subplots(1, 2, figsize=(9, 6), sharey=True)
    for ax, (region_name, mask) in zip(axes, masks.items()):
        if mask.sum() == 0:
            ax.set_title(f"{region_name} (no samples)")
            continue
        region_pred = cnn_pred_degc[mask]
        mean_profile = region_pred.mean(axis=0)
        dists = np.linalg.norm(region_pred - mean_profile, axis=1)
        rep_idx = np.where(mask)[0][np.argmin(dists)]
        ax.plot(target_degc[rep_idx], target_depths, marker="o", color="black", label="GLORYS")
        ax.plot(cnn_pred_degc[rep_idx], target_depths, marker="d", color="darkorange", label="CNN")
        ax.invert_yaxis()
        ax.set_xlabel("Temp (degC)")
        ax.set_title(region_name)
        ax.legend()
    axes[0].set_ylabel("Depth (m)")
    plt.suptitle("Representative CNN vs GLORYS profile per region (test split)")
    plt.savefig(plots_dir / "regional_representative_profiles.png", dpi=110, bbox_inches="tight")
    plt.close()
