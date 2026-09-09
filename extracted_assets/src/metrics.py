"""
metrics.py
==========
Stage: DAY 2 — evaluation metrics for climatology and MLP.

Computes RMSE, MAE, Bias, Correlation:
  - overall (all depths pooled together)
  - depth-wise (each of the 15 depths separately)

Also provides inverse_transform_target() to convert normalized
predictions/targets back to physical degC using the Day 1 saved
per-depth normalization stats, so we never accidentally report
"metrics" that are actually in unitless normalized space when we
meant to report the real scientific numbers.
"""

import numpy as np


def inverse_transform_target(normalized: np.ndarray, target_depths: list,
                              target_stats_per_depth: dict) -> np.ndarray:
    """
    normalized: (n_samples, n_depth) array in normalized space.
    target_depths: list of the 15 depth values, IN THE SAME COLUMN ORDER
        as `normalized`'s second axis (this must match how Y was built
        in mlp_dataset.py, which iterates target_arr's depth axis in
        the order it appears in ml_ready_dataset.nc - i.e. config.TARGET_DEPTHS order).
    target_stats_per_depth: the dict loaded from normalization_stats.json
        ("target_stats_per_depth"), keyed by str(float(depth)).

    Returns: (n_samples, n_depth) array in degC.
    """
    out = np.empty_like(normalized)
    for d_idx, depth_val in enumerate(target_depths):
        key = str(float(depth_val))
        if key not in target_stats_per_depth:
            raise KeyError(f"[metrics] No normalization stats found for depth "
                            f"{depth_val} (key '{key}'). Available keys: "
                            f"{list(target_stats_per_depth.keys())}")
        s = target_stats_per_depth[key]
        out[:, d_idx] = normalized[:, d_idx] * s["std"] + s["mean"]
    return out


def _safe_corr(pred_flat: np.ndarray, target_flat: np.ndarray) -> float:
    if len(pred_flat) < 2 or np.std(pred_flat) == 0 or np.std(target_flat) == 0:
        return float("nan")
    return float(np.corrcoef(pred_flat, target_flat)[0, 1])


def compute_metrics(pred: np.ndarray, target: np.ndarray, target_depths: list) -> dict:
    """
    pred, target: (n_samples, n_depth) arrays, SAME space (both normalized
    or both degC — caller decides which by passing the right arrays).

    Returns:
        {
          "overall": {"rmse":..., "mae":..., "bias":..., "correlation":...},
          "per_depth": {depth_value_str: {"rmse":...,...}, ...},
          "n_samples": int,
        }
    """
    assert pred.shape == target.shape, \
        f"[metrics] shape mismatch: pred {pred.shape} vs target {target.shape}"

    error = pred - target

    overall = {
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
        "correlation": _safe_corr(pred.flatten(), target.flatten()),
    }

    per_depth = {}
    for d_idx, depth_val in enumerate(target_depths):
        e = error[:, d_idx]
        per_depth[str(float(depth_val))] = {
            "rmse": float(np.sqrt(np.mean(e ** 2))),
            "mae": float(np.mean(np.abs(e))),
            "bias": float(np.mean(e)),
            "correlation": _safe_corr(pred[:, d_idx], target[:, d_idx]),
        }

    return {"overall": overall, "per_depth": per_depth, "n_samples": int(pred.shape[0])}
