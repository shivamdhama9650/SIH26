"""
predictor.py
============
Loads the trained OceanXRay checkpoints (CNN and/or MLP) and runs
inference: raw physical-unit inputs -> normalized -> model -> 15-depth
temperature profile (denormalized back to degC).

IMPORTANT — READ BEFORE DEPLOYING:
The checkpoints only store MODEL WEIGHTS. They do NOT store the
train-split mean/std used to normalize inputs and targets. That file
is `normalization_stats.json`, produced by src/normalization.py during
the Day-1 pipeline run and copied to:
    /content/drive/MyDrive/oceanxray/data/processed/normalization_stats.json
It was NOT included in oceanxray_without_data.zip (that zip explicitly
excludes data/). You must copy that exact file from your Drive into
this deploy/ folder before real predictions will be meaningful.

If normalization_stats.json is missing, this module falls back to
IDENTITY stats (mean=0, std=1) purely so the code path can be smoke
tested end-to-end. Predictions made in that fallback mode are NOT
physically meaningful — they will be clearly flagged in the output.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.cnn_model import PatchCNN
from src.mlp_model import PointMLP

HERE = Path(__file__).resolve().parent
CKPT_DIR = HERE / "checkpoints"
STATS_PATH = HERE / "normalization_stats.json"


def _load_stats():
    if STATS_PATH.exists():
        with open(STATS_PATH) as f:
            stats = json.load(f)
        return stats, True
    # Fallback identity stats so the pipeline never hard-crashes.
    print(f"[predictor] WARNING: {STATS_PATH.name} not found next to predictor.py. "
          f"Using IDENTITY (mean=0, std=1) placeholder stats — predictions will "
          f"be in RAW normalized units, not real degC. Copy the real file from "
          f"Drive: oceanxray/data/processed/normalization_stats.json")
    return None, False


class OceanXRayPredictor:
    """
    Wraps both trained models. Use predict_cnn() for a 3x3 spatial patch
    (main/locked model) or predict_mlp() for a single point (MVP baseline).
    """

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self.stats, self.stats_available = _load_stats()

        cnn_ckpt = torch.load(CKPT_DIR / "cnn_best.pt", map_location=self.device, weights_only=False)
        self.cnn_channels = cnn_ckpt["channels"]
        self.cnn_depths = None  # filled from stats if available, else from training_config below
        self.cnn = PatchCNN(
            n_input_channels=cnn_ckpt["n_input_channels"],
            patch_size=cnn_ckpt["patch_size"],
            num_filters=cnn_ckpt["num_filters"],
            embedding_dim=cnn_ckpt["embedding_dim"],
            hidden_dim=cnn_ckpt["hidden_dim"],
            n_output_depths=cnn_ckpt["n_output_depths"],
            activation=cnn_ckpt["activation"],
        )
        self.cnn.load_state_dict(cnn_ckpt["model_state_dict"])
        self.cnn.eval()
        self.cnn_patch_size = cnn_ckpt["patch_size"]

        mlp_ckpt = torch.load(CKPT_DIR / "mlp_best.pt", map_location=self.device, weights_only=False)
        self.mlp_channels = mlp_ckpt["channels"]
        self.mlp = PointMLP(
            n_input_channels=mlp_ckpt["n_input_channels"],
            n_output_depths=mlp_ckpt["n_output_depths"],
            hidden_dim=mlp_ckpt["hidden_dim"],
        )
        self.mlp.load_state_dict(mlp_ckpt["model_state_dict"])
        self.mlp.eval()

        # Target depths (meters) — locked architecture, same for both models.
        self.depths = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]

    # ------------------------------------------------------------------
    def _normalize_channel(self, var_name: str, value: np.ndarray) -> np.ndarray:
        if self.stats is None:
            return value  # identity fallback
        s = self.stats["input_stats"][var_name]
        return (value - s["mean"]) / s["std"]

    def _denormalize_targets(self, arr: np.ndarray) -> np.ndarray:
        if self.stats is None:
            return arr  # identity fallback, still "normalized units"
        out = np.zeros_like(arr, dtype=float)
        for i, d in enumerate(self.depths):
            s = self.stats["target_stats_per_depth"][str(float(d))]
            out[i] = arr[i] * s["std"] + s["mean"]
        return out

    # ------------------------------------------------------------------
    def predict_mlp(self, values: dict) -> dict:
        """
        values: {channel_name: float} for each of self.mlp_channels,
        e.g. {"sst": 28.4, "ssh": 0.12, "u_current": 0.05,
              "v_current": -0.02, "v_wind": 3.1}
        Returns {"depths_m": [...], "temperature_degC": [...], "stats_used": bool}
        """
        missing = [c for c in self.mlp_channels if c not in values]
        if missing:
            raise ValueError(f"Missing required channels for MLP: {missing}")

        x_norm = [self._normalize_channel(c, values[c]) for c in self.mlp_channels]
        x = torch.tensor([x_norm], dtype=torch.float32)
        with torch.no_grad():
            out = self.mlp(x).numpy()[0]
        temps = self._denormalize_targets(out)
        return {
            "model": "mlp",
            "depths_m": self.depths,
            "temperature_degC": [round(float(t), 3) for t in temps],
            "stats_used": self.stats_available,
        }

    def predict_cnn(self, patch: dict) -> dict:
        """
        patch: {channel_name: 3x3 nested list/array} for each of
        self.cnn_channels, e.g.
            {"sst": [[..3..],[..3..],[..3..]], "ssh": [[...]], ...}
        Returns {"depths_m": [...], "temperature_degC": [...], "stats_used": bool}
        """
        missing = [c for c in self.cnn_channels if c not in patch]
        if missing:
            raise ValueError(f"Missing required channels for CNN: {missing}")

        p = self.cnn_patch_size
        chans = []
        for c in self.cnn_channels:
            arr = np.asarray(patch[c], dtype=float)
            if arr.shape != (p, p):
                raise ValueError(f"Channel '{c}' must be a {p}x{p} patch, got shape {arr.shape}")
            chans.append(self._normalize_channel(c, arr))
        x = torch.tensor(np.stack(chans)[None, ...], dtype=torch.float32)  # (1, C, p, p)
        with torch.no_grad():
            out = self.cnn(x).numpy()[0]
        temps = self._denormalize_targets(out)
        return {
            "model": "cnn",
            "depths_m": self.depths,
            "temperature_degC": [round(float(t), 3) for t in temps],
            "stats_used": self.stats_available,
        }
