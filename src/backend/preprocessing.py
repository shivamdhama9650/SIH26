import os
import sys
from pathlib import Path
_backend_dir = str(Path(__file__).resolve().parent)
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

import json
import logging
from typing import Tuple, Dict, Any, List
from datetime import date as dt_date
import numpy as np
import torch
from config import settings

logger = logging.getLogger("oceanxray.preprocessing")

class OceanPreprocessor:
    """
    Handles preprocessing and normalization for the OceanXRay 3x3 surface patch.
    Maintains strict scientific channel ordering:
    Channel 0: Sea Surface Temperature (SST, °C)
    Channel 1: Sea Surface Height / Sea Level Anomaly (SSH, m)
    Channel 2: Zonal Ocean Current (u_current, m/s)
    Channel 3: Meridional Ocean Current (v_current, m/s)
    Channel 4: Meridional Wind Velocity (v_wind, m/s)
    """

    CHANNELS = ["sst", "ssh", "u_current", "v_current", "v_wind"]

    # Load normalization statistics for the 5-channel input and target depths
    try:
        stats_path = settings.resolve_path(settings.NORMALIZATION_STATS_PATH)
        with open(stats_path, "r", encoding="utf-8") as f:
            _data = json.load(f)
            INPUT_STATS = _data.get("input_stats", {})
            TARGET_STATS = _data.get("target_stats_per_depth", {})
    except Exception as e:
        logger.error(f"Failed to load normalization stats from {settings.NORMALIZATION_STATS_PATH}: {e}")
        INPUT_STATS = {
            "sst": {"mean": 28.33, "std": 1.73},
            "ssh": {"mean": 0.11, "std": 0.097},
            "u_current": {"mean": 0.021, "std": 0.239},
            "v_current": {"mean": 0.0015, "std": 0.22},
            "v_wind": {"mean": 0.744, "std": 4.07}
        }
        TARGET_STATS = {}

    @classmethod
    def estimate_surface_parameters(cls, lat: float, lon: float, date_str: str) -> Dict[str, float]:
        """
        Calculates climatologically coherent surface parameters for a given lat, lon, and date
        over the North Indian Ocean basin.
        """
        try:
            d = dt_date.fromisoformat(date_str)
            day_of_year = d.timetuple().tm_yday
        except Exception:
            day_of_year = 15
        
        # Seasonal solar cycle
        season_phase = 2 * np.pi * (day_of_year - 80) / 365.25
        seasonal_anomaly = 1.2 * np.sin(season_phase)
        
        # Latitudinal & longitudinal gradients
        lat_effect = -0.15 * (lat - 10.0)
        lon_effect = 0.02 * (lon - 70.0)
        
        sst = float(np.clip(28.0 + lat_effect + lon_effect + seasonal_anomaly, 21.0, 32.5))
        ssh = float(0.04 * np.sin(season_phase + lat * 0.1) + 0.03 * np.cos(lon * 0.08))
        
        # Currents & winds characteristic of North Indian Ocean monsoon circulation
        u_current = float(0.15 * np.cos(season_phase) + 0.02 * np.sin(lat * 0.2))
        v_current = float(0.08 * np.sin(season_phase) - 0.01 * np.cos(lon * 0.1))
        v_wind = float(1.5 * np.sin(season_phase) + 0.5 * np.cos(lat * 0.15))
        
        # SSS proxy for metadata reference
        if lon > 80.0:
            sss = float(np.clip(32.8 - 0.03 * (lon - 80.0), 30.0, 37.0))
        else:
            sss = float(np.clip(35.8 - 0.02 * (75.0 - lon), 30.0, 37.0))

        return {
            "sst": round(sst, 2),
            "sss": round(sss, 2),
            "ssh": round(ssh, 3),
            "u_current": round(u_current, 3),
            "v_current": round(v_current, 3),
            "v_wind": round(v_wind, 2),
            "season_phase": round(season_phase, 3)
        }

    @classmethod
    def build_surface_patch(cls, lat: float, lon: float, date_str: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Builds the 3x3 surface patch across 5 channels: (5, 3, 3)
        """
        params = cls.estimate_surface_parameters(lat, lon, date_str)
        patch = np.zeros((settings.INPUT_CHANNELS, settings.PATCH_SIZE, settings.PATCH_SIZE), dtype=np.float32)
        
        grid_offsets = np.array([-0.25, 0.0, 0.25])
        
        for i, dy in enumerate(grid_offsets):
            for j, dx in enumerate(grid_offsets):
                patch[0, i, j] = params["sst"] - 0.05 * dy + 0.02 * dx
                patch[1, i, j] = params["ssh"] + 0.005 * dy
                patch[2, i, j] = params["u_current"] + 0.01 * dx
                patch[3, i, j] = params["v_current"] - 0.01 * dy
                patch[4, i, j] = params["v_wind"] + 0.1 * dy

        summary = {
            "patch_size": f"{settings.PATCH_SIZE}x{settings.PATCH_SIZE}",
            "center_sst_celsius": params["sst"],
            "center_sss_psu": params["sss"],
            "center_ssh_meters": params["ssh"],
            "center_u_current": params["u_current"],
            "center_v_current": params["v_current"],
            "center_v_wind": params["v_wind"],
            "channels": ["SST (°C)", "SSH (m)", "u_current (m/s)", "v_current (m/s)", "v_wind (m/s)"]
        }
        return patch, summary

    @classmethod
    def normalize_patch(cls, patch: np.ndarray) -> torch.Tensor:
        """
        Normalizes surface patch into PyTorch tensor: shape (1, 5, 3, 3)
        """
        norm_patch = patch.copy()
        for idx, ch_name in enumerate(cls.CHANNELS):
            ch_stat = cls.INPUT_STATS.get(ch_name, {})
            mean = ch_stat.get("mean", 0.0)
            std = ch_stat.get("std", 1.0)
            norm_patch[idx] = (norm_patch[idx] - mean) / (std if std != 0 else 1.0)
        
        return torch.from_numpy(norm_patch).unsqueeze(0).float()

    @classmethod
    def denormalize_profile(cls, norm_outputs: np.ndarray, depths: List[float]) -> List[float]:
        """
        Denormalizes model predicted temperature values back to degrees Celsius.
        """
        temps: List[float] = []
        for idx, d in enumerate(depths):
            # Check depth keys in TARGET_STATS
            key = f"{float(d)}"
            if key not in cls.TARGET_STATS:
                key = f"{float(d):.1f}"
            if key not in cls.TARGET_STATS:
                key = str(int(d))

            stat = cls.TARGET_STATS.get(key, {})
            mean = stat.get("mean", 20.0)
            std = stat.get("std", 2.0)

            norm_val = norm_outputs[idx] if idx < len(norm_outputs) else 0.0
            val_celsius = float(norm_val * std + mean)
            temps.append(round(val_celsius, 2))

        return temps
