import numpy as np
import torch
from typing import Tuple, Dict, Any
from datetime import date as dt_date
from config import settings

class OceanPreprocessor:
    """
    Handles preprocessing and normalization for the OceanXRay 3x3 surface patch.
    Maintains strict scientific channel ordering:
    Channel 0: Sea Surface Temperature (SST, °C)
    Channel 1: Sea Surface Salinity (SSS, PSU)
    Channel 2: Sea Surface Height / Sea Level Anomaly (SSH/SLA, m)
    Channel 3: Normalized Latitudinal/Coriolis Feature
    """
    
    # Established North Indian Ocean surface normalization statistics
    STATS = {
        "sst_mean": 27.8,
        "sst_std": 1.9,
        "sss_mean": 34.6,
        "sss_std": 1.8,
        "ssh_mean": 0.05,
        "ssh_std": 0.18,
    }

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
        
        # Seasonal solar cycle (summer warming peaking around May/June and post-monsoon Oct)
        season_phase = 2 * np.pi * (day_of_year - 80) / 365.25
        seasonal_anomaly = 1.2 * np.sin(season_phase)
        
        # Latitudinal gradient (equator is warmer, northern Arabian Sea can be cooler in winter)
        lat_effect = -0.15 * (lat - 10.0)
        
        # Longitudinal gradient (Bay of Bengal is typically warmer and fresher than Arabian Sea)
        lon_effect = 0.02 * (lon - 70.0)
        
        sst = float(np.clip(28.0 + lat_effect + lon_effect + seasonal_anomaly, 21.0, 32.5))
        
        # Salinity: Bay of Bengal (lon > 80) has high river runoff (fresher: ~31-33 PSU)
        # Arabian Sea (lon < 75) has high evaporation (saline: ~35.5-36.5 PSU)
        if lon > 80.0:
            base_sss = 32.8 - 0.03 * (lon - 80.0)
        else:
            base_sss = 35.8 - 0.02 * (75.0 - lon)
        sss = float(np.clip(base_sss, 30.0, 37.0))
        
        # Sea Surface Height Anomaly (-0.25 to +0.25 m)
        ssh = float(0.04 * np.sin(season_phase + lat * 0.1) + 0.03 * np.cos(lon * 0.08))
        
        return {
            "sst": round(sst, 2),
            "sss": round(sss, 2),
            "ssh": round(ssh, 3),
            "season_phase": round(season_phase, 3)
        }

    @classmethod
    def build_surface_patch(cls, lat: float, lon: float, date_str: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Builds the 3x3 surface patch across 4 channels: (4, 3, 3)
        """
        params = cls.estimate_surface_parameters(lat, lon, date_str)
        patch = np.zeros((settings.INPUT_CHANNELS, settings.PATCH_SIZE, settings.PATCH_SIZE), dtype=np.float32)
        
        # Create subtle spatial gradients across 3x3 grid (-0.25° to +0.25° resolution)
        grid_offsets = np.array([-0.25, 0.0, 0.25])
        
        for i, dy in enumerate(grid_offsets):
            for j, dx in enumerate(grid_offsets):
                # SST channel
                patch[0, i, j] = params["sst"] - 0.05 * dy + 0.02 * dx
                # SSS channel
                patch[1, i, j] = params["sss"] - 0.02 * dx
                # SSH channel
                patch[2, i, j] = params["ssh"] + 0.005 * dy
                # Latitude/Coriolis proxy channel
                patch[3, i, j] = (lat + dy) / 30.0
                
        summary = {
            "patch_size": f"{settings.PATCH_SIZE}x{settings.PATCH_SIZE}",
            "center_sst_celsius": params["sst"],
            "center_sss_psu": params["sss"],
            "center_ssh_meters": params["ssh"],
            "channels": ["SST (°C)", "SSS (PSU)", "SSH (m)", "Normalized Latitude"]
        }
        return patch, summary

    @classmethod
    def normalize_patch(cls, patch: np.ndarray) -> torch.Tensor:
        """
        Normalizes surface patch into PyTorch tensor: shape (1, 4, 3, 3)
        """
        norm_patch = patch.copy()
        norm_patch[0] = (norm_patch[0] - cls.STATS["sst_mean"]) / cls.STATS["sst_std"]
        norm_patch[1] = (norm_patch[1] - cls.STATS["sss_mean"]) / cls.STATS["sss_std"]
        norm_patch[2] = (norm_patch[2] - cls.STATS["ssh_mean"]) / cls.STATS["ssh_std"]
        # Channel 3 is already normalized latitude [0, 1]
        
        tensor = torch.from_numpy(norm_patch).unsqueeze(0).float()
        return tensor
