import os
import sys
from pathlib import Path
_backend_dir = str(Path(__file__).resolve().parent)
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

import time
import logging
from typing import List, Dict, Any, Optional
import numpy as np
import requests
from fastapi import HTTPException, status

from config import settings
from preprocessing import OceanPreprocessor
from schemas import PredictionResponse, LocationResponse

logger = logging.getLogger("oceanxray.model_service")


class ModelService:
    """
    Singleton service managing prediction requests via the deployed OceanXRay
    ML inference API on Render (or deterministic mock mode fallback).
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.active_mode: str = settings.MODEL_MODE.lower()
        self.model_loaded: bool = True if self.active_mode == "model" else False
        self.depths: List[float] = settings.STANDARD_DEPTHS
        self.ml_service_url: str = settings.ML_SERVICE_URL.rstrip("/")
        logger.info(f"ModelService initialized. Mode: '{self.active_mode}', Remote ML API: {self.ml_service_url}")
        self._initialized = True

    def _generate_mock_profile(self, lat: float, lon: float, date_str: str) -> List[float]:
        """
        Generates deterministic, oceanographically realistic vertical temperature profile.
        Used when MODEL_MODE is explicitly set to 'mock'.
        """
        params = OceanPreprocessor.estimate_surface_parameters(lat, lon, date_str)
        surface_temp = params["sst"]
        
        mld = 35.0 + 10.0 * np.sin(params["season_phase"]) + (20.0 - lat) * 0.5
        mld = float(np.clip(mld, 20.0, 55.0))
        deep_temp = 4.6 + 0.02 * (lat - 15.0)

        temperatures: List[float] = []
        for d in self.depths:
            if d <= mld:
                t = surface_temp - 0.008 * d
            else:
                depth_from_mld = d - mld
                scale = 130.0 + 2.0 * lat
                decay = np.exp(-depth_from_mld / scale)
                t = deep_temp + (surface_temp - deep_temp) * decay
                if lon < 75.0 and 200.0 <= d <= 500.0:
                    t += 0.4 * np.exp(-((d - 350.0) / 100.0) ** 2)

            temperatures.append(round(float(t), 2))
            
        return temperatures

    def predict(self, latitude: float, longitude: float, date_str: str) -> PredictionResponse:
        """
        Executes inference for a single target coordinate and observation date.
        Builds the 3x3 surface patch locally, then sends it via HTTP to the deployed
        OceanXRay ML service on Render.
        """
        start_time = time.perf_counter()
        
        # 1. Build 3x3 surface patch (5 channels) using backend oceanographic preprocessing
        patch, patch_summary = OceanPreprocessor.build_surface_patch(latitude, longitude, date_str)

        # 2. Remote ML Model Inference
        if self.active_mode == "model":
            # Format 5x3x3 numpy array into the exact JSON dictionary expected by /predict/cnn
            patch_dict = {
                ch: patch[idx].tolist() for idx, ch in enumerate(settings.CHANNELS)
            }
            endpoint = f"{self.ml_service_url}/predict/cnn"
            
            try:
                logger.info(f"Calling Render ML service at {endpoint} for lat={latitude}, lon={longitude}")
                resp = requests.post(
                    endpoint,
                    json={"patch": patch_dict},
                    timeout=settings.ML_TIMEOUT_SECONDS,
                    headers={"Content-Type": "application/json"}
                )
            except requests.exceptions.Timeout:
                logger.error(f"Render ML service timed out after {settings.ML_TIMEOUT_SECONDS}s at {endpoint}")
                raise HTTPException(
                    status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                    detail=f"OceanXRay ML inference service timed out ({settings.ML_TIMEOUT_SECONDS}s). "
                           "The service on Render may be waking up from cold sleep; please retry in a few seconds."
                )
            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to connect to Render ML service at {endpoint}: {e}")
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"Unable to connect to OceanXRay ML inference service on Render ({endpoint}). Error: {str(e)}"
                )

            if resp.status_code != 200:
                logger.error(f"Render ML service responded with HTTP {resp.status_code}: {resp.text}")
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"OceanXRay ML inference service returned error (HTTP {resp.status_code}): {resp.text}"
                )

            try:
                ml_data = resp.json()
                depths_raw = ml_data["depths_m"]
                temps_raw = ml_data["temperature_degC"]
                if not isinstance(depths_raw, list) or not isinstance(temps_raw, list):
                    raise ValueError("depths_m and temperature_degC must be lists")
                depths = [float(d) for d in depths_raw]
                temperatures = [round(float(t), 2) for t in temps_raw]
            except Exception as e:
                logger.error(f"Invalid response payload from Render ML service: {resp.text} - Error: {e}")
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"Invalid response payload received from OceanXRay ML inference service: {e}"
                )

            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return PredictionResponse(
                success=True,
                mode="model",
                is_demo=False,
                location=LocationResponse(latitude=latitude, longitude=longitude),
                date=date_str,
                depths=depths,
                temperatures=temperatures,
                surface_input_summary=patch_summary,
                metadata={
                    "model_name": "OceanXRay-CNN-Render",
                    "endpoint": endpoint,
                    "inference_time_ms": elapsed_ms,
                    "stats_used": ml_data.get("stats_used", True),
                    "scientific_notice": "Inferred from live OceanXRay CNN model hosted on Render."
                },
                warning_notice=None
            )

        # 3. Deterministic Mock Mode Fallback
        mock_temps = self._generate_mock_profile(latitude, longitude, date_str)
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        
        return PredictionResponse(
            success=True,
            mode="mock",
            is_demo=True,
            location=LocationResponse(latitude=latitude, longitude=longitude),
            date=date_str,
            depths=self.depths,
            temperatures=mock_temps,
            surface_input_summary=patch_summary,
            metadata={
                "model_name": "OceanXRay-Mock-DemoEngine",
                "input_patch": "3x3 synthetic surface grid",
                "inference_time_ms": elapsed_ms,
                "scientific_notice": "DEMO DATA: Operating in mock mode."
            },
            warning_notice="DEMO DATA: Operating in mock mode. Displayed values are deterministic mock estimates."
        )


# Global singleton service
model_service = ModelService()
