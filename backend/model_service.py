import os
import time
import logging
from typing import List, Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn

from config import settings
from preprocessing import OceanPreprocessor
from schemas import PredictionResponse, LocationResponse

logger = logging.getLogger("oceanxray.model_service")

class OceanCNNEncoderDecoder(nn.Module):
    """
    Locked OceanXRay Model Architecture:
    3x3 Surface Patch Input
    ↓
    CNN Encoder
    ↓
    Latent Embedding (128-d)
    ↓
    MLP Decoder
    ↓
    15-Depth Temperature Profile
    """
    def __init__(self, in_channels: int = 4, num_depths: int = 15):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=2, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(64 * 2 * 2, 128),
            nn.ReLU(inplace=True)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_depths)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedding = self.encoder(x)
        out = self.decoder(embedding)
        return out


class ModelService:
    """
    Singleton service managing model lifecycle and inference.
    Supports both real PyTorch CNN inference and deterministic scientific mock mode.
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
        self.model: Optional[nn.Module] = None
        self.model_loaded: bool = False
        self.active_mode: str = "mock"
        self.depths: List[float] = settings.STANDARD_DEPTHS
        self.load_model()
        self._initialized = True

    def load_model(self):
        """
        Loads the trained CNN model from models/cnn_best.pt if configured and available.
        Otherwise initializes the mock mode.
        """
        model_path = os.path.abspath(settings.MODEL_PATH)
        target_mode = settings.MODEL_MODE.lower()

        if target_mode == "model":
            if os.path.exists(model_path):
                try:
                    logger.info(f"Loading real PyTorch CNN model from {model_path}...")
                    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
                    
                    self.model = OceanCNNEncoderDecoder(
                        in_channels=settings.INPUT_CHANNELS,
                        num_depths=len(self.depths)
                    )
                    
                    if isinstance(checkpoint, dict):
                        if "model_state_dict" in checkpoint:
                            self.model.load_state_dict(checkpoint["model_state_dict"])
                        elif "state_dict" in checkpoint:
                            self.model.load_state_dict(checkpoint["state_dict"])
                        else:
                            self.model.load_state_dict(checkpoint)
                            
                        # Load custom depths if saved with checkpoint
                        if "depths" in checkpoint:
                            self.depths = [float(d) for d in checkpoint["depths"]]
                    elif isinstance(checkpoint, nn.Module):
                        self.model = checkpoint

                    self.model.eval()
                    self.model_loaded = True
                    self.active_mode = "model"
                    logger.info("Real CNN model successfully loaded in evaluation mode.")
                    return
                except Exception as e:
                    logger.error(f"Failed to load checkpoint at {model_path}: {e}. Falling back to mock mode.")
                    self.model = None
                    self.model_loaded = False
                    self.active_mode = "mock"
            else:
                logger.warning(
                    f"MODEL_MODE is set to 'model' but checkpoint '{model_path}' was not found. "
                    f"Falling back to MOCK mode."
                )
                self.model = None
                self.model_loaded = False
                self.active_mode = "mock"
        else:
            logger.info("Operating in deterministic MOCK/DEMO mode per configuration.")
            self.model = None
            self.model_loaded = False
            self.active_mode = "mock"

    def _generate_mock_profile(self, lat: float, lon: float, date_str: str) -> List[float]:
        """
        Generates deterministic, oceanographically realistic vertical temperature profile
        for the North Indian Ocean basin.
        Emulates:
        - Warm Mixed Layer (0m - 30m)
        - Sharp Thermocline (50m - 200m)
        - Intermediate Water (300m - 700m)
        - Abyssal Deep Layer (1000m - 1500m)
        """
        params = OceanPreprocessor.estimate_surface_parameters(lat, lon, date_str)
        surface_temp = params["sst"]
        
        # Mixed layer depth (MLD): ~25-45m depending on region and monsoon phase
        mld = 35.0 + 10.0 * np.sin(params["season_phase"]) + (20.0 - lat) * 0.5
        mld = float(np.clip(mld, 20.0, 55.0))
        
        # Deep ocean floor asymptotic temperature (~4.2 to 5.0 °C at 1500m)
        deep_temp = 4.6 + 0.02 * (lat - 15.0)

        temperatures: List[float] = []
        for d in self.depths:
            if d <= mld:
                # Isothermal surface mixed layer
                t = surface_temp - 0.008 * d
            else:
                # Thermocline & deep ocean exponential decay
                # T(z) = T_deep + (T_surface - T_deep) * exp(- (z - MLD) / scale)
                depth_from_mld = d - mld
                scale = 130.0 + 2.0 * lat  # thermocline thickness parameter
                decay = np.exp(-depth_from_mld / scale)
                t = deep_temp + (surface_temp - deep_temp) * decay
                
                # Secondary intermediate adjustment (Red Sea / Persian Gulf water outflow in Arabian Sea)
                if lon < 75.0 and 200.0 <= d <= 500.0:
                    t += 0.4 * np.exp(-((d - 350.0) / 100.0) ** 2)

            temperatures.append(round(float(t), 2))
            
        return temperatures

    def predict(self, latitude: float, longitude: float, date_str: str) -> PredictionResponse:
        """
        Executes inference for a single target coordinate and observation date.
        """
        start_time = time.perf_counter()
        
        # Prepare 3x3 surface patch
        patch, patch_summary = OceanPreprocessor.build_surface_patch(latitude, longitude, date_str)
        
        if self.active_mode == "model" and self.model is not None:
            # Real model inference with torch.no_grad()
            tensor_input = OceanPreprocessor.normalize_patch(patch)
            with torch.no_grad():
                output_tensor = self.model(tensor_input)
                # Raw outputs correspond to predicted temperatures
                pred_temps = output_tensor.squeeze(0).cpu().numpy().tolist()
                pred_temps = [round(float(t), 2) for t in pred_temps]
                
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            
            return PredictionResponse(
                success=True,
                mode="model",
                is_demo=False,
                location=LocationResponse(latitude=latitude, longitude=longitude),
                date=date_str,
                depths=self.depths,
                temperatures=pred_temps,
                surface_input_summary=patch_summary,
                metadata={
                    "model_name": "OceanXRay-CNN-MLP",
                    "checkpoint": settings.MODEL_PATH,
                    "input_patch": "3x3 surface grid",
                    "inference_time_ms": elapsed_ms,
                    "device": "cpu",
                    "scientific_notice": "Predicted using trained OceanXRay CNN model checkpoint."
                },
                warning_notice=None
            )
        else:
            # Deterministic scientific mock mode
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
                    "scientific_notice": "DEMO DATA: Operating in mock mode pending final models/cnn_best.pt checkpoint."
                },
                warning_notice="DEMO DATA: Model checkpoint 'cnn_best.pt' is not currently active. Displayed values are realistic deterministic mock estimates."
            )

# Global singleton service
model_service = ModelService()
