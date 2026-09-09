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
import torch
import torch.nn as nn

from config import settings
from preprocessing import OceanPreprocessor
from schemas import PredictionResponse, LocationResponse

logger = logging.getLogger("oceanxray.model_service")

class PatchCNNEncoder(nn.Module):
    """
    Encoder module matching trained cnn_best.pt checkpoint:
    2x Conv2d(3x3) + ReLU + Flatten + Linear(144 -> 32)
    """
    def __init__(self, in_channels: int = 5, num_filters: int = 16, embedding_dim: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.fc_embed = nn.Linear(num_filters * 3 * 3, embedding_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.flatten(1)
        return self.fc_embed(x)


class PatchCNNDecoder(nn.Module):
    """
    Decoder module matching trained cnn_best.pt checkpoint:
    Linear(32 -> 64) + ReLU + Linear(64 -> 64) + ReLU + Linear(64 -> 15)
    """
    def __init__(self, embedding_dim: int = 32, hidden_dim: int = 64, num_depths: int = 15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_depths)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class OceanCNNEncoderDecoder(nn.Module):
    """
    Full OceanXRay PatchCNN Architecture:
    3x3x5 Surface Patch Input -> CNN Encoder -> 32-d Embedding -> MLP Decoder -> 15-Depth Profile
    """
    def __init__(
        self,
        in_channels: int = 5,
        num_filters: int = 16,
        embedding_dim: int = 32,
        hidden_dim: int = 64,
        num_depths: int = 15
    ):
        super().__init__()
        self.encoder = PatchCNNEncoder(in_channels=in_channels, num_filters=num_filters, embedding_dim=embedding_dim)
        self.decoder = PatchCNNDecoder(embedding_dim=embedding_dim, hidden_dim=hidden_dim, num_depths=num_depths)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embed = self.encoder(x)
        return self.decoder(embed)


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
        self.device: torch.device = torch.device("cpu")
        self.depths: List[float] = settings.STANDARD_DEPTHS
        self.load_model()
        self._initialized = True

    def load_model(self):
        """
        Loads the trained CNN model from models/cnn_best.pt if configured and available.
        Otherwise initializes the mock mode.
        """
        model_path = os.path.abspath(settings.resolve_path(settings.MODEL_PATH))
        target_mode = settings.MODEL_MODE.lower()

        if target_mode == "model":
            if os.path.exists(model_path):
                try:
                    logger.info(f"Loading real PyTorch CNN model from {model_path}...")
                    checkpoint = torch.load(model_path, map_location="cpu")
                    
                    # Extract hyperparameters from checkpoint if present
                    n_in = checkpoint.get("n_input_channels", settings.INPUT_CHANNELS)
                    n_out = checkpoint.get("n_output_depths", len(self.depths))
                    n_filt = checkpoint.get("num_filters", 16)
                    embed_dim = checkpoint.get("embedding_dim", 32)
                    hidden_dim = checkpoint.get("hidden_dim", 64)

                    self.model = OceanCNNEncoderDecoder(
                        in_channels=n_in,
                        num_filters=n_filt,
                        embedding_dim=embed_dim,
                        hidden_dim=hidden_dim,
                        num_depths=n_out
                    )
                    
                    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                        self.model.load_state_dict(checkpoint["model_state_dict"])
                    elif isinstance(checkpoint, dict):
                        self.model.load_state_dict(checkpoint)
                    elif isinstance(checkpoint, nn.Module):
                        self.model = checkpoint

                    # Configure compute device
                    if settings.USE_GPU and torch.cuda.is_available():
                        self.device = torch.device("cuda")
                        logger.info("GPU acceleration enabled. Model moved to CUDA device.")
                    else:
                        self.device = torch.device("cpu")
                        logger.info("Using CPU for model inference.")

                    self.model.to(self.device)
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
        Generates deterministic, oceanographically realistic vertical temperature profile.
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
        """
        start_time = time.perf_counter()
        
        # Prepare 3x3 surface patch (5 channels)
        patch, patch_summary = OceanPreprocessor.build_surface_patch(latitude, longitude, date_str)
        
        if self.active_mode == "model" and self.model is not None:
            # Real model inference
            tensor_input = OceanPreprocessor.normalize_patch(patch).to(self.device)
            with torch.no_grad():
                output_tensor = self.model(tensor_input)
                norm_temps = output_tensor.squeeze(0).cpu().numpy()
                pred_temps = OceanPreprocessor.denormalize_profile(norm_temps, self.depths)
                
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
                    "model_name": "OceanXRay-5Channel-PatchCNN",
                    "checkpoint": settings.MODEL_PATH,
                    "input_patch": f"{settings.PATCH_SIZE}x{settings.PATCH_SIZE} 5-channel surface patch",
                    "inference_time_ms": elapsed_ms,
                    "device": str(self.device),
                    "scientific_notice": "Inferred from trained OceanXRay PatchCNN encoder-decoder."
                },
                warning_notice=None
            )
        else:
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
                warning_notice="DEMO DATA: Model checkpoint is not active. Displayed values are deterministic mock estimates."
            )

# Global singleton service
model_service = ModelService()
