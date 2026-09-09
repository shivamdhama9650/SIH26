import os
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="allow")

    APP_NAME: str = "OceanXRay API"
    APP_VERSION: str = "1.0.0"
    MODEL_MODE: str = os.getenv("MODEL_MODE", "mock").lower()  # "mock" or "model"
    MODEL_PATH: str = os.getenv("MODEL_PATH", "models/cnn_best.pt")
    FRONTEND_ORIGIN: str = os.getenv("FRONTEND_ORIGIN", "http://localhost:3000")
    
    # North Indian Ocean domain bounds
    LAT_MIN: float = 5.0
    LAT_MAX: float = 30.0
    LON_MIN: float = 45.0
    LON_MAX: float = 105.0
    
    # 15 standard oceanographic depth levels (meters)
    STANDARD_DEPTHS: List[float] = [
        0.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 150.0, 200.0, 300.0, 400.0, 500.0, 700.0, 1000.0, 1500.0
    ]
    
    PATCH_SIZE: int = 3  # 3x3 surface patch
    INPUT_CHANNELS: int = 4  # e.g., SST, SSS, SSH/SLA, Surface Dynamics

settings = Settings()
