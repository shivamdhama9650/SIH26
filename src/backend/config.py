import os
from pathlib import Path
from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="allow")

    APP_NAME: str = "OceanXRay API"
    APP_VERSION: str = "1.0.0"
    MODEL_MODE: str = os.getenv("MODEL_MODE", "model").lower()  # "model" or "mock"
    ML_SERVICE_URL: str = os.getenv("ML_SERVICE_URL", "https://ml-model-oceanx.onrender.com")
    ML_TIMEOUT_SECONDS: float = float(os.getenv("ML_TIMEOUT_SECONDS", "25.0"))
    NORMALIZATION_STATS_PATH: str = os.getenv("NORMALIZATION_STATS_PATH", "normalization_stats.json")
    FRONTEND_ORIGIN: str = os.getenv("FRONTEND_ORIGIN", "http://localhost:3000")
    
    # North Indian Ocean domain bounds
    LAT_MIN: float = 5.0
    LAT_MAX: float = 30.0
    LON_MIN: float = 45.0
    LON_MAX: float = 105.0
    
    # 15 standard oceanographic depth levels (meters) trained in OceanXRay
    STANDARD_DEPTHS: List[float] = [
        0.0, 5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 125.0, 150.0, 200.0, 300.0, 500.0, 700.0, 1000.0
    ]
    
    PATCH_SIZE: int = 3  # 3x3 surface patch
    INPUT_CHANNELS: int = 5
    USE_GPU: bool = os.getenv("USE_GPU", "false").lower() == "true"
    CHANNELS: List[str] = ["sst", "ssh", "u_current", "v_current", "v_wind"]

    def resolve_path(self, rel_or_abs: str) -> str:
        """Resolves file path whether running from SIH26 root or backend/ directory."""
        if os.path.isabs(rel_or_abs) and os.path.exists(rel_or_abs):
            return rel_or_abs
        
        # Check relative to cwd
        if os.path.exists(rel_or_abs):
            return os.path.abspath(rel_or_abs)
            
        # Check relative to repo root (one level up from backend)
        root_path = Path(__file__).resolve().parent.parent / rel_or_abs
        if root_path.exists():
            return str(root_path)
            
        # Check inside backend/
        backend_path = Path(__file__).resolve().parent / rel_or_abs
        if backend_path.exists():
            return str(backend_path)
            
        return os.path.abspath(rel_or_abs)

settings = Settings()

