import logging
from typing import List
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from config import settings
from schemas import (
    PredictionRequest,
    PredictionResponse,
    HealthResponse,
    ConfigResponse,
    SamplePoint,
)
from model_service import model_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("oceanxray.api")

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="FastAPI Backend for OceanXRay 15-Depth Ocean Temperature Profile Prediction"
)

# CORS Configuration
allowed_origins = [
    settings.FRONTEND_ORIGIN.strip(),
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:3001",
    "http://127.0.0.1:3001",
    "http://localhost:3002",
    "http://127.0.0.1:3002",
]
# Avoid duplicates
allowed_origins = list(set(allowed_origins))

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Custom validation error handler for clean, user-friendly responses
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = []
    for err in exc.errors():
        field = " -> ".join([str(loc) for loc in err["loc"] if loc != "body"])
        msg = err["msg"]
        # Remove 'Value error, ' prefix if pydantic prepends it
        if msg.startswith("Value error, "):
            msg = msg.replace("Value error, ", "")
        errors.append(f"{field}: {msg}" if field else msg)
    
    error_summary = "; ".join(errors)
    logger.warning(f"Validation failure: {error_summary}")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": error_summary, "errors": errors}
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Returns API health status, loaded model state, and active inference mode.
    """
    return HealthResponse(
        status="ok",
        model_loaded=model_service.model_loaded,
        mode=model_service.active_mode,
        model_path=settings.ML_SERVICE_URL
    )


@app.get("/config", response_model=ConfigResponse)
async def get_configuration():
    """
    Returns North Indian Ocean geographical domain boundaries and vertical depth levels.
    """
    return ConfigResponse(
        latitude_min=settings.LAT_MIN,
        latitude_max=settings.LAT_MAX,
        longitude_min=settings.LON_MIN,
        longitude_max=settings.LON_MAX,
        num_depths=len(model_service.depths),
        depths=model_service.depths,
        patch_size=settings.PATCH_SIZE,
        domain_name="North Indian Ocean (5°N–30°N, 45°E–105°E)",
        architecture="3×3 CNN Encoder + Latent Embedding + MLP Decoder"
    )


@app.get("/sample-points", response_model=List[SamplePoint])
async def get_sample_points():
    """
    Returns curated, valid sample coordinates across the North Indian Ocean domain.
    """
    return [
        SamplePoint(
            name="Central Arabian Sea",
            latitude=18.5,
            longitude=65.0,
            region="Arabian Sea",
            description="Deep basin showing pronounced seasonal thermocline and high salinity."
        ),
        SamplePoint(
            name="Western Bay of Bengal",
            latitude=14.0,
            longitude=84.0,
            region="Bay of Bengal",
            description="Region characterized by fresh surface water layer from river runoff."
        ),
        SamplePoint(
            name="Equatorial Indian Ocean",
            latitude=6.0,
            longitude=75.0,
            region="Equatorial Channel",
            description="Deep tropical warm pool with stable high surface temperatures."
        ),
        SamplePoint(
            name="Northern Arabian Sea",
            latitude=23.5,
            longitude=63.0,
            region="Northern Basin",
            description="Winter convective mixing area with cooler surface water."
        ),
        SamplePoint(
            name="Andaman Sea",
            latitude=11.5,
            longitude=93.5,
            region="Eastern Margin",
            description="Semi-enclosed basin with warm surface layer and internal waves."
        )
    ]


@app.post("/predict", response_model=PredictionResponse)
async def predict_profile(request: PredictionRequest):
    """
    Executes ocean temperature profile prediction across 15 standard depths
    from a 3x3 satellite surface observation patch.
    """
    try:
        response = model_service.predict(
            latitude=request.latitude,
            longitude=request.longitude,
            date_str=request.date
        )
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed for request {request}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to generate temperature prediction profile. Model service encountered an internal error."
        )
