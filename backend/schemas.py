from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator
from datetime import date as dt_date
from config import settings

class PredictionRequest(BaseModel):
    latitude: float = Field(..., description="Latitude coordinate in decimal degrees (5.0 to 30.0)")
    longitude: float = Field(..., description="Longitude coordinate in decimal degrees (45.0 to 105.0)")
    date: str = Field(..., description="Observation date in ISO format (YYYY-MM-DD)")

    @field_validator("latitude")
    @classmethod
    def validate_latitude(cls, v: float) -> float:
        if not (settings.LAT_MIN <= v <= settings.LAT_MAX):
            raise ValueError(
                f"Latitude must be between {settings.LAT_MIN} and {settings.LAT_MAX} degrees North. Got {v}."
            )
        return round(v, 4)

    @field_validator("longitude")
    @classmethod
    def validate_longitude(cls, v: float) -> float:
        if not (settings.LON_MIN <= v <= settings.LON_MAX):
            raise ValueError(
                f"Longitude must be between {settings.LON_MIN} and {settings.LON_MAX} degrees East. Got {v}."
            )
        return round(v, 4)

    @field_validator("date")
    @classmethod
    def validate_date(cls, v: str) -> str:
        try:
            dt_date.fromisoformat(v)
        except ValueError:
            raise ValueError(f"Date '{v}' must be in valid YYYY-MM-DD ISO format.")
        return v


class LocationResponse(BaseModel):
    latitude: float
    longitude: float


class PredictionResponse(BaseModel):
    success: bool
    mode: str = Field(..., description="'model' for real CNN inference, 'mock' for deterministic demo")
    is_demo: bool = Field(..., description="Flag to explicitly mark demonstration / mock predictions")
    location: LocationResponse
    date: str
    depths: List[float]
    temperatures: List[float]
    surface_input_summary: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any]
    warning_notice: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    mode: str
    model_path: str


class ConfigResponse(BaseModel):
    latitude_min: float
    latitude_max: float
    longitude_min: float
    longitude_max: float
    num_depths: int
    depths: List[float]
    patch_size: int
    domain_name: str
    architecture: str


class SamplePoint(BaseModel):
    name: str
    latitude: float
    longitude: float
    region: str
    description: str
