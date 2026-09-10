"""
app.py
======
Minimal FastAPI deployment for OceanXRay inference.

Run locally:
    uvicorn app:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  /health         -> basic status + whether real normalization stats were found
    POST /predict/mlp     -> point-based MLP baseline
    POST /predict/cnn     -> CNN encoder + MLP decoder (main/locked model)
"""

from typing import Dict, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from predictor import OceanXRayPredictor

app = FastAPI(title="OceanXRay Inference API", version="1.0")
predictor = OceanXRayPredictor()


class MLPRequest(BaseModel):
    values: Dict[str, float]


class CNNRequest(BaseModel):
    patch: Dict[str, List[List[float]]]


@app.get("/health")
def health():
    return {
        "status": "ok",
        "normalization_stats_found": predictor.stats_available,
        "cnn_channels": predictor.cnn_channels,
        "mlp_channels": predictor.mlp_channels,
        "depths_m": predictor.depths,
    }


@app.post("/predict/mlp")
def predict_mlp(req: MLPRequest):
    try:
        return predictor.predict_mlp(req.values)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/cnn")
def predict_cnn(req: CNNRequest):
    try:
        return predictor.predict_cnn(req.patch)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
