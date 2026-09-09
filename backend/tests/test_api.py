import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add backend directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main import app
from config import settings

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["mode"] in ["mock", "model"]
    assert "model_loaded" in data

def test_config_endpoint():
    response = client.get("/config")
    assert response.status_code == 200
    data = response.json()
    assert data["latitude_min"] == 5.0
    assert data["latitude_max"] == 30.0
    assert data["longitude_min"] == 45.0
    assert data["longitude_max"] == 105.0
    assert data["num_depths"] == 15
    assert len(data["depths"]) == 15
    assert data["depths"][0] == 0.0
    assert data["depths"][-1] == 1500.0

def test_sample_points_endpoint():
    response = client.get("/sample-points")
    assert response.status_code == 200
    points = response.json()
    assert len(points) >= 3
    for pt in points:
        assert 5.0 <= pt["latitude"] <= 30.0
        assert 45.0 <= pt["longitude"] <= 105.0

def test_predict_endpoint_valid():
    payload = {
        "latitude": 18.5,
        "longitude": 72.5,
        "date": "2025-01-15"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["mode"] == "mock"
    assert data["is_demo"] is True
    assert len(data["depths"]) == 15
    assert len(data["temperatures"]) == 15
    # Surface temperature should be higher than deep ocean temperature
    assert data["temperatures"][0] > data["temperatures"][-1]
    assert data["location"]["latitude"] == 18.5
    assert data["location"]["longitude"] == 72.5
    assert "warning_notice" in data

def test_predict_latitude_out_of_bounds():
    payload = {
        "latitude": 35.0,  # Valid is 5-30
        "longitude": 72.5,
        "date": "2025-01-15"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    assert "Latitude must be between" in response.json()["detail"]

def test_predict_longitude_out_of_bounds():
    payload = {
        "latitude": 15.0,
        "longitude": 40.0,  # Valid is 45-105
        "date": "2025-01-15"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    assert "Longitude must be between" in response.json()["detail"]

def test_predict_invalid_date():
    payload = {
        "latitude": 15.0,
        "longitude": 70.0,
        "date": "not-a-valid-date"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    assert "must be in valid YYYY-MM-DD" in response.json()["detail"]
