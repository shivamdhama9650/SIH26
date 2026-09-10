import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import requests
import sys
import os

# Add backend directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main import app
from config import settings

client = TestClient(app)

MOCK_ML_RESPONSE = {
    "model": "cnn",
    "depths_m": [0.0, 5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 125.0, 150.0, 200.0, 300.0, 500.0, 700.0, 1000.0],
    "temperature_degC": [28.75, 28.64, 28.62, 28.60, 28.54, 28.12, 26.62, 23.82, 20.60, 17.90, 14.93, 12.56, 10.84, 9.42, 7.32],
    "stats_used": True
}

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["mode"] in ["mock", "model"]
    assert "model_loaded" in data
    assert "https://ml-model-oceanx.onrender.com" in data["model_path"]

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
    assert data["depths"][-1] == 1000.0

def test_sample_points_endpoint():
    response = client.get("/sample-points")
    assert response.status_code == 200
    points = response.json()
    assert len(points) >= 3
    for pt in points:
        assert 5.0 <= pt["latitude"] <= 30.0
        assert 45.0 <= pt["longitude"] <= 105.0

@patch("requests.post")
def test_predict_endpoint_success_with_mocked_ml(mock_post):
    # Configure mock response from Render ML service
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = MOCK_ML_RESPONSE
    mock_post.return_value = mock_resp

    payload = {
        "latitude": 18.5,
        "longitude": 72.5,
        "date": "2025-01-15"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    assert data["success"] is True
    assert data["mode"] == "model"
    assert data["is_demo"] is False
    assert len(data["depths"]) == 15
    assert len(data["temperatures"]) == 15
    assert data["temperatures"][0] == 28.75
    assert data["temperatures"][-1] == 7.32
    assert data["location"]["latitude"] == 18.5
    assert data["location"]["longitude"] == 72.5
    assert "surface_input_summary" in data
    assert "metadata" in data
    assert data["metadata"]["model_name"] == "OceanXRay-CNN-Render"

    # Verify requests.post was called with the exact expected payload structure
    mock_post.assert_called_once()
    args, kwargs = mock_post.call_args
    assert "patch" in kwargs["json"]
    patch_dict = kwargs["json"]["patch"]
    for ch in ["sst", "ssh", "u_current", "v_current", "v_wind"]:
        assert ch in patch_dict
        assert len(patch_dict[ch]) == 3
        assert len(patch_dict[ch][0]) == 3

@patch("requests.post")
def test_predict_endpoint_ml_timeout(mock_post):
    mock_post.side_effect = requests.exceptions.Timeout("Connection timed out")
    payload = {
        "latitude": 18.5,
        "longitude": 72.5,
        "date": "2025-01-15"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 504
    assert "timed out" in response.json()["detail"].lower()

@patch("requests.post")
def test_predict_endpoint_ml_connection_error(mock_post):
    mock_post.side_effect = requests.exceptions.ConnectionError("Failed to reach host")
    payload = {
        "latitude": 18.5,
        "longitude": 72.5,
        "date": "2025-01-15"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 502
    assert "unable to connect" in response.json()["detail"].lower()

@patch("requests.post")
def test_predict_endpoint_ml_service_error_500(mock_post):
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.text = "Internal Server Error in model"
    mock_post.return_value = mock_resp

    payload = {
        "latitude": 18.5,
        "longitude": 72.5,
        "date": "2025-01-15"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 502

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
