import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, ANY
from datetime import date
from api.main import app, WeatherFeatures, PredictionResponse, API_KEY

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "ok"
    assert "model_ready" in data

@patch("api.main.latest_weather_features")
def test_latest_weather(mock_latest_weather):
    mock_latest_weather.return_value = WeatherFeatures(
        Date=date.today(),
        Location="Sydney"
    )
    response = client.get("/latest_weather?station_name=Sydney")
    assert response.status_code == 200
    data = response.json()
    assert data["Location"] == "Sydney"

@patch("api.main.load_model")
@patch("api.main.latest_weather_features")
def test_predict_use_latest(mock_latest_weather, mock_load_model):
    from pydantic import BaseModel
    # Patch model predict_proba to emulate a fitted model
    mock_latest_weather.return_value = WeatherFeatures(
        Date=date.today(),
        Location="Sydney"
    )
    # The FastAPI endpoint expects the model to have a predict_proba method,
    # and returns a 2D list [[prob_0, prob_1]] where prob_1 is rain.
    class DummyModel:
        def predict_proba(self, X):
            # Accepts any input, always returns valid shape
            # Return as np.array to ensure correct access in endpoint
            import numpy as np
            return np.array([[0.2, 0.8]])
    mock_load_model.return_value = DummyModel()
    headers = {"Authorization": f"Bearer {API_KEY}"}
    response = client.post(
        "/predict?use_latest=true&station_name=Sydney",
        headers=headers
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert "rain_tomorrow" in data
    assert "rain_probability" in data
    assert isinstance(data["rain_tomorrow"], int)
    assert 0.0 <= data["rain_probability"] <= 1.0

@patch("api.main.load_model")
def test_predict_with_payload(mock_load_model):
    # minimal payload, just required fields
    payload = {
        "Date": str(date.today()),
        "Location": "Sydney"
    }
    class DummyModel:
        def predict_proba(self, X):
            import numpy as np
            return np.array([[0.4, 0.6]])
    mock_load_model.return_value = DummyModel()
    headers = {"Authorization": f"Bearer {API_KEY}"}
    response = client.post(
        "/predict?station_name=Sydney",
        headers=headers,
        json=payload
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert "rain_tomorrow" in data

def test_predict_auth_required():
    payload = {
        "Date": str(date.today()),
        "Location": "Sydney"
    }
    response = client.post(
        "/predict?station_name=Sydney",
        json=payload
    )
    assert response.status_code == 401

def test_predict_missing_parameters():
    headers = {"Authorization": f"Bearer {API_KEY}"}
    response = client.post(
        "/predict?station_name=Sydney",
        headers=headers
        # No payload and use_latest False
    )
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data

@patch("api.main.load_model")
def test_model_error(mock_load_model):
    mock_load_model.side_effect = FileNotFoundError("model missing")
    payload = {
        "Date": str(date.today()),
        "Location": "Sydney"
    }
    headers = {"Authorization": f"Bearer {API_KEY}"}
    response = client.post(
        "/predict?station_name=Sydney",
        headers=headers,
        json=payload
    )
    assert response.status_code == 503
    data = response.json()
    assert "model missing" in data["detail"]