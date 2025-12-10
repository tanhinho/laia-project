"""Unit tests for the API endpoints."""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
from serving.app import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    model = Mock()
    model.predict = Mock(return_value=[300])
    return model


@pytest.fixture
def mock_preprocessor():
    """Create a mock preprocessor for testing."""
    preprocessor = Mock()
    preprocessor.transform = Mock(return_value=[[0.5, 1.0, 0.0]])
    return preprocessor


@patch('serving.app.model')
def test_predict_with_model(client, mock_model, mock_preprocessor):
    """Test predict endpoint with a loaded model."""
    payload = {
        "data": [
            {
                "VendorID": 2,
                "tpep_pickup_datetime": "2011-01-01 00:10:00",
                "passenger_count": 1,
                "trip_distance": 5.5,
                "PULocationID": 145,
                "DOLocationID": 145,
            }
        ]
    }

    # Send a request to the predict endpoint using the mock model and preprocessor
    with (patch('serving.app.model', mock_model),
          patch('serving.app.preprocessor', mock_preprocessor)):
        resp = client.post(
            "/predict",
            json=payload
        )

    # Basic status + JSON checks
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, dict)
    assert "predictions" in data
    assert isinstance(data["predictions"], list)
    assert len(data["predictions"]) == 1
