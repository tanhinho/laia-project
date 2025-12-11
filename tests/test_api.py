import pytest
import numpy as np
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
from serving.app import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    model = Mock()
    # Return prediction in seconds -> then minutes

    # side_effect- return array matching input size

    def predict_side_effect(X):
        # Return one prediction per input row
        return np.array([300] * len(X))
    model.predict = Mock(side_effect=predict_side_effect)
    return model


@pytest.fixture
def mock_preprocessor():
    """Create a mock preprocessor for testing."""
    preprocessor = Mock()
    preprocessor.transform = Mock(
        return_value=np.array([[0.5, 1.0, 0.0, 2.5, 1.0]]))
    return preprocessor


@pytest.fixture
def sample_payload():
    """Create a sample valid payload for testing."""
    return {
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


@pytest.fixture
def sample_payload_multiple():
    """Create a sample payload with multiple trips."""
    return {
        "data": [
            {
                "VendorID": 2,
                "tpep_pickup_datetime": "2011-01-01 00:10:00",
                "passenger_count": 1,
                "trip_distance": 5.5,
                "PULocationID": 145,
                "DOLocationID": 145,
            },
            {
                "VendorID": 1,
                "tpep_pickup_datetime": "2011-01-01 01:15:00",
                "passenger_count": 2,
                "trip_distance": 10.2,
                "PULocationID": 100,
                "DOLocationID": 200,
            },
            {
                "VendorID": 2,
                "tpep_pickup_datetime": "2011-01-01 02:30:00",
                "passenger_count": 4,
                "trip_distance": 3.1,
                "PULocationID": 50,
                "DOLocationID": 75,
            }
        ]
    }


def test_predict_without_model(client, sample_payload):
    """Test predict endpoint when no model is loaded."""
    with (patch('serving.app.model', None),
          patch('serving.app.preprocessor', None)):
        response = client.post("/predict", json=sample_payload)

        assert response.status_code == 503
        data = response.json()
        assert 'detail' in data
        assert 'Model not loaded' in data['detail']


def test_predict_without_preprocessor(client, sample_payload, mock_model):
    """Test predict endpoint when preprocessor is not loaded."""
    with (patch('serving.app.model', mock_model),
          patch('serving.app.preprocessor', None)):
        response = client.post("/predict", json=sample_payload)

        assert response.status_code == 503
        data = response.json()
        assert 'detail' in data
        assert 'Model not loaded' in data['detail']


def test_predict_with_model(client, sample_payload, mock_model, mock_preprocessor):
    """Test predict endpoint with a loaded model and preprocessor."""
    with (patch('serving.app.model', mock_model),
          patch('serving.app.preprocessor', mock_preprocessor)):
        response = client.post("/predict", json=sample_payload)

        assert response.status_code == 200
        data = response.json()
        assert 'predictions' in data
        assert isinstance(data['predictions'], list)
        assert len(data['predictions']) == 1
        # Verify conversion from seconds to minutes (300s = 5 min)
        assert data['predictions'][0] == 5.0


def test_predict_multiple_trips(client, sample_payload_multiple, mock_preprocessor):
    """Test predict endpoint with multiple trips."""
    # Mock preprocessor to return 3 transformed samples
    mock_preprocessor.transform = Mock(return_value=np.array([
        [0.5, 1.0, 0.0, 2.5, 1.0],
        [0.8, 2.0, 1.0, 3.0, 2.0],
        [0.3, 0.5, 0.0, 1.5, 0.5]
    ]))

    # Create a mock model that returns different predictions for each trip
    mock_model_multi = Mock()
    mock_model_multi.predict = Mock(return_value=np.array([300, 600, 900]))

    with (patch('serving.app.model', mock_model_multi),
          patch('serving.app.preprocessor', mock_preprocessor)):
        response = client.post("/predict", json=sample_payload_multiple)

        assert response.status_code == 200
        data = response.json()
        assert 'predictions' in data
        assert len(data['predictions']) == 3
        # Verify predictions are in minutes
        # 300s, 600s, 900s converted
        assert data['predictions'] == [5.0, 10.0, 15.0]


def test_predict_invalid_payload_missing_field(client, mock_model, mock_preprocessor):
    """Test predict endpoint with missing required field."""
    invalid_payload = {
        "data": [
            {
                "VendorID": 2,
                # Missing tpep_pickup_datetime
                "passenger_count": 1,
                "trip_distance": 5.5,
                "PULocationID": 145,
                "DOLocationID": 145,
            }
        ]
    }

    with (patch('serving.app.model', mock_model),
          patch('serving.app.preprocessor', mock_preprocessor)):
        response = client.post("/predict", json=invalid_payload)

        # FastAPI will return 422 for validation errors
        assert response.status_code == 422
        data = response.json()
        assert 'detail' in data


def test_predict_invalid_payload_wrong_type(client, mock_model, mock_preprocessor):
    """Test predict endpoint with wrong data type."""
    invalid_payload = {
        "data": [
            {
                "VendorID": "not_an_int",  # Should be int
                "tpep_pickup_datetime": "2011-01-01 00:10:00",
                "passenger_count": 1,
                "trip_distance": 5.5,
                "PULocationID": 145,
                "DOLocationID": 145,
            }
        ]
    }

    with (patch('serving.app.model', mock_model),
          patch('serving.app.preprocessor', mock_preprocessor)):
        response = client.post("/predict", json=invalid_payload)

        assert response.status_code == 422



def test_predict_preprocessor_error(client, sample_payload, mock_model, mock_preprocessor):
    """Test predict endpoint when preprocessor raises an error."""
    mock_preprocessor.transform = Mock(
        side_effect=Exception("Preprocessing failed"))

    with (patch('serving.app.model', mock_model),
          patch('serving.app.preprocessor', mock_preprocessor)):
        response = client.post("/predict", json=sample_payload)

        assert response.status_code == 500
        data = response.json()
        assert 'detail' in data
        assert 'Preprocessing failed' in data['detail']


def test_predict_model_error(client, sample_payload, mock_model, mock_preprocessor):
    """Test predict endpoint when model raises an error."""
    mock_model.predict = Mock(side_effect=Exception("Prediction failed"))

    with (patch('serving.app.model', mock_model),
          patch('serving.app.preprocessor', mock_preprocessor)):
        response = client.post("/predict", json=sample_payload)

        assert response.status_code == 500
        data = response.json()
        assert 'detail' in data
        assert 'Prediction failed' in data['detail']


def test_predict_correct_dataframe_creation(client, sample_payload, mock_model, mock_preprocessor):
    """Test that predict correctly creates DataFrame from request."""
    with (patch('serving.app.model', mock_model),
          patch('serving.app.preprocessor', mock_preprocessor),
          patch('serving.app.process_data_for_inference') as mock_process):

        # Mock the data processing function
        mock_process.return_value = MagicMock()

        response = client.post("/predict", json=sample_payload)

        assert response.status_code == 200
        # Verify that preprocessor and model were called
        mock_process.assert_called_once()
        mock_preprocessor.transform.assert_called_once()
        mock_model.predict.assert_called_once()


def test_predict_with_optional_ratecode(client, mock_model, mock_preprocessor):
    """Test predict endpoint with optional RatecodeID field."""
    payload_with_ratecode = {
        "data": [
            {
                "VendorID": 2,
                "tpep_pickup_datetime": "2011-01-01 00:10:00",
                "passenger_count": 1,
                "trip_distance": 5.5,
                "RatecodeID": 2, 
                "PULocationID": 145,
                "DOLocationID": 145,
            }
        ]
    }

    with (patch('serving.app.model', mock_model),
          patch('serving.app.preprocessor', mock_preprocessor)):
        response = client.post("/predict", json=payload_with_ratecode)

        assert response.status_code == 200
        data = response.json()
        assert 'predictions' in data
        assert len(data['predictions']) == 1


def test_predict_seconds_to_minutes_conversion(client, sample_payload, mock_preprocessor):
    """Test that predictions are correctly converted from seconds to minutes."""
    # Create a specific mock model for this test
    mock_model_conversion = Mock()
    mock_model_conversion.predict = Mock(return_value=np.array([1800]))

    with (patch('serving.app.model', mock_model_conversion),
          patch('serving.app.preprocessor', mock_preprocessor)):
        response = client.post("/predict", json=sample_payload)

        assert response.status_code == 200
        data = response.json()
        assert 'predictions' in data
        assert data['predictions'][0] == 30.0  # 1800 seconds = 30 minutes
