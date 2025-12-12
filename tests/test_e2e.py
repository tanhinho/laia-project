import os
import pytest
import requests
import time


# Base URL for FastAPI service
FASTAPI_BASE_URL = os.getenv('FASTAPI_BASE_URL', 'http://localhost:9001')


def test_service_health():
    """Test that FastAPI service is healthy and model is loaded."""
    response = requests.get(f"{FASTAPI_BASE_URL}/health")
    assert response.status_code == 200
    data = response.json()
    assert data['status'] == 'healthy'
    assert data['model_loaded'] is True


def test_prediction_single_trip():
    """Test prediction with a single taxi trip."""
    payload = {
        "data": [
            {
                "VendorID": 2,
                "tpep_pickup_datetime": "2011-01-01 00:10:00",
                "passenger_count": 1,
                "trip_distance": 5.5,
                "PULocationID": 145,
                "DOLocationID": 145
            }
        ]
    }

    response = requests.post(
        f"{FASTAPI_BASE_URL}/predict",
        json=payload,
        headers={"Content-Type": "application/json"}
    )

    assert response.status_code == 200
    data = response.json()
    assert 'predictions' in data
    assert len(data['predictions']) == 1
    assert isinstance(data['predictions'][0], (int, float))
    assert data['predictions'][0] > 0


def test_prediction_multiple_trips():
    """Test prediction with multiple taxi trips."""
    payload = {
        "data": [
            {
                "VendorID": 2,
                "tpep_pickup_datetime": "2011-01-01 00:10:00",
                "passenger_count": 1,
                "trip_distance": 5.5,
                "PULocationID": 145,
                "DOLocationID": 145
            },
            {
                "VendorID": 1,
                "tpep_pickup_datetime": "2011-01-01 01:15:00",
                "passenger_count": 2,
                "trip_distance": 10.2,
                "PULocationID": 100,
                "DOLocationID": 200
            }
        ]
    }

    response = requests.post(
        f"{FASTAPI_BASE_URL}/predict",
        json=payload,
        headers={"Content-Type": "application/json"}
    )

    assert response.status_code == 200
    data = response.json()
    assert 'predictions' in data
    assert len(data['predictions']) == 2


def test_api_validation_error():
    """Test API error handling with invalid input."""
    invalid_payload = {
        "data": [
            {
                "VendorID": 2,
                # Missing required field: tpep_pickup_datetime
                "passenger_count": 1,
                "trip_distance": 5.5,
                "PULocationID": 145,
                "DOLocationID": 145
            }
        ]
    }

    response = requests.post(
        f"{FASTAPI_BASE_URL}/predict",
        json=invalid_payload,
        headers={"Content-Type": "application/json"}
    )

    assert response.status_code == 422
    data = response.json()
    assert 'detail' in data
