import requests

BASE_URL = "http://localhost:9001"


def test_predict_returns_200_and_json():
    url = f"{BASE_URL}/predict"

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

    resp = requests.post(url, json=payload)

    # Basic status + JSON checks
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, dict)
    assert "predictions" in data
    assert isinstance(data["predictions"], list)
    assert len(data["predictions"]) == 1
