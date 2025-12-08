import requests

url = "http://localhost:9001/predict"

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

response = requests.post(url, json=payload)
print(f"Status Code: {response.status_code}")
print(f"Response: {response.json()}")
