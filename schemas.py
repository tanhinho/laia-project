from pydantic import BaseModel
from typing import List


class TripData(BaseModel):
    VendorID: int
    tpep_pickup_datetime: str
    passenger_count: int
    trip_distance: float
    PULocationID: int
    DOLocationID: int


class InputPayload(BaseModel):
    records: List[TripData]
