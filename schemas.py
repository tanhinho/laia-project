from pydantic import BaseModel
from typing import List


class TripData(BaseModel):
    VendorID: int
    tpep_pickup_datetime: str
    passenger_count: int
    trip_distance: float
    RatecodeID: int
    PULocationID: int
    DOLocationID: int


class InputPayload(BaseModel):
    data: List[TripData]
