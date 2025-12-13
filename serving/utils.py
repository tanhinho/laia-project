import pandas as pd
import numpy as np

PICKUP_DT_COL = 'tpep_pickup_datetime'
NUMERIC_FEATURES = ['trip_distance']
RAW_CATEGORICAL_COLS = [
    'passenger_count',
    'VendorID',
    'PULocationID',
    'DOLocationID'
]


def process_data_for_inference(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies feature engineering for inference (prediction) without calculating target.
    This version does NOT require tpep_dropoff_datetime.
    """
    # 1. Parse pickup datetime
    df[PICKUP_DT_COL] = pd.to_datetime(df[PICKUP_DT_COL])

    # 2. Feature Engineering (only from pickup time)
    df['pickup_hour'] = df[PICKUP_DT_COL].dt.hour
    df['pickup_day_of_week'] = df[PICKUP_DT_COL].dt.dayofweek
    df['pickup_month'] = df[PICKUP_DT_COL].dt.month

    # 3. Clean Categorical Features
    for col in RAW_CATEGORICAL_COLS:
        if col in df.columns:
            # Convert to string, then to category for memory efficiency
            df[col] = df[col].fillna(-1).astype(str).astype('category')
        else:
            # Add default value if column is missing
            print(f"Warning: Column {col} not found. Adding default value.")
            df[col] = '1'

    # 4. Optimize Numeric Dtypes
    df[NUMERIC_FEATURES] = df[NUMERIC_FEATURES].astype(np.float32)

    return df
