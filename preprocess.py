import os
import glob
import joblib
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from pyarrow import fs
from scipy.sparse import save_npz
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import argparse

# --- Configuration ---
DATA_PATH = 'data'
ARTIFACTS_PATH = 'artifacts'

TRAIN_YEARS = ['2011', '2012']
HOLDOUT_YEARS = ['2013']

# --- Column Definitions ---
PICKUP_DT_COL = 'tpep_pickup_datetime'
DROPOFF_DT_COL = 'tpep_dropoff_datetime'
TARGET_COLUMN = 'duration_sec'

COLUMNS_TO_LOAD = [
    PICKUP_DT_COL,
    DROPOFF_DT_COL,
    'passenger_count',
    'VendorID',
    'trip_distance',
    'PULocationID',
    'DOLocationID'
]

NUMERIC_FEATURES = [
    'trip_distance' 
]

CATEGORICAL_FEATURES = [
    'passenger_count',
    'VendorID',
    'PULocationID',
    'DOLocationID',
    'pickup_hour',
    'pickup_day_of_week',
    'pickup_month'
]

RAW_CATEGORICAL_COLS = [
    'passenger_count',
    'VendorID',
    'PULocationID',
    'DOLocationID'
]

# --- Helper Functions ---

def load_data(data_path: str, years: list, sample_size: int) -> pd.DataFrame:
    """
    Loads, samples, and cleans data from parquet files.
    """
    all_files = []
    for year in years:
        year_path = os.path.join(data_path, year)
        year_files = glob.glob(os.path.join(year_path, 'yellow_tripdata_*.parquet'))
        all_files.extend(year_files)

    if not all_files:
        raise FileNotFoundError(f"No parquet files found for years {years} in {data_path}")

    ESTIMATED_ROWS_PER_FILE = 5_000_000
    total_estimated_rows = len(all_files) * ESTIMATED_ROWS_PER_FILE
    sample_fraction = min(1.0, sample_size / total_estimated_rows)
    sample_fraction = max(0.00001, sample_fraction)

    print(f"Loading {len(all_files)} files (from {years}) with target size {sample_size}")

    df_list = []

    for file in all_files:
        try:
            dataset = ds.dataset(file, format="parquet")
            
            filters = (
                (ds.field('trip_distance') > 0) & 
                (ds.field('trip_distance') < 100)
            )
            
            scanner = dataset.scanner(columns=COLUMNS_TO_LOAD, filter=filters)
            file_sampled_chunks = []

            for batch in scanner.scan_batches():
                df_chunk = batch.record_batch.to_pandas()
                df_sample = df_chunk.sample(frac=sample_fraction, random_state=42)
                file_sampled_chunks.append(df_sample)

            if file_sampled_chunks:
                df = pd.concat(file_sampled_chunks, ignore_index=True)
                df_list.append(df)

        except Exception as e:
            print(f"Warning: Could not process file {file}. Error: {e}")

    if not df_list:
        raise ValueError("No data was loaded. Check file paths and schema.")

    df_concat = pd.concat(df_list, ignore_index=True)

    if len(df_concat) > sample_size:
        df_concat = df_concat.sample(n=sample_size, random_state=42)

    return df_concat


def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies cleaning and feature engineering.
    """
    # 1. Calculate Target Variable (Duration)
    df[PICKUP_DT_COL] = pd.to_datetime(df[PICKUP_DT_COL])
    df[DROPOFF_DT_COL] = pd.to_datetime(df[DROPOFF_DT_COL])
    df[TARGET_COLUMN] = (df[DROPOFF_DT_COL] - df[PICKUP_DT_COL]).dt.total_seconds()

    # 2. Filter Outliers (1 min to 2 hours)
    df = df[(df[TARGET_COLUMN] >= 60) & (df[TARGET_COLUMN] <= 7200)].copy()

    # 3. Filter invalid trip distances
    df = df[df['trip_distance'] > 0].copy()

    # 4. Feature Engineering
    df['pickup_hour'] = df[PICKUP_DT_COL].dt.hour
    df['pickup_day_of_week'] = df[PICKUP_DT_COL].dt.dayofweek
    df['pickup_month'] = df[PICKUP_DT_COL].dt.month

    # 5. Clean Categorical Features
    for col in RAW_CATEGORICAL_COLS:
        if col in df.columns:
            # Cast VendorID, etc. to string so they are treated as categories
            df[col] = df[col].fillna(-1).astype(str).astype('category')

    # 6. Optimize Numeric Dtypes
    df[NUMERIC_FEATURES] = df[NUMERIC_FEATURES].astype(np.float32)
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(np.float32)

    return df


def build_preprocessor() -> ColumnTransformer:
    """Builds a scikit-learn ColumnTransformer for preprocessing."""
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERIC_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ],
        remainder='drop'
    )
    return preprocessor

def process_data_for_inference(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies feature engineering for inference (prediction) without calculating target.
    
    Expected Input Columns:
    - VendorID
    - tpep_pickup_datetime
    - passenger_count
    - trip_distance
    - PULocationID
    - DOLocationID
    """
    # 1. Parse pickup datetime
    if PICKUP_DT_COL in df.columns:
        df[PICKUP_DT_COL] = pd.to_datetime(df[PICKUP_DT_COL])

        # 2. Feature Engineering (only from pickup time)
        df['pickup_hour'] = df[PICKUP_DT_COL].dt.hour
        df['pickup_day_of_week'] = df[PICKUP_DT_COL].dt.dayofweek
        df['pickup_month'] = df[PICKUP_DT_COL].dt.month
    
    # 3. Clean Categorical Features
    # Ensure all categorical columns are strings/categories to match training schema
    for col in RAW_CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(-1).astype(str).astype('category')
        else:
             # Just in case a column is missing, we create it with a default to avoid crashing
             df[col] = '-1'
             df[col] = df[col].astype('category')

    # 4. Optimize Numeric Dtypes
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype(np.float32)

    return df

# --- Main Function ---

def main(train_size: int):
    os.makedirs(ARTIFACTS_PATH, exist_ok=True)

    # 1. Calculate Split Sizes
    ratio_30_over_70 = 0.30 / 0.70
    holdout_size = int(train_size * ratio_30_over_70)

    print(f"--- Starting Preprocessing ---")
    print(f"Train Size: {train_size} (2011-2012)")
    print(f"Holdout Size: {holdout_size} (2013)")

    # 2. Process Training Data
    print("\n--- Processing Training Data ---")
    df_train = load_data(DATA_PATH, TRAIN_YEARS, sample_size=train_size)
    df_train = process_data(df_train)
    
    print("Fitting preprocessor...")
    all_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    preprocessor = build_preprocessor()
    preprocessor.fit(df_train[all_features])

    print("Transforming training data...")
    X_train = preprocessor.transform(df_train[all_features])
    
    # Target is raw seconds (no log transform)
    y_train = df_train[TARGET_COLUMN].values

    print("Saving training artifacts...")
    joblib.dump(preprocessor, os.path.join(ARTIFACTS_PATH, 'preprocessor.pkl'))
    save_npz(os.path.join(ARTIFACTS_PATH, 'X_train.npz'), X_train)
    np.save(os.path.join(ARTIFACTS_PATH, 'y_train.npy'), y_train)

    del df_train, X_train, y_train

    # 3. Process Holdout Data (2013)
    print("\n--- Processing Holdout Data ---")
    df_holdout = load_data(DATA_PATH, HOLDOUT_YEARS, sample_size=holdout_size)
    df_holdout = process_data(df_holdout)

    X_holdout = preprocessor.transform(df_holdout[all_features])
    
    y_holdout = df_holdout[TARGET_COLUMN].values

    del df_holdout

    # 4. Split Holdout into Val/Test
    print("Splitting Holdout into Val/Test (50/50)...")
    X_val, X_test, y_val, y_test = train_test_split(
        X_holdout, y_holdout, test_size=0.5, random_state=42
    )

    print("Saving validation/test artifacts...")
    save_npz(os.path.join(ARTIFACTS_PATH, 'X_val.npz'), X_val)
    np.save(os.path.join(ARTIFACTS_PATH, 'y_val.npy'), y_val)
    save_npz(os.path.join(ARTIFACTS_PATH, 'X_test.npz'), X_test)
    np.save(os.path.join(ARTIFACTS_PATH, 'y_test.npy'), y_test)

    print(f"\n--- Preprocessing complete! ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-size', type=int, default=1_000_000)
    args = parser.parse_args()
    main(train_size=args.train_size)