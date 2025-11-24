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

NUMERIC_FEATURES = ['trip_distance']
CATEGORICAL_FEATURES = [
    'passenger_count',
    'RatecodeID',
    'pickup_hour',
    'pickup_day_of_week',
    'pickup_month'
]
COLUMNS_TO_LOAD = [
    PICKUP_DT_COL,
    DROPOFF_DT_COL,
    'trip_distance',
    'passenger_count',
    'RatecodeID'
]
HARMONIZE_MAP = {
    '2010': {
        'vendor_id': 'VendorID',
        'pickup_datetime': PICKUP_DT_COL,
        'dropoff_datetime': DROPOFF_DT_COL,
        'rate_code': 'RatecodeID',
    }
}
RAW_CATEGORICAL_COLS = [
    'passenger_count',
    'RatecodeID'
]

# --- Helper Functions ---
def load_data(data_path: str, years: list, sample_size: int) -> pd.DataFrame:
    """
    Loads, samples, and harmonizes data from parquet files for given years
    using a memory-efficient, streaming approach with pyarrow.
    This version uses scan_batches() and accesses the .record_batch attribute.
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
    
    print(f"Loading {len(all_files)} files (from {years}) with target size {sample_size} (sample fraction: {sample_fraction:.6f})")

    df_list = []
    
    for file in all_files:
        year = os.path.basename(os.path.dirname(file))
        rename_map = HARMONIZE_MAP.get(year, {})
        load_cols_map = {v: k for k, v in rename_map.items()}
        load_cols = [load_cols_map.get(col, col) for col in COLUMNS_TO_LOAD]

        try:
            dataset = ds.dataset(file, format="parquet")
            filters = (
                (ds.field('trip_distance') > 0) &
                (ds.field('trip_distance') < 100)
            )
            scanner = dataset.scanner(columns=load_cols, filter=filters)
            
            file_sampled_chunks = []
            
            for batch in scanner.scan_batches():
                df_chunk = batch.record_batch.to_pandas()
                df_sample = df_chunk.sample(frac=sample_fraction, random_state=42)
                file_sampled_chunks.append(df_sample)
            
            if file_sampled_chunks:
                df = pd.concat(file_sampled_chunks, ignore_index=True)
                
                if rename_map:
                    df = df.rename(columns=rename_map)
                
                df_list.append(df)
            
        except Exception as e:
            print(f"Warning: Could not process file {file}. Error: {e}")

    if not df_list:
        raise ValueError("No data was loaded. Check file paths and schema.")
        
    df_concat = pd.concat(df_list, ignore_index=True)
    
    if len(df_concat) > sample_size:
        df_concat = df_concat.sample(n=sample_size, random_state=42)
    elif len(df_concat) == 0:
        raise ValueError("No data loaded after sampling. Check sample size and data paths.")
        
    return df_concat

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies all cleaning, feature engineering, and outlier filtering.
    This function is designed to be memory-efficient.
    """
    # 1. Calculate Target Variable
    df[PICKUP_DT_COL] = pd.to_datetime(df[PICKUP_DT_COL])
    df[DROPOFF_DT_COL] = pd.to_datetime(df[DROPOFF_DT_COL])
    df[TARGET_COLUMN] = (df[DROPOFF_DT_COL] - df[PICKUP_DT_COL]).dt.total_seconds()
    
    # 2. Filter Outliers
    df = df[(df[TARGET_COLUMN] >= 60) & (df[TARGET_COLUMN] <= 7200)].copy()

    # 3. Feature Engineering
    df['pickup_hour'] = df[PICKUP_DT_COL].dt.hour
    df['pickup_day_of_week'] = df[PICKUP_DT_COL].dt.dayofweek
    df['pickup_month'] = df[PICKUP_DT_COL].dt.month
    
    # 4. Clean Categorical Features
    for col in RAW_CATEGORICAL_COLS:
        if col in df.columns:
            # Convert to string, then to category for memory efficiency
            df[col] = df[col].fillna(-1).astype(str).astype('category')
        else:
            print(f"Warning: Column {col} not found. Skipping cleaning.")
            
    # 5. Optimize Numeric Dtypes
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

# --- Main Function ---

def main(train_size: int):
    """
    Main preprocessing script.
    - Takes train_size (70%) as input.
    - Calculates holdout_size (30%).
    - Trains on 2010-2011 data.
    - Loads 2012 data and splits it 50/50 for validation and test sets.
    """
    
    os.makedirs(ARTIFACTS_PATH, exist_ok=True)
    
    # --- 1. Calculate Split Sizes ---
    ratio_30_over_70 = 0.30 / 0.70
    holdout_size = int(train_size * ratio_30_over_70)
    
    print(f"--- Starting Preprocessing (Temporal Split 70/15/15) ---")
    print(f"Target Train Size (70%): {train_size} (from {TRAIN_YEARS})")
    print(f"Calculated Holdout Size (30%): {holdout_size} (from {HOLDOUT_YEARS})")
    print(f" -> This will be split into: {holdout_size // 2} (Val) and {holdout_size - (holdout_size // 2)} (Test)")
    print("-" * 50)
    
    # --- 2. Process Training Data ---
    print("--- Processing Training Data ---")
    df_train = load_data(DATA_PATH, TRAIN_YEARS, sample_size=train_size)
    print(f"Cleaning, engineering, and filtering training data...")
    df_train = process_data(df_train)
    print(f"Training samples after filtering: {len(df_train)}")

    if len(df_train) == 0:
        raise ValueError("No training data left after filtering. Aborting.")

    print("Building and fitting preprocessor pipeline...")
    all_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    preprocessor = build_preprocessor()
    preprocessor.fit(df_train[all_features])

    print("Transforming training data...")
    X_train = preprocessor.transform(df_train[all_features])
    y_train = df_train[TARGET_COLUMN].values

    print("Saving training artifacts...")
    preprocessor_path = os.path.join(ARTIFACTS_PATH, 'preprocessor.pkl')
    joblib.dump(preprocessor, preprocessor_path)
    save_npz(os.path.join(ARTIFACTS_PATH, 'X_train.npz'), X_train)
    np.save(os.path.join(ARTIFACTS_PATH, 'y_train.npy'), y_train)
    print(f"Training data shape: {X_train.shape}")
    
    # --- 3. Free Memory ---
    print("\n--- Freeing Training Data Memory ---")
    del df_train, X_train, y_train
    
    # --- 4. Process Holdout Data (2012) ---
    print("\n--- Processing Holdout Data (for Val/Test) ---")
    df_holdout = load_data(DATA_PATH, HOLDOUT_YEARS, sample_size=holdout_size)
    print(f"Cleaning, engineering, and filtering holdout data...")
    df_holdout = process_data(df_holdout)
    print(f"Holdout samples after filtering: {len(df_holdout)}")

    if len(df_holdout) == 0:
        raise ValueError("No holdout data left after filtering. Aborting.")

    print("Transforming holdout data...")
    X_holdout = preprocessor.transform(df_holdout[all_features])
    y_holdout = df_holdout[TARGET_COLUMN].values
    
    del df_holdout # Free memory
    
    # --- 5. Split Holdout Data into Val/Test ---
    print("\nSplitting holdout data into Validation and Test sets (50/50 split)...")
    X_val, X_test, y_val, y_test = train_test_split(
        X_holdout, 
        y_holdout, 
        test_size=0.5, 
        random_state=42
    )
    
    del X_holdout, y_holdout # Free memory
    
    # --- 6. Save Validation and Test Artifacts ---
    print("Saving validation artifacts...")
    save_npz(os.path.join(ARTIFACTS_PATH, 'X_val.npz'), X_val)
    np.save(os.path.join(ARTIFACTS_PATH, 'y_val.npy'), y_val)
    print(f"Validation data shape: {X_val.shape}")

    print("Saving test artifacts...")
    save_npz(os.path.join(ARTIFACTS_PATH, 'X_test.npz'), X_test)
    np.save(os.path.join(ARTIFACTS_PATH, 'y_test.npy'), y_test)
    print(f"Test data shape: {X_test.shape}")

    print(f"\n--- Preprocessing complete! ---")
    print(f"All artifacts saved in: {ARTIFACTS_PATH}")

# --- Main execution block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess NYC Taxi data with temporal split.")
    
    parser.add_argument(
        '--train-size',
        type=int,
        default=1_000_000,
        help='Number of samples for the training set (represents 70% of total).'
    )
    
    args = parser.parse_args()
    
    main(train_size=args.train_size)