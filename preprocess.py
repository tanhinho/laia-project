import os
import glob
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import save_npz

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

DATA_PATH = 'data'
ARTIFACTS_PATH = 'artifacts'

TRAIN_YEARS = ['2010', '2011']
VAL_YEARS = ['2012']

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
RAW_CATEGORICAL_COLS = [
    'passenger_count',
    'RatecodeID'
]

def harmonize_schema(df: pd.DataFrame, year: str) -> pd.DataFrame:
    """Harmonizes the schema of the DataFrame based on the year by renaming columns to a standard format."""
    if year == '2010':
        rename_map = {
            'vendor_id': 'VendorID',
            'pickup_datetime': PICKUP_DT_COL,
            'dropoff_datetime': DROPOFF_DT_COL,
            'rate_code': 'RatecodeID',
        }
        df = df.rename(columns=rename_map)
    
    return df

def load_and_sample_data(data_dir: str, year: str, sample_size: int = None) -> pd.DataFrame:
    """Load all parquet files for a specific year, sample each one, and harmonize."""
    year_path = os.path.join(data_dir, year)
    files = glob.glob(os.path.join(year_path, 'yellow_tripdata_*.parquet'))
    if not files:
        raise FileNotFoundError(f"No parquet files found in directory: {year_path}")
    
    df_list = []
    sample_per_file = None
    if sample_size:
        sample_per_file = int(sample_size / len(files)) + 1

    for file in files:
        df = pd.read_parquet(file)
        
        if sample_per_file:
            df = df.sample(n=min(len(df), sample_per_file), random_state=42)
            
        df_list.append(df)

    df_concat = pd.concat(df_list, ignore_index=True)
    df_concat = harmonize_schema(df_concat, year)
    return df_concat

def calculate_duration(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates the trip duration in seconds."""
    df[PICKUP_DT_COL] = pd.to_datetime(df[PICKUP_DT_COL])
    df[DROPOFF_DT_COL] = pd.to_datetime(df[DROPOFF_DT_COL])
    df[TARGET_COLUMN] = (df[DROPOFF_DT_COL] - df[PICKUP_DT_COL]).dt.total_seconds()
    return df

def filter_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Filters outliers based on duration and trip distance."""
    df = df[(df[TARGET_COLUMN] >= 60) & (df[TARGET_COLUMN] <= 7200)]
    df = df[df['trip_distance'] > 0]
    return df

def clean_and_engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans categorical features and engineers new temporal features."""
    df[PICKUP_DT_COL] = pd.to_datetime(df[PICKUP_DT_COL])
    df['pickup_hour'] = df[PICKUP_DT_COL].dt.hour
    df['pickup_day_of_week'] = df[PICKUP_DT_COL].dt.dayofweek
    df['pickup_month'] = df[PICKUP_DT_COL].dt.month
    
    for col in RAW_CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(-1).astype(str)
        else:
            print(f"Warning: Column {col} not found. Skipping cleaning.")
            
    return df

def build_preprocessor(numeric_features: list, categorical_features: list) -> ColumnTransformer:
    """Builds a scikit-learn ColumnTransformer for preprocessing."""
    
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )
    return preprocessor

def main():
    """Main preprocessing script."""
    
    os.makedirs(ARTIFACTS_PATH, exist_ok=True)
    
    TRAIN_SAMPLE_SIZE = 1_000_000
    VAL_SAMPLE_SIZE = 250_000
    
    print(f"Loading, sampling, and harmonizing training data ({', '.join(TRAIN_YEARS)})...")
    train_dfs = [load_and_sample_data(DATA_PATH, year, sample_size=int(TRAIN_SAMPLE_SIZE / len(TRAIN_YEARS))) for year in TRAIN_YEARS]
    df_train = pd.concat(train_dfs)
    
    print(f"Loading, sampling, and harmonizing validation data ({', '.join(VAL_YEARS)})...")
    val_dfs = [load_and_sample_data(DATA_PATH, year, sample_size=int(VAL_SAMPLE_SIZE / len(VAL_YEARS))) for year in VAL_YEARS]
    df_val = pd.concat(val_dfs)

    print(f"Total training samples collected: {len(df_train)}")
    print(f"Total validation samples collected: {len(df_val)}")

    print("Processing data (calculating duration, filtering, cleaning, feature engineering)...")
    df_train = calculate_duration(df_train)
    df_train = filter_outliers(df_train)
    df_train = clean_and_engineer_features(df_train)
    
    df_val = calculate_duration(df_val)
    df_val = filter_outliers(df_val)
    df_val = clean_and_engineer_features(df_val)

    print("Building and fitting preprocessor pipeline...")
    all_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    
    missing_cols = [col for col in all_features if col not in df_train.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in training data: {missing_cols}")
        
    preprocessor = build_preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    
    preprocessor.fit(df_train[all_features])

    print("Transforming training and validation data...")
    X_train = preprocessor.transform(df_train[all_features])
    y_train = df_train[TARGET_COLUMN].values
    
    X_val = preprocessor.transform(df_val[all_features])
    y_val = df_val[TARGET_COLUMN].values

    print("Saving preprocessor and processed data...")
    preprocessor_path = os.path.join(ARTIFACTS_PATH, 'preprocessor.pkl')
    joblib.dump(preprocessor, preprocessor_path)
    
    save_npz(os.path.join(ARTIFACTS_PATH, 'X_train.npz'), X_train)
    np.save(os.path.join(ARTIFACTS_PATH, 'y_train.npy'), y_train)
    save_npz(os.path.join(ARTIFACTS_PATH, 'X_val.npz'), X_val)
    np.save(os.path.join(ARTIFACTS_PATH, 'y_val.npy'), y_val)
    
    print(f"--- Preprocessing complete! ---")
    print(f"Preprocessor saved to: {preprocessor_path}")
    print(f"Processed data saved in: {ARTIFACTS_PATH}")
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")

if __name__ == "__main__":
    main()