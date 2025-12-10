import os
import glob
import time
import numpy as np
import pandas as pd
import requests
import pyarrow.dataset as ds
import argparse

# Configuration
API_URL = "http://localhost:9001/predict"
DATA_PATH = 'data'
HOLDOUT_YEARS = ['2013']
NUM_SAMPLES = 1000
NUM_RUNS = 100

PICKUP_DT_COL = 'tpep_pickup_datetime'
DROPOFF_DT_COL = 'tpep_dropoff_datetime'


def load_test_samples(data_path: str, years: list, num_samples: int) -> pd.DataFrame:
    """Load random consecutive samples from the holdout data."""
    all_files = []
    for year in years:
        year_path = os.path.join(data_path, year)
        year_files = glob.glob(os.path.join(
            year_path, 'yellow_tripdata_*.parquet'))
        all_files.extend(year_files)

    if not all_files:
        raise FileNotFoundError(
            f"No parquet files found for years {years} in {data_path}")

    # Load from first file
    file = all_files[0]
    columns_to_load = [
        'VendorID', PICKUP_DT_COL, DROPOFF_DT_COL, 'trip_distance',
        'passenger_count', 'RatecodeID', 'PULocationID', 'DOLocationID'
    ]

    dataset = ds.dataset(file, format="parquet")
    filters = (ds.field('trip_distance') > 0) & (
        ds.field('trip_distance') < 100)
    df = dataset.scanner(columns=columns_to_load,
                         filter=filters).to_table().to_pandas()

    # Calculate duration
    df[PICKUP_DT_COL] = pd.to_datetime(df[PICKUP_DT_COL])
    df[DROPOFF_DT_COL] = pd.to_datetime(df[DROPOFF_DT_COL])
    df['duration_sec'] = (df[DROPOFF_DT_COL] -
                          df[PICKUP_DT_COL]).dt.total_seconds()

    # Filter valid durations
    df = df[(df['duration_sec'] >= 60) & (df['duration_sec'] <= 7200)].copy()

    if len(df) < num_samples:
        return df

    # Select random consecutive samples
    start_idx = np.random.randint(0, len(df) - num_samples + 1)
    return df.iloc[start_idx:start_idx + num_samples].reset_index(drop=True)


def prepare_payload(df: pd.DataFrame) -> dict:
    """Convert DataFrame to API payload."""
    data_list = []
    for _, row in df.iterrows():
        data_list.append({
            "VendorID": int(row['VendorID']) if pd.notna(row['VendorID']) else 1,
            "tpep_pickup_datetime": row[PICKUP_DT_COL].strftime('%Y-%m-%d %H:%M:%S'),
            "passenger_count": int(row['passenger_count']) if pd.notna(row['passenger_count']) else 1,
            "trip_distance": float(row['trip_distance']),
            "RatecodeID": int(row['RatecodeID']) if pd.notna(row['RatecodeID']) else 1,
            "PULocationID": int(row['PULocationID']) if pd.notna(row['PULocationID']) else 1,
            "DOLocationID": int(row['DOLocationID']) if pd.notna(row['DOLocationID']) else 1
        })
    return {"data": data_list}


def test_api(payload: dict, y_true: np.ndarray) -> dict:
    """Test API and compute metrics."""
    try:
        start_time = time.time()
        response = requests.post(API_URL, json=payload, timeout=30)
        latency = time.time() - start_time

        if response.status_code != 200:
            return None

        predictions = np.array(response.json()['predictions'])

        # Convert to minutes
        y_true_min = y_true / 60
        predictions_min = predictions

        # Compute MSE in minutes
        mse = np.mean((y_true_min - predictions_min) ** 2)

        return {
            'mse': mse,
            'num_samples': len(predictions),
            'latency': latency
        }

    except Exception as e:
        print(f"Error: {e}")
        return None


def run_test(run_num: int, num_samples: int) -> dict:
    """Run a single test iteration."""
    print(f"\n--- Run {run_num} ---")

    # Load samples
    df_test = load_test_samples(DATA_PATH, HOLDOUT_YEARS, num_samples)
    y_true = df_test['duration_sec'].values

    # Test API
    payload = prepare_payload(df_test)
    metrics = test_api(payload, y_true)

    if metrics:
        print(
            f"MSE: {metrics['mse']:.2f} min | Latency: {metrics['latency']/metrics['num_samples']*1000:.2f} ms/sample")

    return metrics


def main(num_runs: int = NUM_RUNS, num_samples: int = NUM_SAMPLES):
    print("=" * 60)
    print(f"API Drift Testing - {num_runs} run(s) of {num_samples} samples")
    print("=" * 60)

    results = []
    for i in range(1, num_runs + 1):
        metrics = run_test(i, num_samples)
        if metrics:
            results.append(metrics)
        else:
            print(f"Run {i} failed")

    if results:
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)

        mse_values = [r['mse'] for r in results]
        latency_values = [r['latency']/r['num_samples']*1000 for r in results]

        print(f"Total runs:           {len(results)}")
        print(f"MSE (avg):            {np.mean(mse_values):.2f} minutes")
        print(f"MSE (std):            {np.std(mse_values):.2f} minutes")
        print(f"MSE (min):            {np.min(mse_values):.2f} minutes")
        print(f"MSE (max):            {np.max(mse_values):.2f} minutes")
        print(f"Latency (avg):        {np.mean(latency_values):.2f} ms/sample")
        print("=" * 60)

    else:
        print("\nAll tests failed. Please check if the API is running.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test API for model drift")
    parser.add_argument('--runs', type=int, default=NUM_RUNS,
                        help='Number of test runs')
    parser.add_argument('--samples', type=int,
                        default=NUM_SAMPLES, help='Samples per run')

    args = parser.parse_args()
    main(num_runs=args.runs, num_samples=args.samples)
