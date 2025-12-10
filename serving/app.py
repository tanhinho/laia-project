import os
import joblib
import pandas as pd
import mlflow.pyfunc
from fastapi import FastAPI, HTTPException
from schemas import InputPayload
from mlflow.tracking import MlflowClient
from utils import process_data_for_inference
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Histogram, Counter


OUTPUT_MINUTES_HISTOGRAM = Histogram(
    'model_prediction_minutes', 
    'Distribution of predicted duration in minutes', 
    buckets=[1, 5, 10, 15, 20, 30, 45, 60, 90, 120]
)

INPUT_DISTANCE_HISTOGRAM = Histogram(
    'input_feature_distance', 
    'Distribution of input trip distance',
    buckets=[0, 2, 5, 10, 20, 50, 100]
)

app = FastAPI()

# Configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

if MLFLOW_TRACKING_URI is None:
    raise ValueError("MLFLOW_TRACKING_URI environment variable is not set.")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME")

if MODEL_NAME is None:
    MODEL_NAME = "linear_regression"
    print(f"MLFLOW_MODEL_NAME not set. Defaulting to: {MODEL_NAME}")


MODEL_ALIAS = os.getenv("MODEL_ALIAS")
if MODEL_ALIAS is None:
    MODEL_ALIAS = "production"
    print(f"MODEL_ALIAS not set. Defaulting to: {MODEL_ALIAS}")


# artifact holder vars
model = None
preprocessor = None

# prometheus dep
instrumentator = Instrumentator().instrument(app)

@app.on_event("startup")
def load_artifacts():
    # expose metrics endpoint for scraping
    instrumentator.expose(app)
    
    global model, preprocessor
    try:
        print(f"Loading model: {MODEL_NAME}@{MODEL_ALIAS}...", flush=True)
        model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"

        # 1. Load the Model
        print(f"Downloading model from {model_uri}...", flush=True)
        model = mlflow.sklearn.load_model(model_uri)
        print("Model loaded successfully.", flush=True)

        # 2. Download and Load the Preprocessor artifact
        print("Loading preprocessor...", flush=True)
        client = MlflowClient()
        mv = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)

        # Download from the run artifacts, not model registry artifacts
        print(f"Downloading preprocessor from run {mv.run_id}...", flush=True)
        local_path = mlflow.artifacts.download_artifacts(
            run_id=mv.run_id,
            artifact_path="preprocessor/preprocessor.pkl")
        preprocessor = joblib.load(local_path)

        print("Artifacts loaded successfully.", flush=True)
    except Exception as e:
        print(f"Error loading artifacts: {e}", flush=True)
        raise


@app.post("/predict")
def predict(payload: InputPayload):
    if not model or not preprocessor:
        raise HTTPException(
            status_code=503, detail="Model not loaded. Please try again later.")

    try:
        
        data_dicts = [item.model_dump() for item in payload.data]
        df = pd.DataFrame(data_dicts)

        # Log key features before processing to catch raw data drift
        if 'trip_distance' in df.columns:
             for val in df['trip_distance']:
                 INPUT_DISTANCE_HISTOGRAM.observe(val)

        df_clean = process_data_for_inference(df)
        X_input = preprocessor.transform(df_clean)
        seconds_predictions = model.predict(X_input)
        minutes_predictions = seconds_predictions / 60.0

        # logs for stability check
        for pred in minutes_predictions:
            OUTPUT_MINUTES_HISTOGRAM.observe(pred)

        return {"predictions": minutes_predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
