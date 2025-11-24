import os
import joblib
import pandas as pd
import mlflow.pyfunc
from fastapi import FastAPI, HTTPException
from schemas import InputPayload
from mlflow.tracking import MlflowClient

# Import your custom preprocessing logic
from preprocess import process_data


app = FastAPI()

# Configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5050")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "linear_regression")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "Production")


# artifact holder vars
model = None
preprocessor = None


@app.on_event("startup")
def load_artifacts():
    global model, preprocessor
    try:
        print(f"Loading model: {MODEL_NAME}@{MODEL_ALIAS}...")
        model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"

        # 1. Load the Model
        model = mlflow.sklearn.load_model(model_uri)

        # 2. Download and Load the Preprocessor artifact
        client = MlflowClient()
        mv = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
        artifact_uri = client.get_model_version_download_uri(
            MODEL_NAME, mv.version)
        local_path = mlflow.artifacts.download_artifacts(
            artifact_uri=f"{artifact_uri}/preprocessor/preprocessor.pkl")
        preprocessor = joblib.load(local_path)

        print("Artifacts loaded successfully.")
    except Exception as e:
        print(f"Error loading artifacts: {e}")


@app.post("/predict")
def predict(payload: InputPayload):
    if not model or not preprocessor:
        raise HTTPException(
            status_code=503, detail="Model not loaded. Please try again later.")

    try:
        # Convert Pydantic list to DataFrame
        data_dicts = [item.dict() for item in payload.data]
        df = pd.DataFrame(data_dicts)

        # 1. Preprocess (Cleaning + Feature Eng) using your script
        df_clean = process_data(df)

        # 2. Transform (Scaling/Encoding) using loaded preprocessor
        # Note: Ensure columns match what the preprocessor expects
        X_input = preprocessor.transform(df_clean)

        # 3. Predict
        predictions = model.predict(X_input)

        return {"predictions": predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
