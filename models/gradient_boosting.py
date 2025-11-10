import os
import scipy.sparse
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow

mlflow.set_tracking_uri("http://10.17.0.185:5050")
experiment_name = "gradient_boosting_regression"
artifact_uri = "mlflow-artifacts:/"
existing_experiment = mlflow.get_experiment_by_name(experiment_name)
if existing_experiment:
    # Check if the existing experiment has a problematic artifact location
    if existing_experiment.artifact_location.startswith("/mlflow") or \
       existing_experiment.artifact_location.startswith("file:///mlflow"):
        print(f"Existing experiment has Docker path artifact location: {existing_experiment.artifact_location}")
        print("Creating new experiment with remote artifact storage...")
        # Use a new experiment name for remote artifact storage
        experiment_name = "iris_classification_remote"
        try:
            experiment_id = mlflow.create_experiment(experiment_name, artifact_location=artifact_uri)
            print(f"Created new experiment '{experiment_name}'")
        except Exception:
            pass
    else:
        print(f"Using existing experiment '{experiment_name}'")
else:
    # Create new experiment with proper artifact location
    try:
        experiment_id = mlflow.create_experiment(experiment_name, artifact_location=artifact_uri)
        print(f"Created new experiment '{experiment_name}' with remote artifact storage")
    except Exception as e:
        print(f"Note: {e}")
mlflow.set_experiment(experiment_name)

# Define the path to your artifacts
ARTIFACTS_PATH = 'artifacts'

print("Loading processed data...")

X_train = scipy.sparse.load_npz(os.path.join(ARTIFACTS_PATH, 'X_train.npz'))
y_train = np.load(os.path.join(ARTIFACTS_PATH, 'y_train.npy'))

X_val = scipy.sparse.load_npz(os.path.join(ARTIFACTS_PATH, 'X_val.npz'))
y_val = np.load(os.path.join(ARTIFACTS_PATH, 'y_val.npy'))

print("Data loaded successfully.")

print("Training GradientBoostingRegressor model...")

best_mae = float("inf")
best_run_id = None

with mlflow.start_run() as run:
    # --- Initialize the model ---
    gbr_model = GradientBoostingRegressor(
        n_estimators=300,      
        learning_rate=0.05,    
        max_depth=5,           
        random_state=42
    )

    # --- Train the model ---
    gbr_model.fit(X_train, y_train)

    print("Model training complete.")

    # --- Evaluate model ---
    print("Evaluating model on validation data...")

    y_pred = gbr_model.predict(X_val)

    # Compute metrics
    mae = mean_absolute_error(y_val, y_pred)
    rmse = mean_squared_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)

    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("R2", r2)

    signature = mlflow.models.infer_signature(X_train, gbr_model.predict(X_train))
    mlflow.sklearn.log_model(gbr_model, "gradient_boosting_model", signature=signature, input_example=X_train[:5])

    if mae < best_mae:
        best_mae = mae
        best_run_id = run.info.run_id

    print("--- Gradient Boosting Model Evaluation ---")
    print(f"Mean Absolute Error (MAE): {mae:.2f} seconds")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f} seconds")
    print(f"R² Score: {r2:.3f}")

# Register the best model
print(f"\nBest model: MAE={best_mae:.4f}")
model_uri = f"runs:/{best_run_id}/gradient_boosting_model"
registered_model = mlflow.register_model(model_uri, "gradient_boosting_regressor")
print(f"Registered model version: {registered_model.version}")