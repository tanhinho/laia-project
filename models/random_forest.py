import os
import numpy as np
import scipy.sparse
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import mlflow
import itertools


def random_forest():
    COMMIT_SHA = os.getenv('COMMIT_SHA')
    if not COMMIT_SHA:
        raise EnvironmentError("Missing required env var: COMMIT_SHA")

    def expand_grid(grid):
        keys = grid.keys()
        vals = grid.values()
        for combo in itertools.product(*vals):
            yield dict(zip(keys, combo))

    mlflow.set_tracking_uri("http://10.17.0.185:5050")
    experiment_name = "random_forest"
    artifact_uri = "mlflow-artifacts:/"
    existing_experiment = mlflow.get_experiment_by_name(experiment_name)
    if existing_experiment:
        # Check if the existing experiment has a problematic artifact location
        if existing_experiment.artifact_location.startswith("/mlflow") or \
                existing_experiment.artifact_location.startswith("file:///mlflow"):
            print(
                f"Existing experiment has Docker path artifact location: {existing_experiment.artifact_location}")
            print("Creating new experiment with remote artifact storage...")
            # Use a new experiment name for remote artifact storage
            experiment_name = "iris_classification_remote"
            try:
                experiment_id = mlflow.create_experiment(
                    experiment_name, artifact_location=artifact_uri)
                print(f"Created new experiment '{experiment_name}'")
            except Exception:
                pass
        else:
            print(f"Using existing experiment '{experiment_name}'")
    else:
        # Create new experiment with proper artifact location
        try:
            experiment_id = mlflow.create_experiment(
                experiment_name, artifact_location=artifact_uri)
            print(f"Created new experiment '{experiment_name}' with remote artifact storage")
        except Exception as e:
            print(f"Note: {e}")
    mlflow.set_experiment(experiment_name)

    # --- 1. Load Data ---

    ARTIFACTS_PATH = 'artifacts'

    param_grid = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [10, 20, 30],
    }

    param_combinations = list(expand_grid(param_grid))
    print(f"Total runs: {len(param_combinations)}")

    print("Loading processed data...")

    X_train = scipy.sparse.load_npz(os.path.join(ARTIFACTS_PATH, 'X_train.npz'))
    y_train = np.load(os.path.join(ARTIFACTS_PATH, 'y_train.npy'))

    X_val = scipy.sparse.load_npz(os.path.join(ARTIFACTS_PATH, 'X_val.npz'))
    y_val = np.load(os.path.join(ARTIFACTS_PATH, 'y_val.npy'))

    print("Data loaded successfully.")

    # --- 2. Train Model (Your Code) ---

    print("Training RandomForestRegressor model...")

    best_mse = float("inf")
    best_run_id = None
    best_params = None

    for params in param_combinations:
        with mlflow.start_run() as run:
            rf_model = RandomForestRegressor(
                random_state=42,
                **params
            )

            rf_model.fit(X_train, y_train)

            print("Model training complete.")

            # --- 3. Evaluate Model ---

            print("Evaluating model on validation data...")

            y_pred = rf_model.predict(X_val)

            mae = mean_absolute_error(y_val, y_pred)
            mse = mean_squared_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)

            mlflow.log_metric("MAE", mae)
            mlflow.log_metric("MSE", mse)
            mlflow.log_metric("R2", r2)

            signature = mlflow.models.infer_signature(X_train, rf_model.predict(X_train))
            mlflow.sklearn.log_model(rf_model, "random_forest",
                                     signature=signature, input_example=X_train[:5])

            if mse < best_mse:
                best_mse = mse
                best_run_id = run.info.run_id

            print("--- Model Evaluation ---")
            print(f"MAE: {mae:.2f} seconds")
            print(f"MSE: {mse:.2f} seconds")
            print(f"R² Score: {r2:.3f}")

    # Register the best model
    print(f"\nBest model: MSE={best_mse:.4f}")
    model_uri = f"runs:/{best_run_id}/random_forest"
    registered_model = mlflow.register_model(model_uri, "random_forest")

    try:
        client = mlflow.tracking.MlflowClient()
        client.set_registered_model_alias(
            name=experiment_name, alias=COMMIT_SHA, version=registered_model.version
        )
    except Exception as e:
        print(f"Could not set model alias: {e}")
        raise e

    print(f"Registered model version: {registered_model.version}")
    return best_mse, registered_model
