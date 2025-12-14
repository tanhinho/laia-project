import mlflow
import os
import argparse
from gradient_boosting import gradient_boosting
from random_forest import random_forest
from linear_regression import linear_regression

COMMIT_SHA = os.getenv('COMMIT_SHA')
if not COMMIT_SHA:
    raise EnvironmentError("Missing required env var: COMMIT_SHA")


def get_best_existing_model():
    """
    Fetch the best model from existing MLflow runs based on MSE.
    Returns the model object with run_id and name, and artifact_path.
    """
    client = mlflow.tracking.MlflowClient()

    # Get the experiment (assuming all models are in the same experiment)
    experiment_name = os.getenv('MLFLOW_EXPERIMENT_NAME', 'default')
    experiment = client.get_experiment_by_name(experiment_name)

    if not experiment:
        raise RuntimeError(f"No experiment found with name: {experiment_name}")

    # Search for all runs in the experiment, ordered by MSE
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="",
        order_by=["metrics.mse ASC"],
        max_results=1
    )

    if not runs:
        raise RuntimeError(
            "No existing runs found. You must train models first.")

    best_run = runs[0]
    best_mse = best_run.data.metrics.get('mse')

    print(f"Best existing model found:")
    print(f"  Run ID: {best_run.info.run_id}")
    print(f"  MSE: {best_mse}")
    print(f"  Tags: {best_run.data.tags}")

    # Check which artifact was logged
    artifacts = client.list_artifacts(best_run.info.run_id)
    artifact_path = None

    for artifact in artifacts:
        if artifact.path in ['linear_regression', 'random_forest', 'gradient_boosting_model']:
            artifact_path = artifact.path
            break

    if not artifact_path:
        raise RuntimeError(
            "Could not determine model artifact path from best run")

    # Map artifact path to registered model name
    name_by_artifact = {
        "linear_regression": "linear_regression",
        "random_forest": "random_forest",
        "gradient_boosting_model": "gradient_boosting_regressor",
    }

    registered_model_name = name_by_artifact.get(artifact_path)

    # Create a mock model object with the necessary attributes
    class ModelInfo:
        def __init__(self, run_id, name):
            self.run_id = run_id
            self.name = name

    return ModelInfo(best_run.info.run_id, registered_model_name), artifact_path


def main():
    parser = argparse.ArgumentParser(
        description='Train models or use existing best model')
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip training and use the best existing model from MLflow')
    args = parser.parse_args()

    if args.skip_training:
        print("Skipping training - using best existing model from MLflow...")
        best_model, artifact_path = get_best_existing_model()
        best_model_name = best_model.name
    else:
        print("Running Linear Regression Model...")
        lr_mse, lr_model = linear_regression()
        print("Linear Regression Model run complete.\n")

        print("Running Random Forest Model...")
        rf_mse, rf_model = random_forest()
        print("Random Forest Model run complete.\n")

        print("Running Gradient Boosting Model...")
        gb_mse, gb_model = gradient_boosting()
        print("Gradient Boosting Model run complete.\n")

        # Determine the best model based on MSE
        best_mse = min(lr_mse, rf_mse, gb_mse)
        if best_mse == lr_mse:
            best_model = lr_model
        elif best_mse == rf_mse:
            best_model = rf_model
        else:
            best_model = gb_model

        # Use the actual registered-model name returned by mlflow.register_model
        best_model_name = best_model.name

        # Map registered-model name -> artifact path used in log_model(...)
        artifact_path_by_name = {
            "linear_regression": "linear_regression",
            "random_forest": "random_forest",
            "gradient_boosting_regressor": "gradient_boosting_model",
        }

        artifact_path = artifact_path_by_name[best_model_name]

    best_model_uri = f"runs:/{best_model.run_id}/{artifact_path}"
    model = mlflow.register_model(best_model_uri, "best_model")

    try:
        client = mlflow.tracking.MlflowClient()
        client.set_registered_model_alias(
            name=model.name, alias=COMMIT_SHA, version=model.version
        )
        print(
            f"✅ Set alias '{COMMIT_SHA}' for {model.name} version {model.version}")
    except Exception as e:
        print(f"Could not set model alias: {e}")
        raise e


if __name__ == "__main__":
    main()
