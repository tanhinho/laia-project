import mlflow
import os
from gradient_boosting import gradient_boosting
from random_forest import random_forest
from linear_regression import linear_regression

COMMIT_SHA = os.getenv('COMMIT_SHA')
if not COMMIT_SHA:
    raise EnvironmentError("Missing required env var: COMMIT_SHA")


def main():
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
    except Exception as e:
        print(f"Could not set model alias: {e}")
        raise e


if __name__ == "__main__":
    main()
