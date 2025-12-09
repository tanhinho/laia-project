### 1. Build the Training Image

```bash
# The '-t' flag names the image 'training-image'
# The '-f' flag points to the Dockerfile
# The '.' is the build context (current directory)
sudo docker build -t training-image -f Dockerfile.training .
```

### 2. Run Model Training

```bash
sudo docker run --rm \
  -v $(pwd)/data:/app/data \
  -e MLFLOW_TRACKING_URI=http://10.17.0.185:5050 \
  -e MLFLOW_EXPERIMENT_NAME=linear_regression \
  training-image
```

**Environment Variables:**

- `MLFLOW_TRACKING_URI`: MLflow server endpoint
- `MLFLOW_EXPERIMENT_NAME`: Name of the experiment (e.g., linear_regression, random_forest)

### 3. Build the Serving Image

```bash
sudo docker build -t serving-image -f Dockerfile.serving .
```

### 4. Run the Inference Server

```bash
sudo docker run -d --name fastapi \
-p 9001:9001 \
-e MLFLOW_TRACKING_URI=http://10.17.0.185:5050 \
-e MLFLOW_MODEL_NAME=linear_regression \
-e MODEL_ALIAS=production \
serving-image
```

**Environment Variables:**

- `MLFLOW_TRACKING_URI`: MLflow server endpoint
- `MLFLOW_MODEL_NAME`: Name of the registered model
- `MODEL_ALIAS`: Model version alias (default: production)

**API Access:**

- Server runs on port 9001
- Access API documentation at http://localhost:8000/docs

## Project Structure

```
├── app.py                      # Inference API
├── preprocess.py               # Data preprocessing utilities
├── schemas.py                  # API schemas
├── test_api.py                 # API tests
├── Dockerfile.training         # Training container
├── Dockerfile.serving          # Serving container
├── requirements.training.txt   # Training dependencies
├── requirements.serving.txt    # Serving dependencies
├── models/                     # ML models
│   ├── linear_regression.py
│   ├── random_forest.py
│   └── gradient_boosting.py
└── experiment_tracking/        # MLflow setup
    └── ml-flow/
```
