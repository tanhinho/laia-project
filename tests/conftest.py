import os
import pytest
from unittest.mock import Mock, patch

os.environ.setdefault('MLFLOW_TRACKING_URI', 'http://localhost:5000')
os.environ.setdefault('MLFLOW_MODEL_NAME', 'test_model')
os.environ.setdefault('MODEL_ALIAS', 'test_alias')


@pytest.fixture(autouse=True)
def mock_startup_event():
    #Mock the startup event to prevent actual model loading during tests
    with patch('serving.app.load_artifacts'):
        yield
