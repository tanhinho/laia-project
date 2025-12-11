"""Pytest configuration and shared fixtures."""
import os
import pytest
from unittest.mock import Mock, patch

# Set required environment variables before importing the app
os.environ.setdefault('MLFLOW_TRACKING_URI', 'http://localhost:5000')
os.environ.setdefault('MLFLOW_MODEL_NAME', 'test_model')
os.environ.setdefault('MODEL_ALIAS', 'test_alias')


@pytest.fixture(autouse=True)
def mock_startup_event(request):
    """Mock the startup event to prevent actual model loading during unit tests.

    Skip this fixture for e2e tests as they need the real service.
    """
    # Skip mocking for e2e tests
    if 'test_e2e' in request.node.fspath.basename:
        yield
    else:
        with patch('serving.app.load_artifacts'):
            yield
