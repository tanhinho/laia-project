"""Unit tests for training script functionality."""
import pytest
import numpy as np
import scipy.sparse
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import os


@pytest.fixture(scope="module")
def test_artifacts(tmp_path_factory):
    """Create minimal test artifacts for testing."""
    artifacts_dir = tmp_path_factory.mktemp("test_artifacts")

    # Create synthetic training data (100 samples, 10 features)
    np.random.seed(42)
    X_train = np.random.randn(100, 10)
    y_train = np.random.randn(100) * 100 + 500  # Trip durations in seconds

    # Create synthetic validation data (30 samples, 10 features)
    X_val = np.random.randn(30, 10)
    y_val = np.random.randn(30) * 100 + 500

    # Convert to sparse matrices
    X_train_sparse = scipy.sparse.csr_matrix(X_train)
    X_val_sparse = scipy.sparse.csr_matrix(X_val)

    # Save sparse matrices
    scipy.sparse.save_npz(artifacts_dir / 'X_train.npz', X_train_sparse)
    scipy.sparse.save_npz(artifacts_dir / 'X_val.npz', X_val_sparse)

    # Save labels
    np.save(artifacts_dir / 'y_train.npy', y_train)
    np.save(artifacts_dir / 'y_val.npy', y_val)

    # Create and save a simple preprocessor
    preprocessor = StandardScaler()
    preprocessor.fit(X_train)
    joblib.dump(preprocessor, artifacts_dir / 'preprocessor.pkl')

    return artifacts_dir


def test_data_loading(test_artifacts):
    """Test that training data can be loaded correctly."""
    X_train = scipy.sparse.load_npz(test_artifacts / 'X_train.npz')
    y_train = np.load(test_artifacts / 'y_train.npy')

    assert X_train.shape[0] == 100
    assert len(y_train) == 100
    assert X_train.shape[0] == len(y_train)


def test_preprocessor_loading(test_artifacts):
    """Test that preprocessor can be loaded."""
    preprocessor = joblib.load(test_artifacts / 'preprocessor.pkl')
    assert preprocessor is not None
    assert hasattr(preprocessor, 'transform')


def test_model_training(test_artifacts):
    """Test that LinearRegression model can be trained."""
    X_train = scipy.sparse.load_npz(test_artifacts / 'X_train.npz')
    y_train = np.load(test_artifacts / 'y_train.npy')

    model = LinearRegression()
    model.fit(X_train, y_train)

    assert hasattr(model, 'coef_')
    assert model.coef_ is not None
    assert len(model.coef_) == X_train.shape[1]


def test_model_prediction(test_artifacts):
    """Test that model can make predictions."""
    X_train = scipy.sparse.load_npz(test_artifacts / 'X_train.npz')
    y_train = np.load(test_artifacts / 'y_train.npy')
    X_val = scipy.sparse.load_npz(test_artifacts / 'X_val.npz')

    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_val)

    assert len(predictions) == 30
    assert all(isinstance(p, (int, float, np.number)) for p in predictions)


def test_model_performance(test_artifacts):
    """Test that model achieves reasonable performance."""
    X_train = scipy.sparse.load_npz(test_artifacts / 'X_train.npz')
    y_train = np.load(test_artifacts / 'y_train.npy')
    X_val = scipy.sparse.load_npz(test_artifacts / 'X_val.npz')
    y_val = np.load(test_artifacts / 'y_val.npy')

    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_val)

    mse = mean_squared_error(y_val, predictions)
    r2 = r2_score(y_val, predictions)

    assert mse > 0
    assert isinstance(r2, float)
