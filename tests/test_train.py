import pytest
import numpy as np
import scipy.sparse
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os


ARTIFACTS_PATH = 'artifacts'


def test_artifacts_exist():
    """Test that required artifacts exist."""
    assert os.path.exists(ARTIFACTS_PATH)
    assert os.path.exists(os.path.join(ARTIFACTS_PATH, 'X_train.npz'))
    assert os.path.exists(os.path.join(ARTIFACTS_PATH, 'y_train.npy'))
    assert os.path.exists(os.path.join(ARTIFACTS_PATH, 'X_val.npz'))
    assert os.path.exists(os.path.join(ARTIFACTS_PATH, 'y_val.npy'))
    assert os.path.exists(os.path.join(ARTIFACTS_PATH, 'preprocessor.pkl'))


def test_data_loading():
    """Test that training data can be loaded correctly."""
    X_train = scipy.sparse.load_npz(
        os.path.join(ARTIFACTS_PATH, 'X_train.npz'))
    y_train = np.load(os.path.join(ARTIFACTS_PATH, 'y_train.npy'))

    assert X_train.shape[0] > 0
    assert len(y_train) > 0
    assert X_train.shape[0] == len(y_train)


def test_preprocessor_loading():
    """Test that preprocessor can be loaded."""
    preprocessor = joblib.load(os.path.join(
        ARTIFACTS_PATH, 'preprocessor.pkl'))
    assert preprocessor is not None
    assert hasattr(preprocessor, 'transform')


def test_model_training():
    """Test that LinearRegression model can be trained."""
    X_train = scipy.sparse.load_npz(
        os.path.join(ARTIFACTS_PATH, 'X_train.npz'))
    y_train = np.load(os.path.join(ARTIFACTS_PATH, 'y_train.npy'))

    model = LinearRegression()
    model.fit(X_train, y_train)

    assert hasattr(model, 'coef_')
    assert model.coef_ is not None
    assert len(model.coef_) == X_train.shape[1]


def test_model_prediction():
    """Test that model can make predictions."""
    X_train = scipy.sparse.load_npz(
        os.path.join(ARTIFACTS_PATH, 'X_train.npz'))
    y_train = np.load(os.path.join(ARTIFACTS_PATH, 'y_train.npy'))
    X_val = scipy.sparse.load_npz(os.path.join(ARTIFACTS_PATH, 'X_val.npz'))

    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_val)

    assert len(predictions) == X_val.shape[0]
    assert all(isinstance(p, (int, float, np.number)) for p in predictions)
    assert all(p > 0 for p in predictions)  # Trip durations should be positive


def test_model_performance():
    """Test that model achieves reasonable performance."""
    X_train = scipy.sparse.load_npz(
        os.path.join(ARTIFACTS_PATH, 'X_train.npz'))
    y_train = np.load(os.path.join(ARTIFACTS_PATH, 'y_train.npy'))
    X_val = scipy.sparse.load_npz(os.path.join(ARTIFACTS_PATH, 'X_val.npz'))
    y_val = np.load(os.path.join(ARTIFACTS_PATH, 'y_val.npy'))

    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_val)

    mse = mean_squared_error(y_val, predictions)
    r2 = r2_score(y_val, predictions)

    assert mse > 0
    assert r2 > 0  # Model should have some predictive power
