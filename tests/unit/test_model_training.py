"""
Unit tests for the model training module.
"""

import os
import tempfile
import pandas as pd
import pytest
import joblib
import numpy as np
from xgboost import XGBClassifier
from unittest import mock

from _model_training import train_model, save_model, predict_model, main


@pytest.fixture()
def sample_data() -> pd.DataFrame:
    """Returns a simple sample dataset for testing."""
    return pd.DataFrame(
        {
            "feature1": [1, 2, 3, 4],
            "feature2": [0.5, 0.3, 0.8, 0.9],
            "is_fraud": [0, 1, 0, 1],
        }
    )


def test_train_model(sample_data: pd.DataFrame) -> None:
    """Test training the XGBoost model."""
    model = train_model(sample_data, target_column="is_fraud")
    assert isinstance(model, XGBClassifier)


def test_save_and_load_model(sample_data: pd.DataFrame) -> None:
    """Test saving and loading the trained model."""
    model = train_model(sample_data, target_column="is_fraud")

    with tempfile.TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "model")  # no extension
        save_model(model, path)  # function will append .pkl
        expected_path = path + ".pkl"

        assert os.path.exists(expected_path)

        # Confirm file content is a valid model
        loaded_model = joblib.load(expected_path)
        assert isinstance(loaded_model, XGBClassifier)


@mock.patch("_model_training.save_model")
@mock.patch("_model_training.train_model")
@mock.patch("_model_training.pd.read_csv")
@mock.patch("_model_training.os.path.exists", return_value=True)  # Changed to True
def test_main(mock_exists, mock_read_csv, mock_train_model, mock_save_model):
    """Test the main function with mocked I/O and model flow."""

    # Step 1: Setup fake DataFrame for read_csv
    mock_read_csv.return_value = pd.DataFrame(
        {"feature1": [1, 2], "feature2": [0.1, 0.2], "is_fraud": [0, 1]}
    )

    # Step 2: Setup fake model return
    mock_train_model.return_value = XGBClassifier()

    # Step 3: Call main
    main()

    # Step 4: Assertions
    mock_exists.assert_called_once()
    mock_read_csv.assert_called_once()
    mock_train_model.assert_called_once()
    mock_save_model.assert_called_once()


def test_predict_model_with_mock_model():
    """Test predict_model with a mock logistic regression classifier."""
    # Step 1: Create dummy training data
    X_train = pd.DataFrame(
        {
            "feature1": [0, 1, 0, 1],
            "feature2": [1, 1, 0, 0],
        }
    )
    y_train = [0, 1, 0, 1]

    # Step 2: Train a simple logistic regression model
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Step 3: Predict on new dummy data
    X_test = pd.DataFrame(
        {
            "feature1": [0.5, 0.2],
            "feature2": [0.8, 0.1],
        }
    )

    predictions = predict_model(model, X_test)

    # Step 4: Assertions
    assert predictions is not None
    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == len(X_test)
    assert np.isin(predictions, [0, 1]).all(), "Predictions should be binary"
