"""
Unit tests for the model training module.
"""

import os
import tempfile
import pandas as pd
import pytest
import joblib
from xgboost import XGBClassifier
from _model_training import *
from unittest import mock


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
        path = os.path.join(temp_dir, "model.joblib")
        save_model(model, path)
        assert os.path.exists(path)

        loaded_model = joblib.load(path)
        assert isinstance(loaded_model, XGBClassifier)


@mock.patch("os.path.exists", return_value=True)
@mock.patch("pandas.read_csv")
@mock.patch("_model_training.train_model")
@mock.patch("_model_training.save_model")
def test_main(mock_save, mock_train, mock_read_csv, mock_exists):
    # Setup fake DataFrame
    mock_read_csv.return_value = mock.Mock()
    mock_train.return_value = mock.Mock()

    main()

    mock_exists.assert_called_once()
    mock_read_csv.assert_called_once()
    mock_train.assert_called_once()
    mock_save.assert_called_once()
