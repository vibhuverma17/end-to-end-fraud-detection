# tests/e2e/test_full_workflow.py

import os
import sys
import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src"))
)

import _data_preparation
import _preprocessing
import _model_training


def test_end_to_end_pipeline():
    # Step 1: Generate dummy fraud data
    raw_data = _data_preparation.generate_dummy_fraud_data()

    # Step 2: Preprocess the data
    processed_data, _ = _preprocessing.preprocess_data(raw_data)

    # Step 3: Train the model
    model = _model_training.train_model(processed_data, target_column="is_fraud")

    # Step 4: Prepare features (remove target column for prediction)
    features = processed_data.drop(columns=["is_fraud"])

    # Step 5: Predict using the trained model
    predictions = _model_training.predict_model(model, features)

    # Step 6: Validate predictions
    assert predictions is not None, "Predictions should not be None"
    assert len(predictions) == len(features), "Prediction count should match input"
    assert all(
        pred in [0, 1] for pred in predictions
    ), "Predictions must be binary (0 or 1)"
