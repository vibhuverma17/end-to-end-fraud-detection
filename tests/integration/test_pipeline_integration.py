# tests/integration/test_pipeline_integration.py

import pytest
import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src"))
)

import _data_preparation, _preprocessing, _model_training


def test_data_pipeline_to_model_training():
    # Step 1: Load raw data
    raw_data = _data_preparation.generate_dummy_fraud_data()

    # Step 2: Preprocess it
    processed_data, _ = _preprocessing.preprocess_data(raw_data)

    # Step 3: Train the model
    model = _model_training.train_model(processed_data, target_column="is_fraud")

    # Step 4: Assert integration-level expectations
    assert model is not None, "Model should not be None after training"
    assert hasattr(model, "predict"), "Model should have a predict method"
