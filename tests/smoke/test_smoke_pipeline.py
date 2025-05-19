import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src"))
)


def test_smoke_pipeline_runs():
    from _data_preparation import generate_dummy_fraud_data
    from _preprocessing import preprocess_data
    from _model_training import train_model, predict_model

    # Just make sure the pipeline runs without crashing
    data = generate_dummy_fraud_data()
    processed, _ = preprocess_data(data)
    model = train_model(processed, target_column="is_fraud")

    features = processed.drop(columns=["is_fraud"])
    preds = predict_model(model, features)

    assert len(preds) == len(features)
