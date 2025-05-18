import pytest
import pandas as pd
from _preprocessing import load_data, preprocess_data, split_data


@pytest.fixture
def tmp_csv(tmp_path):
    # Create a small dummy DataFrame similar to your real data
    df = pd.DataFrame(
        {
            "transaction_amount": [100.0, 150.5, 200.0],
            "transaction_time": [10, 15, 20],
            "user_id": [1001, 1002, 1003],
            "location": ["US", "EU", "ASIA"],
            "is_fraud": [0, 1, 0],
        }
    )
    file_path = tmp_path / "temp_fraud_data.csv"
    df.to_csv(file_path, index=False)
    return file_path


def test_load_data(tmp_csv):
    df = load_data(str(tmp_csv))
    assert isinstance(df, pd.DataFrame)
    assert "is_fraud" in df.columns


def test_preprocess_data(tmp_csv):
    df = load_data(str(tmp_csv))
    X_processed, y, preprocessor = preprocess_data(df)
    assert X_processed.shape[0] == len(df)
    assert len(y) == len(df)
    from sklearn.compose import ColumnTransformer

    assert isinstance(preprocessor, ColumnTransformer)


def test_split_data(tmp_csv):
    df = load_data(str(tmp_csv))
    X_processed, y, _ = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X_processed, y)
    assert len(X_train) > 0 and len(X_test) > 0
    assert len(y_train) > 0 and len(y_test) > 0


def test_load_data_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_data("non_existent_file.csv")


def test_preprocess_missing_columns():
    # Missing required columns
    df = pd.DataFrame({"transaction_amount": [100], "location": ["US"]})
    with pytest.raises(ValueError):
        preprocess_data(df)
