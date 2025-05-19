import pytest
import pandas as pd
from _preprocessing import load_data, preprocess_data


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
    df_processed, preprocessor = preprocess_data(df)

    # Check that output is a DataFrame and includes the target column
    assert isinstance(df_processed, pd.DataFrame)
    assert "is_fraud" in df_processed.columns

    # Ensure row counts match
    assert df_processed.shape[0] == len(df)

    from sklearn.compose import ColumnTransformer

    assert isinstance(preprocessor, ColumnTransformer)


def test_load_data_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_data("non_existent_file.csv")


def test_preprocess_missing_columns():
    # Missing required columns
    df = pd.DataFrame({"transaction_amount": [100], "location": ["US"]})
    with pytest.raises(ValueError):
        preprocess_data(df)
