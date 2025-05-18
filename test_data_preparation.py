"""Tests for data preparation module."""
import pytest

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from _data_preparation import generate_dummy_fraud_data


def test_data_shape():
    """
    Test that the generated data has the correct number of samples and columns.
    """
    dataframe = generate_dummy_fraud_data(num_samples=1000)
    assert dataframe.shape[0] == 1000, "DataFrame should have 1000 rows."
    assert "is_fraud" in dataframe.columns, "'is_fraud' column must exist."


def test_fraud_ratio():
    """
    Test that the generated data has the expected fraud ratio.
    """
    dataframe = generate_dummy_fraud_data(num_samples=10000, fraud_ratio=0.1)
    actual_fraud_ratio = dataframe["is_fraud"].mean()
    lower_bound = 0.08
    upper_bound = 0.12
    assert (
        lower_bound <= actual_fraud_ratio <= upper_bound
    ), f"Fraud ratio {actual_fraud_ratio:.4f} is outside the expected range."


def test_column_names():
    """
    Test that all expected columns are present in the generated DataFrame.
    """
    dataframe = generate_dummy_fraud_data(num_samples=10)
    expected_columns = {
        "transaction_amount",
        "transaction_time",
        "user_id",
        "location",
        "is_fraud",
    }
    assert set(dataframe.columns) == expected_columns, "DataFrame columns mismatch."


def test_transaction_amount_positive():
    """
    Test that all transaction amounts are positive.
    """
    dataframe = generate_dummy_fraud_data(num_samples=100)
    assert (
        dataframe["transaction_amount"] > 0
    ).all(), "All transaction amounts should be positive."


def test_location_values():
    """
    Test that all location values are from the allowed set.
    """
    dataframe = generate_dummy_fraud_data(num_samples=100)
    allowed_locations = {"US", "EU", "ASIA", "OTHER"}
    assert set(dataframe["location"].unique()).issubset(
        allowed_locations
    ), "Unexpected location values found."


@pytest.mark.parametrize("fraud_ratio", [0.0, 1.0])
def test_extreme_fraud_ratios(fraud_ratio):
    """
    Test that extreme fraud ratios (0 and 1) produce correct labels.
    """
    dataframe = generate_dummy_fraud_data(num_samples=100, fraud_ratio=fraud_ratio)
    assert (
        dataframe["is_fraud"].nunique() == 1
    ), "All labels should be the same for extreme fraud ratios."
    assert dataframe["is_fraud"].iloc[0] == int(
        fraud_ratio
    ), "Label does not match fraud_ratio."


def test_random_state_reproducibility():
    """
    Test that using the same random_state produces identical data.
    """
    df1 = generate_dummy_fraud_data(num_samples=100, random_state=123)
    df2 = generate_dummy_fraud_data(num_samples=100, random_state=123)
    assert df1.equals(df2), "DataFrames with the same random_state should be identical."


def test_input_validation():
    """
    Test that input validation works correctly.
    """
    # Test invalid num_samples
    with pytest.raises(ValueError, match="num_samples must be a positive integer"):
        generate_dummy_fraud_data(num_samples=-1)

    with pytest.raises(ValueError, match="num_samples must be a positive integer"):
        generate_dummy_fraud_data(num_samples=0)

    # Test invalid fraud_ratio
    with pytest.raises(ValueError, match="fraud_ratio must be between 0 and 1"):
        generate_dummy_fraud_data(fraud_ratio=-0.1)

    with pytest.raises(ValueError, match="fraud_ratio must be between 0 and 1"):
        generate_dummy_fraud_data(fraud_ratio=1.1)

    # Test invalid random_state
    with pytest.raises(ValueError, match="random_state must be an integer"):
        generate_dummy_fraud_data(random_state="not_an_int")
