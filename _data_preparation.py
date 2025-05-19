"""Data preparation module for fraud detection."""
import numpy as np
import pandas as pd
import os


def generate_dummy_fraud_data(
    num_samples: int = 10000, fraud_ratio: float = 0.05, random_state: int = 42
) -> pd.DataFrame:
    """
    Generate a synthetic dataset for fraud detection.

    The dataset simulates transaction features with a controlled ratio
    of fraudulent transactions.

    Args:
        num_samples (int): Number of samples to generate. Defaults to 10,000.
        fraud_ratio (float): Proportion of fraudulent transactions in the data.
                             Should be between 0 and 1. Defaults to 0.05.
        random_state (int): Random seed for reproducibility. Defaults to 42.

    Returns:
        pd.DataFrame: DataFrame containing the simulated transactions with
                      columns:
                      - transaction_amount (float): Amount of the transaction.
                      - transaction_time (int): Hour of the transaction (0-23).
                      - user_id (int): Simulated user ID.
                      - location (str): Geographical region of the transaction.
                      - is_fraud (int): Binary label, 1 for fraud, 0 otherwise.
    """
    # Input validation
    if not isinstance(num_samples, int) or num_samples <= 0:
        raise ValueError("num_samples must be a positive integer")
    if not isinstance(fraud_ratio, (int, float)) or not 0 <= fraud_ratio <= 1:
        raise ValueError("fraud_ratio must be between 0 and 1")
    if not isinstance(random_state, int):
        raise ValueError("random_state must be an integer")

    np.random.seed(random_state)

    transaction_amount = np.random.exponential(scale=100, size=num_samples)
    transaction_time = np.random.randint(0, 24, size=num_samples)
    user_id = np.random.randint(1000, 2000, size=num_samples)
    location = np.random.choice(
        ["US", "EU", "ASIA", "OTHER"], size=num_samples, p=[0.5, 0.2, 0.2, 0.1]
    )
    is_fraud = np.random.choice(
        [0, 1], size=num_samples, p=[1 - fraud_ratio, fraud_ratio]
    )

    dataframe = pd.DataFrame(
        {
            "transaction_amount": transaction_amount,
            "transaction_time": transaction_time,
            "user_id": user_id,
            "location": location,
            "is_fraud": is_fraud,
        }
    )

    return dataframe


def main():
    """
    Main function to generate dummy data and save it to CSV.
    """
    data_frame = generate_dummy_fraud_data()

    # Ensure data directory exists
    os.makedirs("./data", exist_ok=True)

    # Save to CSV
    data_frame.to_csv("./data/fraud_data.csv", index=False)
    print("✅Base Dummy Data generated and saved to ./data/fraud_data.csv")


if __name__ == "__main__":
    main()
