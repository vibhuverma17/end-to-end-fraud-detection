import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_data(csv_path: str) -> pd.DataFrame:
    """Load the fraud data CSV."""
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"CSV file not found at path: {csv_path}") from e

    if df.empty:
        raise ValueError("Loaded data is empty.")
    return df


def preprocess_data(df: pd.DataFrame):
    """
    Preprocess the fraud dataset.

    Args:
        df (pd.DataFrame): Raw fraud dataset.

    Returns:
        X_processed (np.ndarray or sparse matrix): Preprocessed features.
        y (pd.Series): Target labels.
        preprocessor (ColumnTransformer): Fitted preprocessor object.
    """
    required_cols = {
        "transaction_amount",
        "transaction_time",
        "user_id",
        "location",
        "is_fraud",
    }
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Missing required columns in data: {missing}")

    df = df.dropna(subset=["is_fraud"])
    X = df.drop(columns=["is_fraud"])
    y = df["is_fraud"]

    numeric_features = ["transaction_amount", "transaction_time", "user_id"]
    categorical_features = ["location"]

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    X_processed = preprocessor.fit_transform(X)

    # Combine processed features and target into a single DataFrame
    feature_names = numeric_features + list(
        preprocessor.named_transformers_["cat"].get_feature_names_out(
            categorical_features
        )
    )
    X_processed_df = pd.DataFrame(
        X_processed.toarray() if hasattr(X_processed, "toarray") else X_processed,
        columns=feature_names,
        index=X.index,
    )
    # Add the target column to the DataFrame
    X_processed_df["is_fraud"] = y.values
    return X_processed_df, preprocessor


if __name__ == "__main__":
    import os

    df_raw = load_data("data/fraud_data.csv")
    df_processed, preprocessor = preprocess_data(df_raw)

    os.makedirs("data/processed", exist_ok=True)
    df_processed.to_csv("data/processed/fraud_data.csv", index=False)
    print("✅ Preprocessed data saved to data/processed/fraud_data.csv")
