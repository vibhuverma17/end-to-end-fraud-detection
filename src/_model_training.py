"""
Model training module using XGBoost for fraud detection.
"""

import os
import joblib
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def load_data(data_path: str) -> pd.DataFrame:
    """Loads CSV data from the given path."""
    return pd.read_csv(data_path)


def train_model(data: pd.DataFrame, target_column: str) -> xgb.XGBClassifier:
    """Trains an XGBoost model."""
    X = data.drop(columns=[target_column])
    y = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = xgb.XGBClassifier(eval_metric="logloss")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(
        "Classification Report:\n",
        classification_report(y_test, y_pred, zero_division=0),
    )

    return model


def save_model(model: xgb.XGBClassifier, model_path: str) -> None:
    """Save the trained model to a .pkl file with a clean extension."""
    # Remove common double extensions
    base, ext = os.path.splitext(model_path)
    if ext not in ["", ".pkl"]:
        model_path = base  # drop weird extensions
    model_path = model_path + ".pkl"

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"✅ Model saved to {model_path}")


def main() -> None:
    """Main function to run model training."""
    data_path = "data/processed/fraud_data.csv"
    model_output_path = "models/fraud_xgb_model.joblib"
    target_column = "is_fraud"

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = load_data(data_path)
    model = train_model(df, target_column)
    save_model(model, model_output_path)


if __name__ == "__main__":
    main()
