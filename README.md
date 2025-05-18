# End-to-End Fraud Detection Model Deployment Workflow

This project is an end-to-end workflow for building, deploying, and monitoring a fraud detection model.

## 🚀 Features

* Data generation, preprocessing, and model training.
* Model explainability with SHAP.
* Experiment tracking with MLflow.
* Automated code quality checks with pre-commit (pylint, pytest).
* Scalable deployment options (AWS SageMaker, Docker).

## ⚡️ Setup

1. Clone the repo:

   ```bash
   git clone <your-repository-url>
   cd fraud-detection-project
   ```

2. Create and activate the conda environment:

   ```bash
   conda env create -f environment.yml
   conda activate fraud_detection_env
   ```

3. Enable pre-commit hooks:

   ```bash
   pre-commit install
   ```

4. Run tests and check coverage:

   ```bash
   pytest --cov=. --cov-report=term
   ```

## ✅ Usage

* Generate dummy data:

  ```bash
  python data_preparation.py
  ```

* Train and test the model:

  ```bash
  python train_model.py
  ```

* Deploy the model:

  * Locally with Flask/FastAPI.
  * On AWS SageMaker.

## 📂 Current Structure

* `data_preparation.py` - Data generation and preprocessing functions.
* `test_data_preparation.py` - Unit tests for data preparation.
* `environment.yml` - Conda environment configuration with all dependencies.
* `.pre-commit-config.yaml` - Pre-commit hooks configuration.

## 🛡️ License

MIT