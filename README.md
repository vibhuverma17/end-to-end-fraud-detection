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
   git clone https://github.com/YOUR_USERNAME/end-to-end-fraud-detection.git
   cd end-to-end-fraud-detection
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Enable pre-commit hooks:

   ```bash
   pre-commit install
   ```

4. Run tests and check coverage:

   ```bash
   pytest --cov=./
   ```

## ✅ Usage

* Generate dummy data:

  ```bash
  python generate_dummy_data.py
  ```

* Train and test the model:

  ```bash
  python train_model.py
  ```

* Deploy the model:

  * Locally with Flask/FastAPI.
  * On AWS SageMaker.

## 📂 Directory Structure

* `data/` - Raw and processed data files.
* `scripts/` - Python scripts for data generation, training, and deployment.
* `tests/` - Unit and integration tests.

## 🛡️ License

MIT
