## End-to-End Fraud Detection Model Training Pipeline

This project implements a complete pipeline for training a machine learning model to detect fraudulent transactions. It includes data preparation, preprocessing, model training, testing, and automation for reproducibility and quality assurance.

## 🚀 Features

- Modular pipeline scripts for data prep, preprocessing, and model training.
- Pre-configured conda environment for reproducibility.
- Testing with `pytest` and coverage reporting.
- Code quality automation with `pre-commit` and `pylint`.
- SHAP-based model explainability (planned).
- Scalable deployment options (e.g., AWS SageMaker).

## ⚙️ Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/vibhuverma17/end-to-end-fraud-detection.git
   cd end-to-end-fraud-detection
   ```

2. **Create and activate the conda environment:**

   ```bash
   conda env create -f environment.yml
   conda activate fraud_detection_env
   ```

3. **Enable pre-commit hooks:**

   ```bash
   pre-commit install
   ```

4. **Run tests and check code coverage:**

   ```bash
   pytest --cov=. --cov-report=term
   ```

## 🚦 Run the Pipeline

You can run the entire training pipeline using the provided shell script:

```bash
bash run_pipeline.sh
```

This will:
- Recreate the environment (optional step in script)
- Install hooks
- Run tests
- Execute:
  - `_data_preparation.py`
  - `_preprocessing.py`
  - `_model_training.py`

## 🧪 Individual Steps (Optional)

If you want to run components manually:

- **Prepare data:**

  ```bash
  python _data_preparation.py
  ```

- **Preprocess data:**

  ```bash
  python _preprocessing.py
  ```

- **Train the model:**

  ```bash
  python _model_training.py
  ```

## 📂 Project Structure

```
end-to-end/
│
├── data/                     # Sample datasets
├── src/
│   ├── __init__.py
│   ├── _data_preparation.py
│   ├── _preprocessing.py
│   ├── _model_training.py
│
├── tests/
│   ├── unit/
│   │   ├── test_data_preparation.py
│   │   ├── test_preprocessing.py
│   │   ├── test_model_training.py
│   ├── integration/
│   │   └── test_pipeline_integration.py
│   ├── e2e/
│   │   └── test_full_workflow.py
│   ├── smoke/
│       └── test_smoke_pipeline.py
│
├── .github/workflows/        # GitHub Actions CI/CD workflows
│   └── test.yml
│
├── requirements.txt
├── pytest.ini
└── README.md
```

## 🛡️ License

MIT License
