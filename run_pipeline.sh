#!/bin/bash
conda remove --name fraud_detection_env --all

conda env create --file environment.yml

conda activate fraud_detection_env

# Step 3: Install pre-commit hooks
pre-commit install

# Step 4: Run tests with coverage
pytest --cov=src --cov-report=term

# Step 5: Run pipeline scripts
python src/_data_preparation.py
python src/_preprocessing.py
python src/_model_training.py
