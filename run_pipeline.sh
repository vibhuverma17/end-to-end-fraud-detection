#!/bin/bash
conda remove --name fraud_detection_env --all

conda env create --file environment.yml

conda activate fraud_detection_env


pre-commit install

pytest --cov=. --cov-report=term

python _data_preparation.py
python _preprocessing.py
python _model_training.py
