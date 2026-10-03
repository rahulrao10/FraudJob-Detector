# Scam Job Checker (browser extension + ML backend)

Train models on the Kaggle "Fake Job Postings" CSV, serve the best one through Flask,
and check postings from a Chrome/Edge extension.

## Setup (Windows PowerShell)
    python -m venv venv
    venv\Scripts\Activate.ps1
    pip install -r requirements.txt

(macOS/Linux: `source venv/bin/activate`)

## Data
Put the Kaggle file at `data/fake_job_postings.csv`.

## Train (run from the project root)
    python -m src.train_classical     # Naive Bayes, Linear SVM, Logistic Regression
    python -m src.train_nn            # embedding neural network (PyTorch)
    python -m src.compare             # comparison table -> results/comparison.csv

## Run the API
    python backend/app.py             # http://127.0.0.1:5000/health

## Load the extension
1. Open chrome://extensions (or edge://extensions)
2. Turn on Developer mode
3. Load unpacked -> select the `extension` folder
4. Open a job posting, click the extension icon, press "Check this job posting"

## Layout
- src/        text cleaning, models, training, rules, prediction
- backend/    Flask API (/health, /analyze)
- extension/  Manifest V3 popup
- results/    metrics per model + comparison.csv
