"""Determine Youden cutoffs in training data and predict validation probabilities.

Run after train_candidate_models.py. Both CSVs must contain the same prepared
predictors used during training. No imputation, scaling, model refitting, or
validation-based threshold optimization is performed here.
"""

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_curve


TRAIN_FILE = Path("traindata.csv")
VALIDATION_FILE = Path("valdata.csv")
MODEL_DIR = Path("model_outputs")
OUTPUT_DIR = Path("evaluation_outputs")
ENCODING = "GBK"
FEATURES = ["PCT", "CD3.CD4", "SFTSV_RNA", "IL_6"]
MODEL_FILES = {
    "Logistic": "logistic_model.pkl",
    "DecisionTree": "tree_model.pkl",
    "RandomForest": "rf_model.pkl",
    "XGBoost": "xgb_model.pkl",
    "LightGBM": "lgb_model.pkl",
    "SVM": "svm_model.pkl",
    "ANN": "ann_model.pkl",
}


def load_cohort(path):
    """Read each cohort once and retain the original predictor order."""
    data = pd.read_csv(path, encoding=ENCODING)
    required = FEATURES + ["Outcome"]
    missing_columns = [column for column in required if column not in data.columns]
    if missing_columns:
        raise ValueError(f"{path}: missing columns {missing_columns}")
    numeric_data = data[required].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric_data.to_numpy(dtype=float)).all():
        raise ValueError(f"{path}: predictors and outcome must be complete and finite")
    if set(numeric_data["Outcome"].unique()) != {0, 1}:
        raise ValueError(f"{path}: Outcome must contain both 0 (survival) and 1 (death)")
    return numeric_data[FEATURES], numeric_data["Outcome"].astype(int)


def predict_death_probability(model_name, model, predictors):
    if model_name == "Logistic":
        design_matrix = sm.add_constant(predictors, has_constant="add")
        probabilities = model.predict(design_matrix)
    else:
        # Locate death explicitly rather than assuming it is the second class.
        death_columns = np.flatnonzero(np.asarray(model.classes_) == 1)
        if len(death_columns) != 1:
            raise ValueError(f"{model_name}: could not identify Outcome class 1")
        probabilities = model.predict_proba(predictors)[:, death_columns[0]]
    probabilities = np.asarray(probabilities, dtype=float).reshape(-1)
    if (
        len(probabilities) != len(predictors)
        or not np.isfinite(probabilities).all()
        or np.any((probabilities < 0) | (probabilities > 1))
    ):
        raise ValueError(f"{model_name}: invalid predicted probabilities")
    return probabilities


def find_optimal_threshold(y_true, predicted_probabilities):
    """Retain the original ROC/argmax rule, including its first-tie convention."""
    fpr, tpr, thresholds = roc_curve(y_true, predicted_probabilities)
    j_scores = tpr - fpr
    best_index = np.argmax(j_scores)
    return float(thresholds[best_index]), float(j_scores[best_index])


def main():
    X_train, y_train = load_cohort(TRAIN_FILE)
    X_val, y_val = load_cohort(VALIDATION_FILE)
    models = {}
    train_probabilities = {}
    validation_probabilities = {}
    thresholds = {}
    threshold_rows = []

    for name, filename in MODEL_FILES.items():
        with (MODEL_DIR / filename).open("rb") as model_file:
            models[name] = pickle.load(model_file)

        train_probabilities[name] = predict_death_probability(
            name, models[name], X_train
        )
        threshold, youden_j = find_optimal_threshold(
            y_train, train_probabilities[name]
        )
        thresholds[name] = threshold
        threshold_rows.append({
            "Model": name,
            "Training_Threshold": threshold,
            "Training_Youden_J": youden_j,
        })

        # Validation outcomes do not enter threshold estimation.
        validation_probabilities[name] = predict_death_probability(
            name, models[name], X_val
        )
        print(f"{name}: training cutoff = {threshold:.4f}, J = {youden_j:.4f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(threshold_rows).to_csv(
        OUTPUT_DIR / "training_youden_thresholds.csv",
        index=False,
        encoding="utf-8-sig",
    )
    # Full-precision thresholds and probabilities are available for subsequent
    # evaluation. This script does not calculate classification metrics.
    return models, thresholds, train_probabilities, validation_probabilities, y_train, y_val


if __name__ == "__main__":
    (
        models,
        thresholds,
        train_probabilities,
        validation_probabilities,
        y_train,
        y_val,
    ) = main()
