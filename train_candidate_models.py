from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import statsmodels.api as sm
import lightgbm as lgb
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score

TRAIN_FILE = Path("traindata.csv")
VALIDATION_FILE = Path("valdata.csv")
INPUT_ENCODING = "GBK"  # Use "utf-8-sig" if your CSVs were exported in UTF-8.
OUTPUT_DIR = Path("model_outputs")
FEATURES = ["PCT", "CD3.CD4", "SFTSV_RNA", "IL_6"]
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_cohort(path):
    frame = pd.read_csv(path, encoding=INPUT_ENCODING)
    required = FEATURES + ["Outcome"]
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}; match your actual input names.")
    for name in required:
        frame[name] = pd.to_numeric(frame[name], errors="raise")
    if not np.isfinite(frame[required].to_numpy(dtype=float)).all():
        raise ValueError(f"{path}: missing or nonfinite values remain; preprocess before fitting.")
    if set(frame["Outcome"].unique()) != {0, 1}:
        raise ValueError(f"{path}: Outcome must include both 0 (survival) and 1 (in-hospital death).")
    frame["Outcome"] = frame["Outcome"].astype(int)
    return frame


# Logistic regression (statsmodels; unpenalized, as in the original source).
train_data = load_cohort(TRAIN_FILE)
X_train = train_data[['PCT', 'CD3.CD4', 'SFTSV_RNA', 'IL_6']]
X_train_const = sm.add_constant(X_train, has_constant="add")
Y_train = train_data['Outcome']
logist_model = sm.Logit(Y_train, X_train_const).fit(disp=0)
print(logist_model.summary())
Y_train_logist = logist_model.predict(X_train_const)
auc_logist = roc_auc_score(Y_train, Y_train_logist)
auc_logist
coef = logist_model.params
formula_logit = (
    f"Logit(P) = {coef['const']:.6f} "
    f"{coef['PCT']:+.6f} × PCT "
    f"{coef['CD3.CD4']:+.6f} × CD3.CD4 "
    f"{coef['SFTSV_RNA']:+.6f} × SFTSV_RNA "
    f"{coef['IL_6']:+.6f} × IL_6"
)
print(formula_logit)
print("\nP = 1 / [1 + exp(-Logit(P))]")
print(
    "P(in-hospital mortality) = 1 / {1 + exp[-("
    f"{coef['const']:.6f} "
    f"{coef['PCT']:+.6f} × PCT "
    f"{coef['CD3.CD4']:+.6f} × CD3.CD4 "
    f"{coef['SFTSV_RNA']:+.6f} × SFTSV_RNA "
    f"{coef['IL_6']:+.6f} × IL_6"
    ")]}"
)
result_table = pd.DataFrame({
    "Variable": logist_model.params.index,
    "Beta": logist_model.params.values,
    "SE": logist_model.bse.values,
    "OR": np.exp(logist_model.params.values),
    "OR_95CI_Lower": np.exp(logist_model.conf_int()[0].values),
    "OR_95CI_Upper": np.exp(logist_model.conf_int()[1].values),
    "P_value": logist_model.pvalues.values
})
print(result_table.round(4))
result_table.to_csv(
    OUTPUT_DIR / "Logistic_regression_equation_and_coefficients.csv",
    index=False,
    encoding="utf-8-sig"
)
with open(OUTPUT_DIR / "logistic_model.pkl", 'wb') as f:
    pickle.dump(logist_model, f)
train_data_ml = train_data
val_data_ml = load_cohort(VALIDATION_FILE)
X_train = train_data_ml[['PCT', 'CD3.CD4', 'SFTSV_RNA', 'IL_6']]
y_train = train_data_ml['Outcome']
X_val = val_data_ml[['PCT', 'CD3.CD4', 'SFTSV_RNA', 'IL_6']]
y_val = val_data_ml['Outcome']

    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 10],
    'n_estimators': [50, 100, 200],
    'subsample': [0.6, 0.8, 1.0]
}
grid_search_xgb = GridSearchCV(
    estimator=xgb_default,
    param_grid=param_grid,
    scoring='roc_auc',
    cv=5,
    n_jobs=-1,
    verbose=1
)
grid_search_xgb.fit(X_train, y_train)
best_auc_xgb = grid_search_xgb.best_score_
xgb_model_best = grid_search_xgb.best_estimator_
best_params_xgb = grid_search_xgb.best_params_
print("Selected XGBoost parameters:", best_params_xgb)
print("Tuned XGBoost parameters:", pd.DataFrame.from_dict(xgb_model_best.get_params(), orient='index'))
print("Default XGBoost validation AUC:", auc_xgbd)
print("Best XGBoost mean training CV AUC:", best_auc_xgb)
with open(OUTPUT_DIR / "xgb_model.pkl", 'wb') as f:
    pickle.dump(xgb_model_best, f)

lgb_default = lgb.LGBMClassifier(random_state=123)
lgb_default.fit(X_train, y_train)
y_val_pred_prob_lgbd = lgb_default.predict_proba(X_val)[:, 1]
auc_lgbd = roc_auc_score(y_val, y_val_pred_prob_lgbd)
print("Default LightGBM validation AUC:", auc_lgbd)
param_grid = {
    'learning_rate': [0.1, 0.2],
    'num_leaves': [31, 50, 100],
    'n_estimators': [100],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}
grid_search_lgb = GridSearchCV(
    estimator=lgb_default,
    param_grid=param_grid,
    scoring='roc_auc',
    cv=5,
    n_jobs=-1,
    verbose=1
)
grid_search_lgb.fit(X_train, y_train)
best_auc_lgb = grid_search_lgb.best_score_
lgb_model_best = grid_search_lgb.best_estimator_
best_params_lgb = grid_search_lgb.best_params_
print("Selected LightGBM parameters:", best_params_lgb)
print("Tuned LightGBM parameters:", pd.DataFrame.from_dict(lgb_model_best.get_params(), orient='index'))
print("Default LightGBM validation AUC:", auc_lgbd)
print("Best LightGBM mean training CV AUC:", best_auc_lgb)
with open(OUTPUT_DIR / "lgb_model.pkl", 'wb') as f:
    pickle.dump(lgb_model_best, f)

svm_default = SVC(probability=True, random_state=123)
svm_default.fit(X_train, y_train)
y_val_pred_prob_svmd = svm_default.predict_proba(X_val)[:, 1]
auc_svmd = roc_auc_score(y_val, y_val_pred_prob_svmd)
print("Default SVM validation AUC:", auc_svmd)
param_grid = {
    'C': [0.1],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto', 0.01],
    'degree': [2, 3, 4]
}
grid_search_svm = GridSearchCV(
    estimator=svm_default,
    param_grid=param_grid,
    scoring='roc_auc',
    cv=5,
    n_jobs=-1,
    verbose=1
)
grid_search_svm.fit(X_train, y_train)
best_auc_svm = grid_search_svm.best_score_
svm_model_best = grid_search_svm.best_estimator_
best_params_svm = grid_search_svm.best_params_
print("Selected SVM parameters:", best_params_svm)
print("Tuned SVM parameters:", pd.DataFrame.from_dict(svm_model_best.get_params(), orient='index'))
print("Default SVM validation AUC:", auc_svmd)
print("Best SVM mean training CV AUC:", best_auc_svm)
with open(OUTPUT_DIR / "svm_model.pkl", 'wb') as f:
    pickle.dump(svm_model_best, f)

ann_default = MLPClassifier(random_state=123, max_iter=500)
ann_default.fit(X_train, y_train)
y_val_pred_prob_annd = ann_default.predict_proba(X_val)[:, 1]
auc_annd = roc_auc_score(y_val, y_val_pred_prob_annd)
print("Default ANN validation AUC:", auc_annd)
param_grid = {
    'hidden_layer_sizes': [(25,), (50,), (100,), (10, 10), (50, 50), (100, 100), (50, 50, 50)],
    'activation': ['relu', 'tanh', 'logistic']
}
grid_search_ann = GridSearchCV(
    estimator=ann_default,
    param_grid=param_grid,
    scoring='roc_auc',
    cv=5,
    n_jobs=-1,
    verbose=1
)
grid_search_ann.fit(X_train, y_train)
best_auc_ann = grid_search_ann.best_score_
ann_model_best = grid_search_ann.best_estimator_
best_params_ann = grid_search_ann.best_params_
print("Selected ANN parameters:", best_params_ann)
print("Tuned ANN parameters:", pd.DataFrame.from_dict(ann_model_best.get_params(), orient='index'))
print("Default ANN validation AUC:", auc_annd)
print("Best ANN mean training CV AUC:", best_auc_ann)
with open(OUTPUT_DIR / "ann_model.pkl", 'wb') as f:
    pickle.dump(ann_model_best, f)

searches = {
    "Decision tree": grid_search_tree, "Random forest": grid_search_rf,
    "XGBoost": grid_search_xgb, "LightGBM": grid_search_lgb,
    "SVM": grid_search_svm, "ANN": grid_search_ann,
}
pd.DataFrame([
    {"Model": name, "Mean_CV_AUC": search.best_score_,
     "Selected_parameters": repr(search.best_params_)}
    for name, search in searches.items()
]).to_csv(OUTPUT_DIR / "selected_hyperparameters.csv", index=False, encoding="utf-8-sig")
for name, search in searches.items():
    pd.DataFrame(search.cv_results_).to_csv(
        OUTPUT_DIR / (name.replace(" ", "_") + "_grid_results.csv"), index=False
    )
(OUTPUT_DIR / "logistic_equation.txt").write_text
