"""SFTS candidate-model fitting and training-only hyperparameter tuning.

Cleaned from the supplied source excerpt; original grids and model settings
are retained. This script expects already prepared training/validation CSVs.
No imputation, feature selection, scaling or additional log transformation is
performed here. Ensure the inputs use the exact scales used for the manuscript.

cv=5 preserves GridSearchCV's original stratified, unshuffled classifier folds.
Default-model validation AUCs are descriptive; they are not tuning criteria.
This script does not automatically choose an algorithm based on validation AUC.
It does not include threshold selection or the full performance analysis.

Source provenance: the supplied excerpt contains references to a third-party
teaching-code source (派大珍). Promotional contact details have been removed.
Confirm rights to redistribute adapted third-party code before public upload.

Generated CSVs and pickle objects may contain or retain patient-level data.
Keep generated outputs out of the public repository. Do not load untrusted pickle files.
"""

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
# Decision tree
tree_default = DecisionTreeClassifier(random_state=123)
tree_default.fit(X_train, y_train)
print("决策树模型默认参数:", pd.DataFrame.from_dict(tree_default.get_params(), orient='index'))
y_val_pred_prob_treed = tree_default.predict_proba(X_val)[:, 1]
auc_treed = roc_auc_score(y_val, y_val_pred_prob_treed)
print("默认参数决策树模型的验证集 AUC:", auc_treed)
param_grid = {
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [5, 10, 20],
    'max_features': ['sqrt', None],
    'ccp_alpha': [0.0, 0.01, 0.1]
}
grid_search_tree = GridSearchCV(
    estimator=tree_default,
    param_grid=param_grid,
    scoring='roc_auc',
    cv=5,
    n_jobs=1,
    verbose=1
)
grid_search_tree.fit(X_train, y_train)
best_auc_tree = grid_search_tree.best_score_
tree_model_best = grid_search_tree.best_estimator_
best_params = grid_search_tree.best_params_
print("最佳决策树模型参数组合:", best_params)
print("调优后决策树模型详细参数:", pd.DataFrame.from_dict(tree_model_best.get_params(), orient='index'))
print("参数调优决策树模型最佳模型的交叉验证 AUC:", best_auc_tree)
print("默认参数决策树模型的验证集 AUC:", auc_treed)
with open(OUTPUT_DIR / "tree_model.pkl", 'wb') as f:
    pickle.dump(tree_model_best, f)
# Random forest
rf_model_default = RandomForestClassifier(random_state=123, oob_score=True)
rf_model_default.fit(X_train, y_train)
print("模型默认参数:", pd.DataFrame.from_dict(rf_model_default.get_params(), orient='index'))
y_val_pred_prob_rfd = rf_model_default.predict_proba(X_val)[:, 1]
auc_rfd = roc_auc_score(y_val, y_val_pred_prob_rfd)
print("默认参数模型的验证集 AUC:", auc_rfd)
param_grid = {
    'n_estimators': np.arange(50, 500, 50),
    'max_features': list(range(2, round(np.sqrt(X_train.shape[1])) + 1))
}
grid_search_rf = GridSearchCV(
    estimator=rf_model_default,
    param_grid=param_grid,
    scoring='roc_auc',
    cv=5,
    n_jobs=-1,
    verbose=1
)
grid_search_rf.fit(X_train, y_train)
print("训练集特征数:", X_train.shape[1])
for name, values in param_grid.items():
    print(f"{name} 候选值: {list(values)}")
best_auc_rf = grid_search_rf.best_score_
rf_model_best = grid_search_rf.best_estimator_
best_params_rf = grid_search_rf.best_params_
best_ntree = best_params_rf['n_estimators']
best_mtry = best_params_rf['max_features']
print("最佳RF模型参数组合: n_estimators =", best_ntree, ", max_features =", best_mtry)
print("调优后RF模型详细参数:", pd.DataFrame.from_dict(rf_model_best.get_params(), orient='index'))
print("默认参数RF模型的验证集 AUC:", auc_rfd)
print("调优后RF模型最佳模型的交叉验证 AUC", best_auc_rf)
with open(OUTPUT_DIR / "rf_model.pkl", 'wb') as f:
    pickle.dump(rf_model_best, f)
# XGBoost
xgb_default = XGBClassifier(random_state=123, eval_metric='logloss')
xgb_default.fit(X_train, y_train)
y_val_pred_prob_xgbd = xgb_default.predict_proba(X_val)[:, 1]
auc_xgbd = roc_auc_score(y_val, y_val_pred_prob_xgbd)
print("默认参数XGBoost模型的验证集 AUC:", auc_xgbd)
param_grid = {
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
print("最佳XGBoost参数组合:", best_params_xgb)
print("调优后XGBoost模型详细参数:", pd.DataFrame.from_dict(xgb_model_best.get_params(), orient='index'))
print("默认参数XGBoost模型的验证集 AUC:", auc_xgbd)
print("参数调优XGBoost模型最佳模型的交叉验证 AUC:", best_auc_xgb)
with open(OUTPUT_DIR / "xgb_model.pkl", 'wb') as f:
    pickle.dump(xgb_model_best, f)
# subsample_freq remains at its original default (0); subsample does not enable row sampling.
# LightGBM
lgb_default = lgb.LGBMClassifier(random_state=123)
lgb_default.fit(X_train, y_train)
y_val_pred_prob_lgbd = lgb_default.predict_proba(X_val)[:, 1]
auc_lgbd = roc_auc_score(y_val, y_val_pred_prob_lgbd)
print("默认参数LightGBM模型的验证集 AUC:", auc_lgbd)
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
print("最佳LightGBM参数组合:", best_params_lgb)
print("调优LightGBM模型详细参数:", pd.DataFrame.from_dict(lgb_model_best.get_params(), orient='index'))
print("默认参数LightGBM模型的验证集 AUC:", auc_lgbd)
print("参数调优LightGBM模型最佳模型的交叉验证 AUC:", best_auc_lgb)
with open(OUTPUT_DIR / "lgb_model.pkl", 'wb') as f:
    pickle.dump(lgb_model_best, f)
# SVM
svm_default = SVC(probability=True, random_state=123)
svm_default.fit(X_train, y_train)
y_val_pred_prob_svmd = svm_default.predict_proba(X_val)[:, 1]
auc_svmd = roc_auc_score(y_val, y_val_pred_prob_svmd)
print("默认参数SVM模型的验证集 AUC:", auc_svmd)
param_grid = {
    'C': [0.1],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto', 0.01],
    'degree': [2, 3, 4]  # Retained original grid; ignored without kernel='poly'.
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
print("最佳SVM参数组合:", best_params_svm)
print("调优SVM模型详细参数:", pd.DataFrame.from_dict(svm_model_best.get_params(), orient='index'))
print("默认参数SVM模型的验证集 AUC:", auc_svmd)
print("参数调优SVM模型最佳模型的交叉验证 AUC:", best_auc_svm)
with open(OUTPUT_DIR / "svm_model.pkl", 'wb') as f:
    pickle.dump(svm_model_best, f)
# ANN
ann_default = MLPClassifier(random_state=123, max_iter=500)
ann_default.fit(X_train, y_train)
y_val_pred_prob_annd = ann_default.predict_proba(X_val)[:, 1]
auc_annd = roc_auc_score(y_val, y_val_pred_prob_annd)
print("默认参数ANN模型的验证集 AUC:", auc_annd)
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
print("最佳ANN参数组合:", best_params_ann)
print("调优ANN模型详细参数:", pd.DataFrame.from_dict(ann_model_best.get_params(), orient='index'))
print("默认参数ANN模型的验证集 AUC:", auc_annd)
print("参数调优ANN模型最佳模型的交叉验证 AUC:", best_auc_ann)
with open(OUTPUT_DIR / "ann_model.pkl", 'wb') as f:
    pickle.dump(ann_model_best, f)

# Record selected settings and mean training cross-validation AUCs.
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
(OUTPUT_DIR / "logistic_equation.txt").write_text(
    formula_logit + "\nP = 1 / [1 + exp(-Logit(P))]\n", encoding="utf-8"
)
