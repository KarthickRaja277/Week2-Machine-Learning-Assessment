# Task -4 (23/11/2026) - Karthick Raja M

import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import StandardScaler
from src.data_utils import load_data, make_preprocessor

OUT = Path(__file__).resolve().parent
df = load_data()
X = df.drop(columns=["default"]).copy()
X["application_date"] = X["application_date"].astype(str)
y = df["default"].astype(int)
# Keep test set untouched until the single final test evaluation.
X_dev, X_test, y_dev, y_test = train_test_split(
    X, y, test_size=.20, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(
    X_dev, y_dev, test_size=.25, random_state=42, stratify=y_dev)

estimators = {
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42,
        n_jobs=-1, class_weight="balanced_subsample")
}
pipes = {}
for name, estimator in estimators.items():
    pipe = Pipeline([("preprocess", make_preprocessor()), ("model", estimator)])
    pipe.fit(X_train, y_train); pipes[name] = pipe

try:
    from xgboost import XGBClassifier
    xgb_pipe = Pipeline([("preprocess", make_preprocessor()),
        ("model", XGBClassifier(n_estimators=300, max_depth=4, learning_rate=.05,
          subsample=.85, colsample_bytree=.85, eval_metric="logloss",
          random_state=42, n_jobs=-1))])
    xgb_pipe.fit(X_train, y_train)
    pipes["XGBoost"] = xgb_pipe
except ImportError:
    raise SystemExit("XGBoost is required for Task 4. Run: pip install xgboost")

# Required metrics for every classifier.
print("CLASSIFICATION METRICS (validation set)")
metric_rows = []
for name, pipe in pipes.items():
    pred = pipe.predict(X_val)
    acc = accuracy_score(y_val, pred)
    prec = precision_score(y_val, pred, zero_division=0)
    rec = recall_score(y_val, pred, zero_division=0)
    f1 = f1_score(y_val, pred, zero_division=0)
    cm = confusion_matrix(y_val, pred)
    auc = roc_auc_score(y_val, pipe.predict_proba(X_val)[:, 1])
    metric_rows.append({"Model": name, "Accuracy": acc, "Precision": prec,
                        "Recall": rec, "F1": f1, "ROC-AUC": auc})
    print(f"\n{name}: accuracy={acc:.3f}, precision={prec:.3f}, recall={rec:.3f}, F1={f1:.3f}, ROC-AUC={auc:.3f}")
    print("Confusion matrix:\n", cm)
pd.DataFrame(metric_rows).to_csv(OUT / "classification_metrics.csv", index=False)

print("\nWhy accuracy can mislead:")
print("If defaults are rare, a model predicting 'no default' for everyone can achieve high accuracy")
print("while detecting almost none of the actual defaulters. Precision, recall, F1 and ROC-AUC")
print("provide additional information about minority-class detection and ranking quality.")

# ROC curves for three models in one chart.
roc_names = ["Logistic Regression", "Random Forest", "XGBoost"]
plt.figure(figsize=(8,6))
for name in roc_names:
    fpr, tpr, _ = roc_curve(y_val, pipes[name].predict_proba(X_val)[:, 1])
    auc = roc_auc_score(y_val, pipes[name].predict_proba(X_val)[:, 1])
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
plt.plot([0,1], [0,1], linestyle="--", label="Chance")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC curves – validation set"); plt.legend(); plt.tight_layout()
plt.savefig(OUT / "roc_curves.png", dpi=150); plt.close()

# Find a probability threshold achieving recall >= 0.75; report its precision.
threshold_model_name = max(roc_names, key=lambda n: roc_auc_score(
    y_val, pipes[n].predict_proba(X_val)[:, 1]))
prob = pipes[threshold_model_name].predict_proba(X_val)[:, 1]
threshold_candidates = np.unique(prob)
valid = []
for threshold in threshold_candidates:
    pred = (prob >= threshold).astype(int)
    r = recall_score(y_val, pred, zero_division=0)
    if r >= .75:
        valid.append((precision_score(y_val, pred, zero_division=0), threshold, r))
if valid:
    precision_at_threshold, chosen_threshold, recall_at_threshold = max(valid)
    print(f"\nThreshold model: {threshold_model_name}; threshold={chosen_threshold:.4f}; "
          f"recall={recall_at_threshold:.3f}; precision={precision_at_threshold:.3f}")
else:
    print("\nNo tested threshold reached 75% recall.")

#  Random Forest with three tuned parameters.
rf_pipe = Pipeline([("preprocess", make_preprocessor()),
                    ("model", RandomForestClassifier(random_state=42, n_jobs=-1,
                     class_weight="balanced_subsample"))])
rf_grid = {"model__n_estimators": [100, 200],
           "model__max_depth": [None, 8],
           "model__min_samples_split": [2, 5]}
rf_search = GridSearchCV(rf_pipe, rf_grid, cv=2, scoring="roc_auc", n_jobs=1, verbose=0)
rf_search.fit(X_train, y_train)
print("\nRandom Forest GridSearchCV best parameters:", rf_search.best_params_)
print("Random Forest best CV ROC-AUC:", round(rf_search.best_score_, 4))

# RandomizedSearchCV for XGBoost with n_iter=15.
xgb_pipe = Pipeline([("preprocess", make_preprocessor()),
                     ("model", XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=-1))])
xgb_params = {
    "model__n_estimators": [30, 50, 80, 120],
    "model__max_depth": [2, 3, 4],
    "model__learning_rate": [.01, .03, .05, .1, .2],
    "model__subsample": [.7, .8, .9, 1.0],
    "model__colsample_bytree": [.7, .8, .9, 1.0],
    "model__min_child_weight": [1, 3, 5]
}
xgb_search = RandomizedSearchCV(xgb_pipe, xgb_params, n_iter=15, cv=2,
    scoring="roc_auc", random_state=42, n_jobs=1, verbose=0)
xgb_search.fit(X_train, y_train)
print("\nXGBoost RandomizedSearchCV best parameters:", xgb_search.best_params_)
print("XGBoost best CV ROC-AUC:", round(xgb_search.best_score_, 4))

# Choose the best tuned classifier by validation ROC-AUC, then evaluate it once on test.
candidates = {"Tuned Random Forest": rf_search.best_estimator_,
              "Tuned XGBoost": xgb_search.best_estimator_}
best_name, best_model, best_auc = None, None, -1
for name, model in candidates.items():
    score = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
    if score > best_auc:
        best_name, best_model, best_auc = name, model, score
print(f"\nSelected best model using validation ROC-AUC: {best_name} ({best_auc:.3f})")
test_pred = best_model.predict(X_test)
test_prob = best_model.predict_proba(X_test)[:, 1]
print("FINAL TEST evaluation (performed once):")
print(f"Accuracy={accuracy_score(y_test, test_pred):.3f}; "
      f"Precision={precision_score(y_test, test_pred, zero_division=0):.3f}; "
      f"Recall={recall_score(y_test, test_pred, zero_division=0):.3f}; "
      f"F1={f1_score(y_test, test_pred, zero_division=0):.3f}; "
      f"ROC-AUC={roc_auc_score(y_test, test_prob):.3f}")
print("Test confusion matrix:\n", confusion_matrix(y_test, test_pred))

# Regression: predict loan_amount using other available fields; compare with mean baseline.
reg_X = df.drop(columns=["default"]).copy()
X["application_date"] = X["application_date"].astype(str)
reg_y = df["loan_amount"]
reg_train, reg_test, y_reg_train, y_reg_test = train_test_split(
    reg_X, reg_y, test_size=.20, random_state=42)
reg_model = Pipeline([("preprocess", make_preprocessor(include_loan_amount=False)),
                      ("model", LinearRegression())])
baseline = DummyRegressor(strategy="mean")
reg_model.fit(reg_train, y_reg_train); baseline.fit(reg_train, y_reg_train)
reg_pred = reg_model.predict(reg_test); base_pred = baseline.predict(reg_test)
def report_regression(name, actual, predicted):
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    print(f"{name}: MAE={mean_absolute_error(actual, predicted):.2f}, "
          f"RMSE={rmse:.2f}, R2={r2_score(actual, predicted):.3f}")
print("\nREGRESSION: model vs mean baseline")
report_regression("Linear Regression", y_reg_test, reg_pred)
report_regression("Mean baseline", y_reg_test, base_pred)
