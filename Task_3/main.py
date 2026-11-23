# Task -3 (23/11/2026) - Karthick Raja M
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from src.data_utils import load_data, split_xy, make_preprocessor, FeatureEngineer, NUMERIC, CATEGORICAL, ORDINAL, EDUCATION_ORDER

OUT = Path(__file__).resolve().parent
df = load_data()
X, y = split_xy(df)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=.20, random_state=42, stratify=y)

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=200, oob_score=True,
        bootstrap=True, random_state=42, n_jobs=-1, class_weight="balanced_subsample")
}
fitted = {}
rows = []
for name, estimator in models.items():
    pipe = Pipeline([("preprocess", make_preprocessor()), ("model", estimator)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_val)
    score = accuracy_score(y_val, pred)
    fitted[name] = pipe
    rows.append({"Model": name, "Validation score (accuracy)": score})
    print(f"{name}: validation accuracy={score:.3f}")
    if name == "Random Forest":
        print(f"Random Forest OOB score: {pipe.named_steps['model'].oob_score_:.3f}")

# Logistic regression probabilities for five rows, compared with hard predictions.
lr = fitted["Logistic Regression"]
print("\nLogistic Regression predict_proba (first 5 validation rows):")
print(lr.predict_proba(X_val.iloc[:5]))
print("predict:", lr.predict(X_val.iloc[:5]).tolist())

# XGBoost with early stopping, with validation transformed by the training-fitted preprocessing.
try:
    from xgboost import XGBClassifier
    xgb_prep = make_preprocessor()
    Xtr_t = xgb_prep.fit_transform(X_train, y_train)
    Xval_t = xgb_prep.transform(X_val)
    xgb_model = XGBClassifier(n_estimators=500, max_depth=4, learning_rate=.05,
        subsample=.85, colsample_bytree=.85, objective="binary:logistic",
        eval_metric="logloss", early_stopping_rounds=25, random_state=42,
        n_jobs=-1)
    xgb_model.fit(Xtr_t, y_train, eval_set=[(Xval_t, y_val)], verbose=False)
    xgb_pipe = Pipeline([("preprocess", xgb_prep), ("model", xgb_model)])
    fitted["XGBoost"] = xgb_pipe
    rows.append({"Model": "XGBoost", "Validation score (accuracy)": accuracy_score(y_val, xgb_pipe.predict(X_val))})
    print("XGBoost validation accuracy:", f"{rows[-1]['Validation score (accuracy)']:.3f}")
except ImportError:
    print("XGBoost is not installed. Install requirements.txt to run this required model.")

# Top five feature importances for forest and XGBoost.
def transformed_feature_names(preprocessor):
    try:
        return preprocessor.named_steps["columns"].get_feature_names_out()
    except Exception:
        return np.array([f"feature_{i}" for i in range(len(fitted["Random Forest"].named_steps["model"].feature_importances_))])

for model_name in ["Random Forest", "XGBoost"]:
    if model_name in fitted:
        model = fitted[model_name].named_steps["model"]
        names = transformed_feature_names(fitted[model_name].named_steps["preprocess"])
        imp = pd.Series(model.feature_importances_, index=names).sort_values(ascending=False).head(5)
        print(f"\nTop 5 features – {model_name}:\n{imp.to_string()}")

# Regression: estimate loan amount using other applicant attributes (loan_amount excluded from X).
reg_X = df.drop(columns=["default", "loan_amount"])
reg_y = df["loan_amount"]
reg_train, reg_val, target_train, target_val = train_test_split(
    reg_X, reg_y, test_size=.20, random_state=42)
reg_models = {"LinearRegression": LinearRegression(), "Ridge": Ridge(alpha=1.0),
              "Lasso": Lasso(alpha=1000, max_iter=10000)}
print("\nLoan amount regression coefficients:")
for name, estimator in reg_models.items():
    pipe = Pipeline([("preprocess", make_preprocessor(include_loan_amount=False)), ("model", estimator)])
    pipe.fit(reg_train, target_train)
    r2 = pipe.score(reg_val, target_val)
    coef = pipe.named_steps["model"].coef_
    print(f"{name}: validation R2={r2:.3f}; coefficient count={len(coef)}")
    if name == "Lasso":
        print("Lasso zero coefficients:", int(np.sum(np.isclose(coef, 0))))

# K-Means on two/three numeric customer attributes.
cluster_data = df[["age", "income", "credit_score"]].copy()
cluster_data = cluster_data.fillna(cluster_data.median())
scaled = StandardScaler().fit_transform(cluster_data)
ks, inertias, silhouettes = [], [], []
for k in range(2, 9):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(scaled)
    ks.append(k); inertias.append(km.inertia_); silhouettes.append(silhouette_score(scaled, labels))
    print(f"K={k}: inertia={km.inertia_:.2f}, silhouette={silhouettes[-1]:.3f}")
best_k = ks[int(np.argmax(silhouettes))]
print("Best K by silhouette:", best_k)
final_km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df["customer_segment"] = final_km.fit_predict(scaled)
print("Segment profiles (mean age, income, credit score):")
print(df.groupby("customer_segment")[["age", "income", "credit_score"]].mean().round(1))

fig, ax = plt.subplots(figsize=(8,5))
ax.plot(ks, inertias, marker="o"); ax.set_xlabel("Number of clusters (k)")
ax.set_ylabel("Inertia"); ax.set_title("K-Means elbow curve"); fig.tight_layout()
fig.savefig(OUT / "kmeans_elbow.png", dpi=150); plt.close(fig)
fig, ax = plt.subplots(figsize=(8,5))
ax.plot(ks, silhouettes, marker="o"); ax.set_xlabel("Number of clusters (k)")
ax.set_ylabel("Silhouette score"); ax.set_title("K-Means silhouette scores")
fig.tight_layout(); fig.savefig(OUT / "kmeans_silhouette.png", dpi=150); plt.close(fig)

pd.DataFrame(rows).to_csv(OUT / "model_validation_results.csv", index=False)
print("\nValidation results saved to model_validation_results.csv")
