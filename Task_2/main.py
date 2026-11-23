# Task -2 (23/11/2026) - Karthick Raja M
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from src.data_utils import load_data, split_xy, make_preprocessor

OUT = Path(__file__).resolve().parent
df = load_data()
X, y = split_xy(df)

numeric = X.select_dtypes(include="number")
knn_experiment = Pipeline([("imputer", KNNImputer(n_neighbors=5)),
                           ("scaler", StandardScaler())])
knn_result = knn_experiment.fit_transform(numeric)
print("KNNImputer experiment: numeric input shape", numeric.shape,
      "-> output shape", knn_result.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.20, random_state=42, stratify=y)


from sklearn.linear_model import LogisticRegression
pipeline = Pipeline([
    ("preprocess", make_preprocessor(scale=True)),
    ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced"))
])
scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="accuracy")
print(f"Pipeline CV accuracy: {scores.mean():.3f} +/- {scores.std():.3f}")

pipeline.fit(X_train, y_train)
model_path = OUT / "loan_default_pipeline.joblib"
joblib.dump(pipeline, model_path)
print("Saved fitted pipeline:", model_path)

new_rows = X_test.iloc[:3].copy()
loaded_pipeline = joblib.load(model_path)
predictions = loaded_pipeline.predict(new_rows)
print("Predictions for 3 new raw rows:", predictions.tolist())
print("Raw rows used for prediction:")
print(new_rows.to_string(index=False))
