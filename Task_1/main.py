
# Task 1 - (23/11/206) - Karthick Raja M

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.data_utils import load_data, split_xy


OUT = Path(__file__).resolve().parent

df = load_data()
X, y = split_xy(df)

X = X.copy()
date = pd.to_datetime(X["application_date"], errors="coerce")
X["application_year"] = date.dt.year
X["application_month"] = date.dt.month
X = X.drop(columns=["application_date"])


# Train, Validation and Test Split (60/20/20)

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.40,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("Train / Validation / Test Split")

print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print(f"Test samples: {len(X_test)}")

print(f"Training default rate: {y_train.mean():.3f}")
print(f"Validation default rate: {y_val.mean():.3f}")


#Preprocessing

numeric = X_train.select_dtypes(include=np.number).columns.tolist()

categorical = X_train.select_dtypes(
    exclude=np.number
).columns.tolist()

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

prep = ColumnTransformer([
    ("num", numeric_pipeline, numeric),
    ("cat", categorical_pipeline, categorical)
])


# Decision Tree - Overfitting Demonstration

depths = [1, 3, 5, 10, None]

train_scores = []
val_scores = []

print("\nDECISION TREE RESULTS")

for depth in depths:

    model = Pipeline([
        ("prep", prep),
        ("tree", DecisionTreeClassifier(
            max_depth=depth,
            random_state=42
        ))
    ])

    model.fit(X_train, y_train)

    train_accuracy = accuracy_score(
        y_train,
        model.predict(X_train)
    )

    val_accuracy = accuracy_score(
        y_val,
        model.predict(X_val)
    )

    train_scores.append(train_accuracy)
    val_scores.append(val_accuracy)

    print(
        f"Depth {depth}: "
        f"train accuracy={train_accuracy:.3f}, "
        f"validation accuracy={val_accuracy:.3f}"
    )


# Plot Train vs Validation Accuracy

plt.figure(figsize=(8, 5))

plt.plot(
    [str(d) for d in depths],
    train_scores,
    marker="o",
    label="Train Accuracy"
)

plt.plot(
    [str(d) for d in depths],
    val_scores,
    marker="o",
    label="Validation Accuracy"
)

plt.xlabel("Decision Tree Max Depth")
plt.ylabel("Accuracy")
plt.title("Overfitting: Training vs Validation Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    OUT / "overfitting_depth_comparison.png",
    dpi=150
)

plt.show()
plt.close()


# Logistic Regression - 5 Fold Cross Validation

logistic_pipe = Pipeline([
    ("prep", prep),
    ("model", LogisticRegression(
        max_iter=10000,
        solver="lbfgs",
        class_weight="balanced",
        random_state=42
    ))
])

cv_scores = cross_val_score(
    logistic_pipe,
    X_train,
    y_train,
    cv=5,
    scoring="accuracy"
)

print("\nLOGISTIC REGRESSION - 5 FOLD CROSS VALIDATION")

print(f"Individual CV scores: {cv_scores}")

print(
    f"Mean CV Accuracy: {cv_scores.mean():.3f}"
)

print(
    f"Standard Deviation: {cv_scores.std():.3f}"
)

print(
    f"CV Result: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}"
)


# Data Leakage Demonstration

num_frame = X.select_dtypes(include=np.number).copy()

X_a, X_b, y_a, y_b = train_test_split(
    num_frame,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Incorrect method: Fit imputer and scaler on the complete dataset

leaky_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

leaky_pipeline.fit(num_frame)

X_a_leaky = leaky_pipeline.transform(X_a)
X_b_leaky = leaky_pipeline.transform(X_b)


# Correct method: Fit imputer and scaler only on training data

correct_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

correct_pipeline.fit(X_a)

X_a_correct = correct_pipeline.transform(X_a)
X_b_correct = correct_pipeline.transform(X_b)


print("\nDATA LEAKAGE DEMONSTRATION")

print(
    "Incorrect full-data scaler mean:",
    np.round(
        leaky_pipeline.named_steps["scaler"].mean_[:3], 2
    )
)

print(
    "Correct train-only scaler mean:",
    np.round(
        correct_pipeline.named_steps["scaler"].mean_[:3], 2
    )
)

print("\nData Leakage Explanation:")

print("1. Data leakage happens when test or validation information enters training.")
print("2. Scaling before splitting uses information from the entire dataset.")
print("3. Correct scaling fits the imputer and scaler using training data only.")
print("4. The same fitted transformations are applied to validation and test data.")


# Supervised vs Unsupervised Learning

print("\nProblem classification")

problems = [
    (
        "Spam Detection",
        "Supervised Learning - Classification"
    ),
    (
        "Customer Segmentation",
        "Unsupervised Learning - Clustering"
    ),
    (
        "House Price Prediction",
        "Supervised Learning - Regression"
    ),
    (
        "Anomaly Detection",
        "Unsupervised Learning - Anomaly Detection"
    )
]

for problem, answer in problems:
    print(f"{problem}: {answer}")


#Machine Learning Workflow - 7 Steps

print("\nMACHINE LEARNING WORKFLOW")

workflow = [
    "Define the business problem and target.",
    "Collect data and understand the dataset.",
    "Clean data and handle missing values and outliers.",
    "Perform exploratory data analysis and feature engineering.",
    "Split data into training, validation and testing sets.",
    "Train models and tune hyperparameters using cross-validation.",
    "Evaluate the final model and deploy and monitor it."
]

for i, step in enumerate(workflow, start=1):
    print(f"{i}. {step}")


print(
    f"Chart :"
    f"{OUT / 'overfitting_depth_comparison.png'}"
)
