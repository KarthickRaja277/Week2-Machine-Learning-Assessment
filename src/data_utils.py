"""Shared reproducible loan dataset and preprocessing utilities."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "loan_applicants.csv"

def make_dataset(n_samples=1800, random_state=42):
    rng = np.random.default_rng(random_state)
    age = rng.integers(21, 66, n_samples)
    income = np.clip(rng.lognormal(10.75, 0.55, n_samples), 12000, 250000)
    credit_score = np.clip(rng.normal(660, 75, n_samples), 300, 850)
    employment_years = np.clip(rng.gamma(2.2, 3.0, n_samples), 0, 40)
    loan_amount = np.clip(income * rng.uniform(0.12, 0.85, n_samples) +
                          rng.normal(5000, 9000, n_samples), 2000, 150000)
    loan_term = rng.choice([12, 24, 36, 48, 60], n_samples, p=[.08,.15,.34,.18,.25])
    education = rng.choice(["High School", "Bachelor", "Master", "Doctorate"],
                           n_samples, p=[.30,.42,.22,.06])
    employment_type = rng.choice(["Salaried", "Self-employed", "Contract", "Unemployed"],
                                 n_samples, p=[.62,.20,.13,.05])
    gender = rng.choice(["Female", "Male", "Other"], n_samples, p=[.48,.49,.03])
    region = rng.choice(["North", "South", "East", "West"], n_samples)
    purpose = rng.choice(["Home", "Vehicle", "Education", "Medical", "Personal"], n_samples)
    dates = pd.Timestamp("2022-01-01") + pd.to_timedelta(
        rng.integers(0, 1460, n_samples), unit="D")
    # Probability is intentionally imbalanced and driven by plausible risk signals.
    logit = (-2.1 + (680-credit_score)/95 + (loan_amount/income-0.42)*2.0
             + (employment_type == "Unemployed")*1.0
             + (employment_type == "Contract")*0.25
             - employment_years*0.035 + (loan_term >= 48)*0.25
             + (age < 25)*0.25)
    probability = 1 / (1 + np.exp(-logit))
    default = rng.binomial(1, probability)
    df = pd.DataFrame({
        "application_date": dates.astype(str), "age": age, "income": income.round(2),
        "credit_score": credit_score.round(1), "employment_years": employment_years.round(1),
        "loan_amount": loan_amount.round(2), "loan_term": loan_term,
        "education": education, "employment_type": employment_type,
        "gender": gender, "region": region, "purpose": purpose, "default": default
    })
    # Add realistic missing values to exercise imputers.
    for col, frac in {"income": .04, "credit_score": .03, "employment_years": .04,
                      "education": .025, "employment_type": .02}.items():
        idx = rng.choice(df.index, int(n_samples*frac), replace=False)
        df.loc[idx, col] = np.nan
    return df

def load_data():
    if not DATA_PATH.exists():
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        make_dataset().to_csv(DATA_PATH, index=False)
    return pd.read_csv(DATA_PATH, parse_dates=["application_date"])

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds date parts, income-to-loan interaction and log-income feature."""
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        date = pd.to_datetime(X["application_date"], errors="coerce")
        X["application_year"] = date.dt.year
        X["application_month"] = date.dt.month
        if "loan_amount" in X.columns:
            X["income_loan_ratio"] = X["income"] / X["loan_amount"].replace(0, np.nan)
        X["log_income"] = np.log1p(X["income"].clip(lower=0))
        return X.drop(columns=["application_date"])

NUMERIC = ["age", "income", "credit_score", "employment_years", "loan_amount",
           "loan_term", "application_year", "application_month",
           "income_loan_ratio", "log_income"]
CATEGORICAL = ["employment_type", "gender", "region", "purpose"]
ORDINAL = ["education"]
EDUCATION_ORDER = [["High School", "Bachelor", "Master", "Doctorate"]]

def make_preprocessor(scale=True, include_loan_amount=True):
    numeric_columns = NUMERIC if include_loan_amount else [c for c in NUMERIC if c not in ("loan_amount", "income_loan_ratio")]
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipe = Pipeline(numeric_steps)
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    ordinal_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder(categories=EDUCATION_ORDER,
                                   handle_unknown="use_encoded_value", unknown_value=-1))
    ])
    return Pipeline([
        ("features", FeatureEngineer()),
        ("columns", ColumnTransformer([
            ("num", numeric_pipe, numeric_columns),
            ("cat", categorical_pipe, CATEGORICAL),
            ("ord", ordinal_pipe, ORDINAL)
        ], remainder="drop"))
    ])

def split_xy(df):
    # Loan amount is an input for default classification and a separate regression target.
    X = df.drop(columns=["default"]).copy()
    X["application_date"] = X["application_date"].astype(str)
    y = df["default"].astype(int)
    return X, y
