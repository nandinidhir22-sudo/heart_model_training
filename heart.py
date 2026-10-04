from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Heart_dataset"
MODEL_PATH = BASE_DIR / "heart_disease_models.joblib"
TARGET = "HeartDisease"


def make_pipeline(features, estimator):
    """Fit preprocessing and a model together as one pipeline."""
    numeric_columns = features.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = features.select_dtypes(exclude=np.number).columns.tolist()

    numeric_preprocessing = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_preprocessing = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("one_hot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_preprocessing, numeric_columns),
        ("categorical", categorical_preprocessing, categorical_columns),
    ])

    return Pipeline([
        ("preprocessing", preprocessor),
        ("model", estimator),
    ])


# Load the heart dataset and normalize common blank values.
df = pd.read_csv(DATA_PATH)
df = df.replace([" ", "NA", "N/A", ""], np.nan)

# Match the EDA cleaning: zero blood pressure/cholesterol is treated as missing.
for column in ["RestingBP", "Cholesterol"]:
    df[column] = pd.to_numeric(df[column], errors="coerce")
    df[column] = df[column].replace(0, np.nan)

# The target is binary: 0 = no heart disease, 1 = heart disease.
df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
df = df.dropna(subset=[TARGET]).copy()
df[TARGET] = df[TARGET].astype(int)

X = df.drop(columns=[TARGET])
y = df[TARGET]

# Train Logistic Regression and Linear Regression on the full data for the app.
logistic_pipeline = make_pipeline(
    X,
    LogisticRegression(max_iter=2000, random_state=42),
)
linear_pipeline = make_pipeline(
    X,
    LinearRegression(),
)

logistic_pipeline.fit(X, y)
linear_pipeline.fit(X, y)

# Save both fitted pipelines and the exact feature-column order.
joblib.dump(
    {
        "logistic_model": logistic_pipeline,
        "linear_model": linear_pipeline,
        "feature_columns": X.columns.tolist(),
    },
    MODEL_PATH,
)

print(f"Saved models to: {MODEL_PATH}")
print(f"Expected input columns: {X.columns.tolist()}")