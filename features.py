"""Feature extraction, encoding and scaling for churn prediction.

Reads the cleaned data produced by eda.py, engineers features, encodes
categoricals, does a stratified train/test split and fits scalers on the
training set only (no test leakage). Writes three versions of the features:

  raw   - engineered + encoded, unscaled (tree models, readable SHAP values)
  std   - continuous features standardized with StandardScaler (z-scores)
  norm  - all features min-max normalized to [0, 1]
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

ROOT = Path(__file__).parent
CLEAN_PATH = ROOT / "data" / "data_clean.csv"
OUT_DIR = ROOT / "data" / "processed"
ARTIFACT_DIR = ROOT / "artifacts"

TEST_SIZE = 0.2
RANDOM_STATE = 42

ADDON_SERVICES = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]
BINARY_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "PhoneService", "MultipleLines", "PaperlessBilling",
] + ADDON_SERVICES
MULTI_CAT_COLS = ["InternetService", "Contract", "PaymentMethod"]


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def extract_features(df):
    df = df.copy()

    # "No internet service" / "No phone service" duplicate the information in
    # InternetService / PhoneService, so collapse them to plain "No".
    for col in ADDON_SERVICES + ["MultipleLines"]:
        df[col] = df[col].replace({"No internet service": "No", "No phone service": "No"})

    # Average monthly spend over the customer's lifetime. tenure == 0 customers
    # have not been billed yet, so use their current monthly charge.
    df["AvgMonthlyCharge"] = np.where(
        df["tenure"] > 0, df["TotalCharges"] / df["tenure"].replace(0, 1), df["MonthlyCharges"]
    )
    # Positive => customer now pays more than their historical average (price increase).
    df["ChargeIncrease"] = df["MonthlyCharges"] - df["AvgMonthlyCharge"]

    df["NumAddonServices"] = df[ADDON_SERVICES].eq("Yes").sum(axis=1)
    df["NumServices"] = (
        df["NumAddonServices"]
        + df["PhoneService"].eq("Yes")
        + df["MultipleLines"].eq("Yes")
        + df["InternetService"].ne("No")
    )
    # (HasInternet / IsMonthToMonth flags are not added: they would exactly
    # duplicate one-hot columns and split SHAP credit between identical features.)
    df["IsAutoPayment"] = df["PaymentMethod"].str.contains("automatic").astype(int)
    df["IsNewCustomer"] = (df["tenure"] <= 6).astype(int)
    df["LivesAlone"] = (df["Partner"].eq("No") & df["Dependents"].eq("No")).astype(int)
    # Fiber users without security/support churn the most in the EDA.
    df["FiberNoSupport"] = (
        df["InternetService"].eq("Fiber optic")
        & df["OnlineSecurity"].eq("No")
        & df["TechSupport"].eq("No")
    ).astype(int)
    # TotalCharges is heavily right-skewed; log1p makes it closer to normal.
    df["LogTotalCharges"] = np.log1p(df["TotalCharges"])
    return df


def encode(df):
    df = df.copy()
    df["gender"] = df["gender"].map({"Male": 1, "Female": 0})
    for col in BINARY_COLS:
        if col != "gender":
            df[col] = df[col].map({"Yes": 1, "No": 0})
    df = pd.get_dummies(df, columns=MULTI_CAT_COLS, dtype=int)
    df.columns = [c.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_") for c in df.columns]
    return df


def main():
    df = pd.read_csv(CLEAN_PATH)
    y = df["Churn"].map({"Yes": 1, "No": 0}).rename("Churn")

    X = encode(extract_features(df.drop(columns=["customerID", "Churn"])))
    # TotalCharges is replaced by its log; it is also ~ tenure * MonthlyCharges.
    X = X.drop(columns=["TotalCharges"])
    assert X.isna().sum().sum() == 0

    continuous = [
        "tenure", "MonthlyCharges", "LogTotalCharges", "AvgMonthlyCharge",
        "ChargeIncrease", "NumAddonServices", "NumServices",
    ]

    section("FEATURES")
    print(f"{X.shape[1]} features, {len(continuous)} continuous, {X.shape[1] - len(continuous)} binary")
    print(X.dtypes.to_string())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    section("SPLIT")
    print(f"Train: {X_train.shape}, churn rate {y_train.mean():.3f}")
    print(f"Test:  {X_test.shape}, churn rate {y_test.mean():.3f}")

    # Standardize continuous features only; binary 0/1 flags stay as-is so
    # their coefficients / SHAP values remain interpretable.
    std_scaler = StandardScaler().fit(X_train[continuous])
    X_train_std, X_test_std = X_train.copy(), X_test.copy()
    X_train_std[continuous] = std_scaler.transform(X_train[continuous])
    X_test_std[continuous] = std_scaler.transform(X_test[continuous])

    norm_scaler = MinMaxScaler().fit(X_train)
    X_train_norm = pd.DataFrame(norm_scaler.transform(X_train), columns=X.columns, index=X_train.index)
    X_test_norm = pd.DataFrame(norm_scaler.transform(X_test), columns=X.columns, index=X_test.index)

    section("STANDARDIZED (train, continuous) - expect mean 0, std 1")
    print(X_train_std[continuous].agg(["mean", "std"]).round(3).T.to_string())
    section("NORMALIZED (train) - expect range [0, 1]")
    print(X_train_norm[continuous].agg(["min", "max"]).round(3).T.to_string())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(exist_ok=True)
    outputs = {
        "X_train_raw": X_train, "X_test_raw": X_test,
        "X_train_std": X_train_std, "X_test_std": X_test_std,
        "X_train_norm": X_train_norm, "X_test_norm": X_test_norm,
    }
    for name, frame in outputs.items():
        frame.to_csv(OUT_DIR / f"{name}.csv", index=False)
    y_train.to_csv(OUT_DIR / "y_train.csv", index=False)
    y_test.to_csv(OUT_DIR / "y_test.csv", index=False)

    joblib.dump(
        {
            "standard_scaler": std_scaler,
            "minmax_scaler": norm_scaler,
            "continuous_features": continuous,
            "feature_names": list(X.columns),
        },
        ARTIFACT_DIR / "scalers.joblib",
    )
    print(f"\nSaved datasets to {OUT_DIR}/ and scalers to {ARTIFACT_DIR}/scalers.joblib")


if __name__ == "__main__":
    main()
