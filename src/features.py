import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Churn"
SERVICE_COLS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]
BINARY_COLS = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
ONE_HOT_COLS = ["InternetService", "PaymentMethod"]


def clean_data(df):
    df = df.copy()  # orijinal tabloyu bozma
    df = df.drop(columns=["customerID"], errors="ignore")

    df["total_services"] = (df[SERVICE_COLS] == "Yes").sum(axis=1)
    df["is_automatic_payment"] = df["PaymentMethod"].str.contains("automatic").astype(int)

    for col in BINARY_COLS:
        df[col] = df[col].map({"Yes": 1, "No": 0})
    df["gender"] = df["gender"].map({"Female": 1, "Male": 0})


    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)


    for col in SERVICE_COLS:
        df[col] = df[col].replace("No internet service", "No").map({"Yes": 1, "No": 0})
    df["MultipleLines"] = (
        df["MultipleLines"].replace("No phone service", "No").map({"Yes": 1, "No": 0})
    )


    df["Contract"] = df["Contract"].map({"Month-to-month": 0, "One year": 1, "Two year": 2})


    if TARGET in df.columns:
        df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0})

    return df


def split_features_target(df):
    return df.drop(columns=TARGET), df[TARGET]


def build_preprocessor(X):
    numeric_cols = [c for c in X.columns if c not in ONE_HOT_COLS]
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), ONE_HOT_COLS),
            ("num", StandardScaler(), numeric_cols),
        ]
    )