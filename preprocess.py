import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os
import pickle
import numpy as np 


def load_data(path):
    df = pd.read_csv(path,sep=";")
    return df

def preprocess(df):
    df = df.copy()

    # Drop unnecessary columns
    df.drop(columns=["SESSION ID"], inplace=True, errors='ignore')

    # Handle missing values
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())

    # Feature Engineering
    df.columns = df.columns.str.strip().str.upper()

    df["total_clicks"] = df.groupby("ORDER")["ORDER"].transform("count")
    df["avg_price"] = df.groupby("ORDER")["PRICE"].transform("mean")

    # Create Classification Target (Simulated)
    df["purchase"] = (df["PRICE"] > df["PRICE"].median()).astype(int)

    # Regression Target
    df["revenue"] = np.log1p(df["PRICE"] * df["total_clicks"])

    # Encoding categorical columns
    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col])

    # Scaling
    scaler = StandardScaler()
    features = df[[
    "MONTH",
    "DAY",
    "COUNTRY",
    "PAGE",
    "PAGE 1 (MAIN CATEGORY)",
    "COLOUR",
    "LOCATION",
    "MODEL PHOTOGRAPHY"
]]
    scaled_features = scaler.fit_transform(features)
    os.makedirs("models", exist_ok=True)
    pickle.dump(scaler, open("models/scaler.pkl", "wb"))

    return scaled_features, df["purchase"], df["revenue"]