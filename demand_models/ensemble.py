# demand_models/ensemble.py

import os
import numpy as np
import pandas as pd
import joblib

# -------------------------------
# Paths
# -------------------------------
DATA_PATH = "data/processed_data.csv"

MODEL_DIR = "models"
ARIMA_DIR = os.path.join(MODEL_DIR, "arima_models")

XGB_PATH = os.path.join(MODEL_DIR, "xgb_demand.pkl")
RF_PATH  = os.path.join(MODEL_DIR, "rf_demand.pkl")
MLP_PATH = os.path.join(MODEL_DIR, "mlp_demand.pkl")

OUTPUT_PATH = os.path.join(MODEL_DIR, "demand_ensemble.csv")

# -------------------------------
# EXACT feature order (from training)
# -------------------------------
FEATURES = [
    "AgentPrice",
    "CompetitorPrice",
    "PriceChange",
    "NormalizedPrice",
    "DescEmbed1",
    "DescEmbed2",
    "FourierFeature1",
    "FourierFeature2"
]

# -------------------------------
# Load data
# -------------------------------
df = pd.read_csv(DATA_PATH)

df = df.dropna(subset=FEATURES)
X = df[FEATURES].astype(float)

# -------------------------------
# Load trained models
# -------------------------------
xgb_model = joblib.load(XGB_PATH)
rf_model  = joblib.load(RF_PATH)
mlp_model = joblib.load(MLP_PATH)   # pipeline (scaler + MLP)

# -------------------------------
# ML predictions
# -------------------------------
df["xgb_pred"] = xgb_model.predict(X)
df["rf_pred"]  = rf_model.predict(X)
df["mlp_pred"] = mlp_model.predict(X)

# -------------------------------
# ARIMA predictions (per ProductID)
# -------------------------------
arima_preds = []

for _, row in df.iterrows():
    pid = row["ProductID"]
    arima_path = os.path.join(ARIMA_DIR, f"arima_{pid}.pkl")

    if not os.path.exists(arima_path):
        arima_preds.append(np.nan)
        continue

    try:
        arima_model = joblib.load(arima_path)
        forecast = arima_model.forecast(steps=1)
        arima_preds.append(float(forecast.iloc[0]))
    except Exception:
        arima_preds.append(np.nan)

df["arima_pred"] = arima_preds

# -------------------------------
# Fill missing ARIMA safely
# -------------------------------
df["arima_pred"] = df["arima_pred"].fillna(
    df[["xgb_pred", "rf_pred", "mlp_pred"]].mean(axis=1)
)

# -------------------------------
# Weighted ensemble
# -------------------------------
df["ensemble_demand"] = (
    0.40 * df["xgb_pred"] +
    0.30 * df["rf_pred"] +
    0.20 * df["mlp_pred"] +
    0.10 * df["arima_pred"]
)

# -------------------------------
# Save output
# -------------------------------
df[["ProductID", "ensemble_demand"]].to_csv(OUTPUT_PATH, index=False)

print("✅ Demand ensemble built successfully")
print(f"📁 Saved to: {OUTPUT_PATH}")
