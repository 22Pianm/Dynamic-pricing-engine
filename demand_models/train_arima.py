# demand_models/train_arima.py
import pandas as pd
import joblib
import os
from statsmodels.tsa.arima.model import ARIMA

DATA_PATH = "data/processed_data.csv"
MODEL_DIR = "models/arima_models"

os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH, parse_dates=["Timestamp"])

for pid, g in df.groupby("ProductID"):
    g = g.sort_values("Timestamp")

    if len(g) < 30:
        continue  # too little data

    try:
        model = ARIMA(g["Demand"], order=(2, 1, 2))
        fitted = model.fit()
        joblib.dump(fitted, f"{MODEL_DIR}/arima_{pid}.pkl")
        print(f"✅ ARIMA saved for Product {pid}")
    except Exception as e:
        print(f"❌ ARIMA failed for Product {pid}: {e}")
