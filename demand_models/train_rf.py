# demand_models/train_rf.py
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor

DATA_PATH = "data/processed_data.csv"
MODEL_PATH = "models/rf_demand.pkl"

df = pd.read_csv(DATA_PATH)

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

df = df.dropna(subset=FEATURES + ["Demand"])

X = df[FEATURES]
y = df["Demand"]

model = RandomForestRegressor(
    n_estimators=500,
    max_depth=12,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

model.fit(X, y)

joblib.dump(model, MODEL_PATH)
print("✅ RandomForest demand model saved:", MODEL_PATH)
