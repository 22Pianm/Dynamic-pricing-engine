# demand_models/train_mlp.py
import pandas as pd
import joblib
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

DATA_PATH = "data/processed_data.csv"
MODEL_PATH = "models/mlp_demand.pkl"

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

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPRegressor(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        max_iter=800,
        random_state=42
    ))
])

pipeline.fit(X, y)

joblib.dump(pipeline, MODEL_PATH)
print("✅ MLP demand model saved:", MODEL_PATH)
