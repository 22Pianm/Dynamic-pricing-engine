# demand_models/train_xgb.py
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split

DATA_PATH = "data/processed_data.csv"
MODEL_PATH = "models/xgb_demand.pkl"

df = pd.read_csv(DATA_PATH)

TARGET = "Demand"

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

df = df.dropna(subset=FEATURES + [TARGET])

X = df[FEATURES]
y = df[TARGET]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = xgb.XGBRegressor(
    n_estimators=600,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="reg:squarederror",
    random_state=42
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

joblib.dump(model, MODEL_PATH)
print("✅ XGBoost demand model saved:", MODEL_PATH)
