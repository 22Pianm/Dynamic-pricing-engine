import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

# ---------------- CONFIG ----------------
N_PRODUCTS = 50
DAYS = 200
MAX_MARKET_PRICE = 3000
MIN_COST = 300
MAX_COST = 1500

rows = []

# ---------------------------------------
for pid in range(N_PRODUCTS):

    # ---- Product properties ----
    base_cost = np.random.uniform(MIN_COST, MAX_COST)
    base_price = np.random.uniform(base_cost * 1.3, MAX_MARKET_PRICE)

    category = np.random.randint(1, 6)
    brand_strength = np.random.uniform(0.3, 0.9)

    price_elasticity = np.random.uniform(1.0, 2.2)
    demand_trend = np.random.uniform(-0.001, 0.002)

    prev_price = base_price
    prev_demand = 1.0

    for d in range(DAYS):
        date = datetime.now() - timedelta(days=(DAYS - d))

        # ---- Seasonality ----
        t = date.timetuple().tm_yday / 365
        fourier1 = np.sin(2 * np.pi * t)
        fourier2 = np.cos(2 * np.pi * t)

        # ---- Market price ----
        competitor_price = base_price * np.random.uniform(0.8, 1.2)

        # ---- Agent price movement ----
        price_change = np.random.normal(0, 0.04)
        agent_price = np.clip(
            prev_price * (1 + price_change),
            base_cost * 1.1,
            MAX_MARKET_PRICE
        )

        # ---- Demand formulation ----
        price_gap = (agent_price - competitor_price) / competitor_price
        noise = np.random.normal(0, 0.05)

        demand_index = (
            prev_demand
            * (1 - price_elasticity * price_gap)
            * (1 + demand_trend * d)
            * (1 + 0.3 * fourier1)
            + noise
        )

        demand_index = np.clip(demand_index, 0.1, 1.5)

        # ---- Purchase probability ----
        purchase_prob = min(0.9, 0.35 * demand_index)

        if np.random.rand() < purchase_prob:
            quantity = np.random.poisson(lam=1 + demand_index)
            quantity = max(1, quantity)

            reward = (agent_price - base_cost) * quantity

            rows.append({
                "ProductID": pid,
                "Timestamp": date,
                "BaseCost": round(base_cost, 2),
                "AgentPrice": round(agent_price, 2),
                "CompetitorPrice": round(competitor_price, 2),
                "Quantity": quantity,
                "Demand": round(demand_index, 3),
                "PriceChange": round((agent_price - prev_price) / prev_price, 4),
                "NormalizedPrice": round(agent_price / MAX_MARKET_PRICE, 3),
                "DescEmbed1": round(category / 10, 3),
                "DescEmbed2": round(brand_strength, 3),
                "FourierFeature1": round(fourier1, 4),
                "FourierFeature2": round(fourier2, 4),
                "Reward": round(reward, 2)
            })

        prev_price = agent_price
        prev_demand = demand_index

# ---------------------------------------
df = pd.DataFrame(rows)
df.to_csv("data/processed_data.csv", index=False)

print("✅ Dataset generated")
print("Products:", df["ProductID"].nunique())
print("Rows:", df.shape[0])
print(df.head())
