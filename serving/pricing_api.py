from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np

from market.aggregator import get_live_market_signal
from demand_models.utils import DemandEnsemble
from stable_baselines3 import SAC

# -------------------------------
# Load models (inference only)
# -------------------------------
SAC_PATH = "models/sac_pricing_policy.zip"

sac_model = SAC.load(SAC_PATH)
demand_model = DemandEnsemble(model_dir="models")

# -------------------------------
# FastAPI app
# -------------------------------
app = FastAPI(title="Dynamic Pricing API")

# ✅ CORS middleware (added here)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# Request schema
# -------------------------------
class PricingRequest(BaseModel):
    product_id: str
    base_cost: float
    last_price: float
    desc_embed: list[float]  # length 2
    fourier: list[float]     # length 2

# -------------------------------
# Helper function
# -------------------------------
def build_state(
    base_cost,
    last_price,
    competitor_price,
    demand_pred,
    desc_embed,
    fourier
):
    return np.array([
        base_cost,
        last_price,
        competitor_price,
        demand_pred,
        desc_embed[0],
        desc_embed[1],
        fourier[0],
        fourier[1]
    ], dtype=np.float32)

# -------------------------------
# API endpoint
# -------------------------------
@app.post("/price")
def get_dynamic_price(req: PricingRequest):
    # 1️⃣ Live market signal
    market = get_live_market_signal(req.product_id)
    competitor_price = float(market["mean_price"])

    # 2️⃣ Demand prediction (feature parity with training)
    demand_features = np.array([[
        req.last_price,                          # AgentPrice
        competitor_price,                        # CompetitorPrice
        req.last_price - competitor_price,       # PriceChange
        req.last_price / max(competitor_price, 1e-6),  # NormalizedPrice
        req.desc_embed[0],
        req.desc_embed[1],
        req.fourier[0],
        req.fourier[1]
    ]], dtype=np.float32)

    demand_pred = demand_model.predict(demand_features)

    # 3️⃣ Build RL state
    state = build_state(
        req.base_cost,
        req.last_price,
        competitor_price,
        demand_pred,
        req.desc_embed,
        req.fourier
    ).reshape(1, -1)  # SB3 expects batch dimension

    # 4️⃣ SAC action
    action, _ = sac_model.predict(state, deterministic=True)
    price_multiplier = float(np.clip(action[0], -1.0, 1.0))

    # 5️⃣ Final price
    final_price = req.base_cost * (1 + price_multiplier)

    # 6️⃣ Response
    return {
        "final_price": round(final_price, 2),
        "price_multiplier": round(price_multiplier, 4),
        "competitor_price": round(competitor_price, 2),
        "demand_prediction": round(demand_pred, 3),
        "volatility": round(float(market["volatility"]), 3)
    }
