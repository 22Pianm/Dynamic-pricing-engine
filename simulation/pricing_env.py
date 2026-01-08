# simulation/pricing_env.py

import numpy as np
import pandas as pd
from gym import Env, spaces  # use classic gym for SB3 compatibility
from simulation.market_simulator import MarketSimulator


class PricingEnv(Env):
    """
    RL environment for dynamic pricing
    """

    def __init__(
        self,
        data_path="data/processed_data.csv",
        demand_path="models/demand_ensemble.csv",
        price_step=0.1
    ):
        super().__init__()

        # Load data
        self.df = pd.read_csv(data_path, parse_dates=["Timestamp"])
        self.demand_df = pd.read_csv(demand_path)

        # Merge ensemble demand
        self.df = self.df.merge(self.demand_df, on="ProductID", how="left")
        if "ensemble_demand" not in self.df.columns:
            raise ValueError("ensemble_demand column missing")
        self.df["ensemble_demand"] = self.df["ensemble_demand"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # Market simulator
        self.market = MarketSimulator(data_path=data_path)

        self.price_step = float(price_step)
        self.current_idx = 0
        self.agent_price = 0.0

        # Action & observation spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(8,), dtype=np.float32)

    def reset(self):
        self.current_idx = np.random.randint(0, len(self.df))
        row = self.df.iloc[self.current_idx]
        self.agent_price = float(row["AgentPrice"])
        return self._get_state(row)

    def step(self, action):
        # Clip action
        action = float(np.clip(action, -1.0, 1.0))

        # Update agent price
        self.agent_price *= (1 + self.price_step * action)
        self.agent_price = float(np.clip(self.agent_price, 0.01, 10_000.0))

        # Advance timestep
        self.current_idx += 1
        done = self.current_idx >= len(self.df)

        if done:
            return np.zeros(self.observation_space.shape, dtype=np.float32), 0.0, True, {}

        row = self.df.iloc[self.current_idx]

        # Reset agent price if product changes
        if row["ProductID"] != self.df.iloc[self.current_idx - 1]["ProductID"]:
            self.agent_price = float(row["AgentPrice"])

        # Market state
        market_state = self.market.get_market_state(product_id=row["ProductID"], timestamp=row["Timestamp"])
        competitor_price = market_state["CompetitorPrice"]

        # Demand
        demand = row["ensemble_demand"]
        if pd.isna(demand) or np.isinf(demand):
            demand = 0.0
        demand = float(np.clip(demand, 0.0, 1e6))

        # Reward
        cost = float(row["BaseCost"])
        reward = (self.agent_price - cost) * demand
        if not np.isfinite(reward):
            reward = 0.0

        # Next state
        next_state = self._build_state(row, competitor_price, demand)

        return next_state, reward, done, {}

    def _get_state(self, row):
        market_state = self.market.get_market_state(product_id=row["ProductID"], timestamp=row["Timestamp"])
        return self._build_state(row, market_state["CompetitorPrice"], row["ensemble_demand"])

    def _build_state(self, row, competitor_price, demand):
        state = np.array([
            row["BaseCost"],
            self.agent_price,
            competitor_price,
            demand,
            row["DescEmbed1"],
            row["DescEmbed2"],
            row["FourierFeature1"],
            row["FourierFeature2"]
        ], dtype=np.float32)

        # Clean NaN/Inf
        state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)

        # Scale numeric values
        state[0] /= 100.0
        state[1] /= 100.0
        state[2] /= 100.0
        state[3] /= 100.0

        # Clip to observation space
        state = np.clip(state, -10.0, 10.0)

        return state
