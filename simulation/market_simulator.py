# simulation/market_simulator.py

import numpy as np
import pandas as pd
from datetime import datetime

class MarketSimulator:
    def __init__(
        self,
        data_path="data/processed_data.csv",
        noise_std=0.05,
        seasonal_strength=0.1,
        seed=42
    ):
        self.df = pd.read_csv(data_path, parse_dates=["Timestamp"])
        self.noise_std = noise_std
        self.seasonal_strength = seasonal_strength
        self.rng = np.random.default_rng(seed)

        # Pre-group competitor prices by product
        self.comp_price_map = {
            pid: g["CompetitorPrice"].dropna().values
            for pid, g in self.df.groupby("ProductID")
        }

    def _seasonality(self, timestamp: pd.Timestamp):
        """
        Simple weekly + monthly seasonality
        """
        day_of_week = timestamp.dayofweek      # 0–6
        day_of_month = timestamp.day           # 1–31

        weekly = np.sin(2 * np.pi * day_of_week / 7)
        monthly = np.sin(2 * np.pi * day_of_month / 30)

        return self.seasonal_strength * (0.6 * weekly + 0.4 * monthly)

    def get_market_state(self, product_id, timestamp):
        """
        Returns simulated market signals
        """
        # -------- Competitor price --------
        prices = self.comp_price_map.get(product_id)

        if prices is None or len(prices) == 0:
            base_price = self.df["CompetitorPrice"].median()
        else:
            base_price = self.rng.choice(prices)

        # -------- Noise --------
        noise = self.rng.normal(0, self.noise_std * base_price)

        # -------- Seasonality --------
        seasonality = self._seasonality(timestamp)

        competitor_price = max(0.01, base_price + noise + seasonality)

        return {
            "CompetitorPrice": float(competitor_price),
            "Seasonality": float(seasonality),
            "Noise": float(noise)
        }
