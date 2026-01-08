import numpy as np
from market.amazon import get_amazon_prices
from market.google import get_google_prices

def get_live_market_signal(product_id: str) -> dict:
    amazon_prices = get_amazon_prices(product_id)
    google_prices = get_google_prices(product_id)

    all_prices = amazon_prices + google_prices
    if len(all_prices) == 0:
        raise ValueError("No market prices available")

    prices = np.array(all_prices, dtype=np.float32)

    return {
        "mean_price": float(np.mean(prices)),
        "min_price": float(np.min(prices)),
        "max_price": float(np.max(prices)),
        "volatility": float(np.std(prices)),
    }
