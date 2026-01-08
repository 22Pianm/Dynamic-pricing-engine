from market.aggregator import get_live_market_signal

if __name__ == "__main__":
    product_id = "SKU_001"

    market_signal = get_live_market_signal(product_id)

    print("Live Market Signl:")
    print(market_signal)