# market/amazon.py

from serpapi import GoogleSearch

SERPAPI_KEY="93a107764928e4b328e9796934770057ae20ba6ec400d259587988ed516d36cf"

def get_amazon_prices(product_id: str) -> list[float]:
    """
    Fetch live Amazon prices using SerpApi
    """

    params = {
        "engine": "amazon",
        "amazon_domain": "amazon.in",  # adjust to your region
        "type": "product",
        "product_id": product_id,
        "api_key": SERPAPI_KEY
    }

    search = GoogleSearch(params)
    results = search.get_dict()

    prices = []
    if "prices" in results:
        for item in results["prices"]:
            try:
                price_str = item.get("price", "").replace("₹", "").replace(",", "")
                prices.append(float(price_str))
            except:
                continue

    return prices
