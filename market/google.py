# market/google.py

from serpapi import GoogleSearch

SERPAPI_KEY="93a107764928e4b328e9796934770057ae20ba6ec400d259587988ed516d36cf"

def get_google_prices(product_id: str) -> list[float]:
    """
    Fetch live Google Shopping prices using SerpApi
    """

    params = {
        "engine": "google_shopping",
        "q": product_id,
        "hl": "en",
        "gl": "in",
        "api_key": SERPAPI_KEY
    }

    search = GoogleSearch(params)
    results = search.get_dict()

    prices = []
    if "shopping_results" in results:
        for item in results["shopping_results"]:
            try:
                price_str = item.get("price", "").replace("₹", "").replace(",", "")
                prices.append(float(price_str))
            except:
                continue

    return prices
