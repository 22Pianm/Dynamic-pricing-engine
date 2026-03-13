import os
import warnings
import pandas as pd
import numpy as np

from flask import Flask, render_template, request, jsonify
from stable_baselines3 import SAC
from serpapi import GoogleSearch

warnings.filterwarnings("ignore")

app = Flask(__name__)
SERPAPI_KEY=os.getenv("93a107764928e4b328e9796934770057ae20ba6ec400d259587988ed516d36cf")


# ----------------------------
# LOAD MODEL
# ----------------------------

SAC_MODEL_PATH = "models/sac_pricing_policy.zip"
sac_model = SAC.load(SAC_MODEL_PATH)

print("✅ SAC model loaded")


# ----------------------------
# CATEGORY SYSTEM
# ----------------------------

def static_categories(name):

    name = name.lower()

    if "table" in name:
        return [("Furniture › Tables", "tables")]

    if "shirt" in name:
        return [("Clothing › Shirts", "shirts")]

    if "laptop" in name:
        return [("Electronics › Laptops", "laptops")]

    return [("General Products", "general")]


@app.route("/api/categories")
def api_categories():

    product = request.args.get("product", "")

    cats = static_categories(product)

    return jsonify([
        {"name": c[0], "url": c[1]} for c in cats
    ])


# ----------------------------
# GOOGLE SHOPPING COMPETITORS
# ----------------------------

def get_live_competitors(product_name):

    params = {
        "engine": "google_shopping",
        "q": product_name,
        "api_key": SERPAPI_KEY,
        "gl": "in",
        "hl": "en",
    }

    search = GoogleSearch(params)
    data = search.get_dict()

    results = data.get("shopping_results", [])

    rows = []

    for r in results[:10]:

        price = r.get("extracted_price")

        if price:
            rows.append({
                "title": r.get("title", ""),
                "price": float(price)
            })

    if len(rows) == 0:
        raise RuntimeError("No competitor prices found")

    return pd.DataFrame(rows)


# ----------------------------
# DEMAND PROXY
# ----------------------------

def demand_proxy(cost, comp_price, stock):

    demand = 1.0 + (comp_price - cost) / max(cost, 1)

    if stock > 50:
        demand *= 0.9

    return float(np.clip(demand, 0.1, 5.0))


# ----------------------------
# MAIN ROUTE
# ----------------------------

@app.route("/", methods=["GET", "POST"])
def index():

    error = None

    if request.method == "POST":

        try:

            product = request.form["product_name"]
            cost = float(request.form["cost_price"])
            stock = int(request.form["stock_quantity"])

            competitors = get_live_competitors(product)

            avg_comp = competitors["price"].mean()

            demand = demand_proxy(cost, avg_comp, stock)

            state = np.array([[

                cost / 100,
                avg_comp / 100,
                avg_comp / 100,
                demand / 100,
                0,0,0,0

            ]], dtype=np.float32)

            action,_ = sac_model.predict(state, deterministic=True)

            raw_action = float(action[0][0])

            action_value = np.tanh(raw_action)

            lower = cost
            upper = cost * 2

            price = lower + (action_value + 1) / 2 * (upper - lower)

            price = round(float(price),2)

            result = {
                "product_name": product,
                "cost_price": cost,
                "base_price": avg_comp,
                "final_price": price,
                "profit_pct": round((price-cost)/cost*100,2)
            }

            return render_template(
                "result.html",
                result=result,
                competitor_data=competitors.to_dict("records")
            )

        except Exception as e:
            error = str(e)

    return render_template("index.html", error=error)


# ----------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)