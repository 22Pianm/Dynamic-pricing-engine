# demand_models/utils.py

import joblib
import numpy as np

class DemandEnsemble:
    """
    Inference-only demand oracle.
    """

    def __init__(self, model_dir="models"):
        self.xgb = joblib.load(f"{model_dir}/xgb_demand.pkl")
        self.rf  = joblib.load(f"{model_dir}/rf_demand.pkl")
        self.mlp = joblib.load(f"{model_dir}/mlp_demand.pkl")

    def predict(self, X: np.ndarray) -> float:
        """
        X shape: (1, 8)
        Feature order MUST match training.
        """
        xgb_p = self.xgb.predict(X)[0]
        rf_p  = self.rf.predict(X)[0]
        mlp_p = self.mlp.predict(X)[0]

        return float(
            0.4 * xgb_p +
            0.3 * rf_p +
            0.3 * mlp_p
        )
