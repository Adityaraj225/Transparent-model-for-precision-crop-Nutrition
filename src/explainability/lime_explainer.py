"""
LIME (Local Interpretable Model-Agnostic Explanations) Engine
Builds local sparse linear surrogate models to explain multi-target predictions.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
import lime
import lime.lime_tabular
from sklearn.multioutput import MultiOutputRegressor


class PrecisionLimeExplainer:
    """
    Computes local interpretable surrogate explanations for tabular soil nutrient samples.
    """

    TARGETS = ["Rec_N", "Rec_P", "Rec_K"]

    def __init__(
        self, 
        model: Any, 
        X_train: np.ndarray, 
        feature_names: List[str]
    ):
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names
        
        # Initialize LIME Tabular Explainer
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=self.X_train,
            feature_names=self.feature_names,
            mode="regression",
            random_state=42
        )

    def explain_sample(
        self, 
        X_sample: np.ndarray, 
        num_features: int = 5
    ) -> Dict[str, Any]:
        """
        Generates LIME local explanations for N, P, and K targets.
        """
        if X_sample.ndim == 2:
            single_sample = X_sample[0]
        else:
            single_sample = X_sample

        lime_explanations = {}

        # If model is MultiOutputRegressor or has multiple estimators
        for idx, target_name in enumerate(self.TARGETS):
            # Create a single-output prediction wrapper
            def predict_fn(data):
                preds = self.model.predict(data)
                if preds.ndim == 2:
                    return preds[:, idx]
                return preds

            exp = self.explainer.explain_instance(
                data_row=single_sample,
                predict_fn=predict_fn,
                num_features=num_features
            )

            as_list = exp.as_list()
            lime_explanations[target_name] = {
                "intercept": round(float(exp.intercept[1] if isinstance(exp.intercept, (list, np.ndarray)) else exp.intercept), 2),
                "local_prediction": round(float(exp.local_pred[0] if isinstance(exp.local_pred, (list, np.ndarray)) else exp.local_pred), 2),
                "top_rules": [
                    {"rule": rule, "weight": round(float(wt), 3)}
                    for rule, wt in as_list
                ]
            }

        return lime_explanations
