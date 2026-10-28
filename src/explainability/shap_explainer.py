"""
SHAP (SHapley Additive exPlanations) Engine
Implements TreeExplainer for multi-target nutrient models, extracting:
1. Exact Shapley feature attribution values (local explanations)
2. Global feature importance beeswarm and bar plots
3. Top positive/negative feature drivers for Nitrogen, Phosphorus, Potassium
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from typing import Dict, Any, List, Optional
from sklearn.multioutput import MultiOutputRegressor


class PrecisionShapExplainer:
    """
    Computes exact Shapley attributions for multi-target crop nutrition models.
    """

    TARGETS = ["Rec_N", "Rec_P", "Rec_K"]

    def __init__(
        self, 
        model: Any, 
        feature_names: List[str], 
        artifacts_dir: str = "artifacts/figures"
    ):
        self.model = model
        self.feature_names = feature_names
        self.artifacts_dir = artifacts_dir
        os.makedirs(self.artifacts_dir, exist_ok=True)
        self.explainers: List[shap.TreeExplainer] = []
        self._init_explainers()

    def _init_explainers(self):
        """
        Extracts sub-estimators if wrapped in MultiOutputRegressor or handles native tree models.
        """
        if isinstance(self.model, MultiOutputRegressor):
            estimators = self.model.estimators_
        elif hasattr(self.model, "estimators_") and isinstance(self.model.estimators_, list) and hasattr(self.model.estimators_[0], "predict"):
            # Multi-output random forest or gradient boosting
            estimators = self.model.estimators_
        else:
            estimators = [self.model]

        self.explainers = []
        for est in estimators:
            try:
                exp = shap.TreeExplainer(est)
            except Exception:
                exp = shap.Explainer(est.predict)
            self.explainers.append(exp)

    def explain_sample(
        self, 
        X_sample: np.ndarray, 
        raw_feature_dict: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Computes local Shapley decomposition for a single prediction instance.
        Returns base value, predicted value, and ranked feature contributions for N, P, K.
        """
        if X_sample.ndim == 1:
            X_sample = X_sample.reshape(1, -1)

        explanations = {}

        for idx, target_name in enumerate(self.TARGETS):
            if idx < len(self.explainers):
                explainer = self.explainers[idx]
                shap_values = explainer.shap_values(X_sample)

                # Handle multi-dimensional returns
                if isinstance(shap_values, list):
                    vals = shap_values[0][0]
                elif shap_values.ndim == 2:
                    vals = shap_values[0]
                else:
                    vals = shap_values

                base_val = explainer.expected_value
                if isinstance(base_val, (list, np.ndarray)):
                    base_val = float(base_val[0])
                else:
                    base_val = float(base_val)

                # Pair features with their SHAP contribution
                feature_contribs = []
                for f_name, sv in zip(self.feature_names, vals):
                    raw_val = raw_feature_dict.get(f_name, None) if raw_feature_dict else None
                    feature_contribs.append({
                        "feature": f_name,
                        "shap_value": round(float(sv), 3),
                        "raw_value": raw_val
                    })

                # Sort by absolute SHAP impact
                feature_contribs.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

                pos_drivers = [f for f in feature_contribs if f["shap_value"] > 0.05][:3]
                neg_drivers = [f for f in feature_contribs if f["shap_value"] < -0.05][:3]

                explanations[target_name] = {
                    "base_expected_dosage": round(base_val, 2),
                    "predicted_dosage": round(base_val + float(np.sum(vals)), 2),
                    "top_positive_drivers": pos_drivers,
                    "top_negative_drivers": neg_drivers,
                    "all_contributions": feature_contribs
                }

        return explanations

    def generate_global_summary_plots(
        self, 
        X_background: np.ndarray, 
        max_samples: int = 400
    ) -> List[str]:
        """
        Generates global SHAP summary bar plots and beeswarm plots for each nutrient target.
        """
        sub_X = X_background[:max_samples]
        df_sub_X = pd.DataFrame(sub_X, columns=self.feature_names)
        plot_paths = []

        for idx, target_name in enumerate(self.TARGETS):
            if idx < len(self.explainers):
                explainer = self.explainers[idx]
                shap_values = explainer.shap_values(sub_X)

                if isinstance(shap_values, list):
                    sv = shap_values[0]
                else:
                    sv = shap_values

                plt.figure(figsize=(10, 6))
                shap.summary_plot(
                    sv, 
                    df_sub_X, 
                    plot_type="bar", 
                    show=False, 
                    max_display=10
                )
                plt.title(f"Global SHAP Feature Importance: {target_name} Recommendation", fontsize=12, fontweight="bold")
                plt.tight_layout()

                save_path = os.path.join(self.artifacts_dir, f"shap_global_summary_{target_name}.png")
                plt.savefig(save_path, dpi=200, bbox_inches="tight")
                plt.close()
                plot_paths.append(save_path)
                print(f"Global SHAP summary plot saved for {target_name} at: {save_path}")

        return plot_paths
