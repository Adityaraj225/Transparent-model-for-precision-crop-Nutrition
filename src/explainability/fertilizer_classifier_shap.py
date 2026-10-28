"""
SHAP Explainer for Multi-Class Fertilizer Recommendation
Decomposes classification probabilities into exact Shapley attributions,
revealing why a specific fertilizer product was chosen.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from typing import Dict, Any, List, Optional


class FertilizerClassifierShap:
    """
    SHAP explainer specifically designed for the multi-class fertilizer recommendation model.
    """

    def __init__(
        self, 
        model: Any, 
        feature_names: List[str], 
        class_names: List[str],
        artifacts_dir: str = "artifacts/figures"
    ):
        self.model = model
        self.feature_names = feature_names
        self.class_names = class_names
        self.artifacts_dir = artifacts_dir
        os.makedirs(self.artifacts_dir, exist_ok=True)
        self.explainer = shap.TreeExplainer(self.model)

    def explain_sample(
        self, 
        X_sample: np.ndarray, 
        predicted_class_idx: int,
        raw_feature_dict: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Decomposes sample prediction into Shapley feature attributions for the predicted fertilizer.
        """
        if X_sample.ndim == 1:
            X_sample = X_sample.reshape(1, -1)

        shap_values = self.explainer.shap_values(X_sample)

        # Handle different SHAP output formats across multi-class tree models
        # SHAP can return a list of arrays (one per class) or a 3D array (samples, features, classes)
        if isinstance(shap_values, list):
            class_shap = shap_values[predicted_class_idx][0]
        elif shap_values.ndim == 3:
            class_shap = shap_values[0, :, predicted_class_idx]
        else:
            class_shap = shap_values[0]

        feature_contribs = []
        for feat, val in zip(self.feature_names, class_shap):
            feature_contribs.append({
                "feature": feat,
                "shap_value": round(float(val), 3),
                "raw_value": raw_feature_dict.get(feat, None) if raw_feature_dict else None
            })

        feature_contribs.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
        top_positive = [f for f in feature_contribs if f["shap_value"] > 0.02][:4]
        top_negative = [f for f in feature_contribs if f["shap_value"] < -0.02][:3]

        chosen_fertilizer = self.class_names[predicted_class_idx]

        return {
            "recommended_fertilizer": chosen_fertilizer,
            "top_positive_drivers": top_positive,
            "top_negative_drivers": top_negative,
            "all_attributions": feature_contribs
        }

    def generate_global_summary(self, X_background: np.ndarray, max_samples: int = 400) -> str:
        """
        Plots global multi-class feature importance.
        """
        sub_X = X_background[:max_samples]
        df_sub_X = pd.DataFrame(sub_X, columns=self.feature_names)
        shap_values = self.explainer.shap_values(sub_X)

        plt.figure(figsize=(11, 7))
        shap.summary_plot(
            shap_values, 
            df_sub_X, 
            class_names=self.class_names, 
            show=False, 
            max_display=12
        )
        plt.title("Global SHAP Feature Importance: Fertilizer Product Recommendation", fontsize=12, fontweight="bold")
        plt.tight_layout()

        save_path = os.path.join(self.artifacts_dir, "shap_fertilizer_classification.png")
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Global fertilizer classification SHAP plot saved to: {save_path}")
        return save_path
