"""
Unified Precision Crop Nutrition Inference Pipeline
Integrated 2-Tier Architecture:
Tier 1: Recommended Fertilizer Product Classification (Urea, DAP, MOP, Compost, Zinc Sulphate, NPK, SSP)
Tier 2: Quantitative Site-Specific Dosage (N, P, K kg/ha) + ICAR STCR Safety Clamping + Commercial Bags
Explainability: Dual SHAP Attributions + Natural Language Farmer Advisory
"""

import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

from src.preprocessor import AgronomicPreprocessor
from src.fertilizer_preprocessor import FertilizerClassifierPreprocessor
from src.agronomy.stcr_validator import STCRValidator
from src.agronomy.fertilizer_calc import FertilizerCalculator
from src.explainability.shap_explainer import PrecisionShapExplainer
from src.explainability.fertilizer_classifier_shap import FertilizerClassifierShap
from src.explainability.nlg_translator import AgronomicNLGTranslator


class PrecisionNutritionPipeline:
    """
    Unified 2-Tier production pipeline for precision crop nutrition.
    """

    def __init__(
        self,
        champion_dosage_model_path: str = "artifacts/saved_models/champion_model.joblib",
        dosage_preprocessor_path: str = "artifacts/scalers/preprocessor.joblib",
        champion_classifier_path: str = "artifacts/saved_models/champion_fertilizer_classifier.joblib",
        classifier_preprocessor_path: str = "artifacts/scalers/fertilizer_classifier_preprocessor.joblib"
    ):
        # 1. Load Dosage Model (Tier 2)
        self.has_dosage_model = os.path.exists(champion_dosage_model_path) and os.path.exists(dosage_preprocessor_path)
        if self.has_dosage_model:
            self.dosage_preprocessor = AgronomicPreprocessor(dosage_preprocessor_path)
            self.dosage_preprocessor.load()
            meta_dosage = joblib.load(champion_dosage_model_path)
            self.dosage_model = meta_dosage.get("model")
            self.dosage_model_name = meta_dosage.get("name", "Dosage_Champion")
            self.dosage_shap = PrecisionShapExplainer(
                model=self.dosage_model,
                feature_names=self.dosage_preprocessor.feature_names
            )

        # 2. Load Fertilizer Classifier (Tier 1)
        self.has_classifier = os.path.exists(champion_classifier_path) and os.path.exists(classifier_preprocessor_path)
        if self.has_classifier:
            self.clf_preprocessor = FertilizerClassifierPreprocessor(classifier_preprocessor_path)
            self.clf_preprocessor.load()
            meta_clf = joblib.load(champion_classifier_path)
            self.clf_model = meta_clf.get("model")
            self.clf_name = meta_clf.get("champion_name", "Classifier_Champion")
            self.clf_classes = meta_clf.get("class_names", [])
            self.clf_shap = FertilizerClassifierShap(
                model=self.clf_model,
                feature_names=self.clf_preprocessor.feature_names,
                class_names=self.clf_classes
            )

    def predict(
        self, 
        record: Dict[str, Any],
        farm_area_acres: float = 1.0
    ) -> Dict[str, Any]:
        """
        Executes unified inference: Product Classification + Quantitative Dosage + Dual SHAP + Advisory.
        """
        results = {
            "farm_metadata": {
                "farm_area_acres": farm_area_acres,
                "crop": record.get("Crop_Type", record.get("Crop", "Wheat"))
            }
        }

        crop = str(results["farm_metadata"]["crop"]).capitalize()

        # =================================================================
        # TIER 1: Fertilizer Product Recommendation Classification
        # =================================================================
        if self.has_classifier:
            df_clf = pd.DataFrame([record])
            # Ensure required columns exist with sensible defaults if missing
            default_clf_cols = {
                "Soil_Type": "Loamy", "Crop_Type": crop, "Crop_Growth_Stage": "Vegetative",
                "Season": "Kharif", "Irrigation_Type": "Canal", "Previous_Crop": "Wheat",
                "Region": "North", "Fertilizer_Used_Last_Season": 170.0, "Yield_Last_Season": 4.5,
                "Soil_pH": record.get("Soil_pH", 6.5),
                "Soil_Moisture": record.get("Soil_Moisture", 35.0),
                "Organic_Carbon": record.get("Organic_Carbon", record.get("Soil_OC", 0.55)),
                "Electrical_Conductivity": record.get("Electrical_Conductivity", 1.5),
                "Nitrogen_Level": record.get("Nitrogen_Level", record.get("Soil_N", 110)),
                "Phosphorus_Level": record.get("Phosphorus_Level", record.get("Soil_P", 40)),
                "Potassium_Level": record.get("Potassium_Level", record.get("Soil_K", 100)),
                "Temperature": record.get("Temperature", 26.0),
                "Humidity": record.get("Humidity", 60.0),
                "Rainfall": record.get("Rainfall", 120.0)
            }
            for k, v in default_clf_cols.items():
                if k not in df_clf.columns:
                    df_clf[k] = v

            X_clf = self.clf_preprocessor.transform(df_clf)
            pred_idx = int(self.clf_model.predict(X_clf)[0])
            pred_fertilizer = self.clf_classes[pred_idx]

            # Probabilities if available
            probs = {}
            if hasattr(self.clf_model, "predict_proba"):
                proba_vals = self.clf_model.predict_proba(X_clf)[0]
                for c_name, p_val in zip(self.clf_classes, proba_vals):
                    probs[c_name] = round(float(p_val) * 100.0, 1)

            # SHAP Explanation for chosen fertilizer
            clf_shap_exp = self.clf_shap.explain_sample(
                X_sample=X_clf,
                predicted_class_idx=pred_idx,
                raw_feature_dict=record
            )

            results["tier1_product_recommendation"] = {
                "recommended_fertilizer": pred_fertilizer,
                "confidence_score_percent": probs.get(pred_fertilizer, 90.0),
                "class_probabilities_percent": probs,
                "champion_classifier": self.clf_name,
                "shap_reasoning": {
                    "top_drivers_for_choice": clf_shap_exp["top_positive_drivers"],
                    "factors_against_choice": clf_shap_exp["top_negative_drivers"]
                }
            }

        # =================================================================
        # TIER 2: Quantitative Dosage (kg/ha) & Commercial Bag Allocation
        # =================================================================
        if self.has_dosage_model:
            dosage_record = {
                "Crop": crop,
                "Soil_N": float(record.get("Soil_N", record.get("Nitrogen_Level", 180))),
                "Soil_P": float(record.get("Soil_P", record.get("Phosphorus_Level", 25))),
                "Soil_K": float(record.get("Soil_K", record.get("Potassium_Level", 150))),
                "Soil_pH": float(record.get("Soil_pH", 6.5)),
                "Soil_OC": float(record.get("Soil_OC", record.get("Organic_Carbon", 0.55))),
                "Soil_Moisture": float(record.get("Soil_Moisture", 35)),
                "Temperature": float(record.get("Temperature", 26)),
                "Humidity": float(record.get("Humidity", 60)),
                "Rainfall": float(record.get("Rainfall", 120))
            }
            df_dosage = pd.DataFrame([dosage_record])
            X_dose = self.dosage_preprocessor.transform(df_dosage)
            raw_dose = self.dosage_model.predict(X_dose)[0]

            safe_nutrients, safety_meta = STCRValidator.validate_and_clamp(
                crop=crop,
                pred_n=float(raw_dose[0]),
                pred_p=float(raw_dose[1]),
                pred_k=float(raw_dose[2])
            )

            commercial_plan = FertilizerCalculator.calculate_commercial_fertilizers(
                n_rec_kg_ha=safe_nutrients["Nitrogen_N"],
                p_rec_kg_ha=safe_nutrients["Phosphorus_P"],
                k_rec_kg_ha=safe_nutrients["Potassium_K"],
                farm_area_acres=farm_area_acres
            )

            shap_dosage_exp = self.dosage_shap.explain_sample(
                X_sample=X_dose,
                raw_feature_dict=dosage_record
            )

            plain_advisory = AgronomicNLGTranslator.translate_to_plain_language(
                crop=crop,
                raw_inputs=dosage_record,
                shap_explanations=shap_dosage_exp,
                commercial_fertilizers=commercial_plan
            )

            results["tier2_dosage_prescription"] = {
                "validated_nutrients_kg_ha": safe_nutrients,
                "commercial_bags_allocation": commercial_plan["farm_customized"],
                "application_schedule": commercial_plan["application_schedule"],
                "safety_meta": safety_meta,
                "plain_language_advisory": plain_advisory
            }

        return results
