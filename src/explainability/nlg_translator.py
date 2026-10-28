"""
Agronomic Natural Language Generation (NLG) Translation Layer
Converts technical SHAP and LIME mathematical attribution values (phi)
into farmer-friendly, actionable natural language explanations.
"""

from typing import Dict, Any, List


class AgronomicNLGTranslator:
    """
    Translates raw XAI feature contributions into plain-language agricultural advisory.
    """

    FEATURE_FRIENDLY_NAMES = {
        "Soil_N": "Soil Available Nitrogen",
        "Soil_P": "Soil Available Phosphorus",
        "Soil_K": "Soil Available Potassium",
        "Soil_pH": "Soil Acidity/Alkalinity (pH)",
        "Soil_OC": "Soil Organic Carbon (Organic Matter)",
        "Soil_Moisture": "Field Moisture Level",
        "Temperature": "Ambient Temperature",
        "Humidity": "Atmospheric Humidity",
        "Rainfall": "Seasonal Rainfall",
        "NP_Ratio": "Nitrogen-to-Phosphorus Ratio",
        "NK_Ratio": "Nitrogen-to-Potassium Ratio",
        "PK_Ratio": "Phosphorus-to-Potassium Ratio",
        "pH_Stress": "Soil pH Imbalance",
        "Water_Index": "Effective Soil Water Index"
    }

    @classmethod
    def translate_to_plain_language(
        cls, 
        crop: str, 
        raw_inputs: Dict[str, Any], 
        shap_explanations: Dict[str, Any],
        commercial_fertilizers: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generates narrative, human-understandable explanation bullets for each nutrient.
        """
        crop_clean = crop.capitalize()
        advisories = {}

        # 1. Translate Nitrogen (N) Explanation
        n_data = shap_explanations.get("Rec_N", {})
        n_pred = n_data.get("predicted_dosage", 0.0)
        n_base = n_data.get("base_expected_dosage", 0.0)
        n_diff = n_pred - n_base

        n_bullets = []
        if n_diff > 3.0:
            n_bullets.append(f"Recommended Nitrogen is {n_pred:.1f} kg/ha (+{n_diff:.1f} kg/ha above regional baseline of {n_base:.1f} kg/ha).")
        elif n_diff < -3.0:
            n_bullets.append(f"Recommended Nitrogen is reduced to {n_pred:.1f} kg/ha ({abs(n_diff):.1f} kg/ha savings compared to baseline).")
        else:
            n_bullets.append(f"Recommended Nitrogen is at standard baseline dosage: {n_pred:.1f} kg/ha.")

        # Analyze top positive drivers for N
        for driver in n_data.get("top_positive_drivers", []):
            feat = driver["feature"]
            shap_val = driver["shap_value"]
            friendly = cls.FEATURE_FRIENDLY_NAMES.get(feat, feat)

            if "Soil_N" in feat:
                val = raw_inputs.get("Soil_N", "low")
                n_bullets.append(f"• Increased dosage by +{shap_val:.1f} kg/ha because existing soil Nitrogen is low ({val} kg/ha).")
            elif "Soil_OC" in feat:
                val = raw_inputs.get("Soil_OC", "low")
                n_bullets.append(f"• Added +{shap_val:.1f} kg/ha Nitrogen because Soil Organic Carbon is low ({val}%), meaning slow natural nutrient release.")
            elif "Rainfall" in feat:
                val = raw_inputs.get("Rainfall", 0)
                n_bullets.append(f"• Higher seasonal rainfall ({val} mm) increases leaching risk, requiring +{shap_val:.1f} kg/ha compensatory Nitrogen.")
            elif "Crop_" in feat and crop_clean in feat:
                n_bullets.append(f"• Crop '{crop_clean}' has strong vegetative foliage demands requiring +{shap_val:.1f} kg/ha Nitrogen.")

        # Analyze top negative drivers for N
        for driver in n_data.get("top_negative_drivers", []):
            feat = driver["feature"]
            shap_val = driver["shap_value"]
            friendly = cls.FEATURE_FRIENDLY_NAMES.get(feat, feat)

            if "Soil_N" in feat:
                val = raw_inputs.get("Soil_N", "adequate")
                n_bullets.append(f"• Saved {abs(shap_val):.1f} kg/ha Nitrogen because soil already contains adequate native reserves ({val} kg/ha).")
            elif "Soil_OC" in feat:
                val = raw_inputs.get("Soil_OC", "high")
                n_bullets.append(f"• Saved {abs(shap_val):.1f} kg/ha Nitrogen due to rich Soil Organic Carbon ({val}%), supporting active microbial mineralization.")

        advisories["Nitrogen_Advisory"] = n_bullets

        # 2. Translate Phosphorus (P) Explanation
        p_data = shap_explanations.get("Rec_P", {})
        p_pred = p_data.get("predicted_dosage", 0.0)
        p_base = p_data.get("base_expected_dosage", 0.0)
        p_diff = p_pred - p_base

        p_bullets = []
        if p_diff > 2.0:
            p_bullets.append(f"Recommended Phosphorus is {p_pred:.1f} kg/ha (+{p_diff:.1f} kg/ha adjustment).")
        elif p_diff < -2.0:
            p_bullets.append(f"Recommended Phosphorus is reduced to {p_pred:.1f} kg/ha ({abs(p_diff):.1f} kg/ha savings).")
        else:
            p_bullets.append(f"Phosphorus is maintained at normal baseline: {p_pred:.1f} kg/ha.")

        for driver in p_data.get("top_positive_drivers", []):
            feat = driver["feature"]
            shap_val = driver["shap_value"]
            if "Soil_P" in feat:
                val = raw_inputs.get("Soil_P", "low")
                p_bullets.append(f"• Boosted by +{shap_val:.1f} kg/ha due to low available soil Phosphorus ({val} kg/ha) to support root proliferation.")
            elif "Soil_pH" in feat or "pH_Stress" in feat:
                val = raw_inputs.get("Soil_pH", 7.0)
                p_bullets.append(f"• Soil pH ({val}) reduces phosphorus bioavailability, requiring +{shap_val:.1f} kg/ha compensation.")

        for driver in p_data.get("top_negative_drivers", []):
            feat = driver["feature"]
            shap_val = driver["shap_value"]
            if "Soil_P" in feat:
                val = raw_inputs.get("Soil_P", "high")
                p_bullets.append(f"• Reduced by {abs(shap_val):.1f} kg/ha because soil test shows healthy available Phosphorus ({val} kg/ha).")

        advisories["Phosphorus_Advisory"] = p_bullets

        # 3. Translate Potassium (K) Explanation
        k_data = shap_explanations.get("Rec_K", {})
        k_pred = k_data.get("predicted_dosage", 0.0)
        k_bullets = [f"Recommended Potassium dosage is {k_pred:.1f} kg/ha to ensure drought resilience and stalk strength."]

        for driver in k_data.get("top_positive_drivers", []):
            feat = driver["feature"]
            shap_val = driver["shap_value"]
            if "Soil_K" in feat:
                val = raw_inputs.get("Soil_K", "low")
                k_bullets.append(f"• Added +{shap_val:.1f} kg/ha Potassium because available soil Potassium ({val} kg/ha) is deficient.")

        for driver in k_data.get("top_negative_drivers", []):
            feat = driver["feature"]
            shap_val = driver["shap_value"]
            if "Soil_K" in feat:
                val = raw_inputs.get("Soil_K", "high")
                k_bullets.append(f"• Reduced Potassium application by {abs(shap_val):.1f} kg/ha due to abundant existing reserves ({val} kg/ha).")

        advisories["Potassium_Advisory"] = k_bullets

        # 4. Summary Executive Prescription
        farm_info = commercial_fertilizers.get("farm_customized", {})
        summary = (
            f"Prescription for {farm_info.get('farm_area_acres', 1.0)} Acre of {crop_clean}: "
            f"Apply {farm_info.get('Urea_bags_50kg', 0):.1f} bags of Urea (50kg), "
            f"{farm_info.get('DAP_bags_50kg', 0):.1f} bags of DAP (50kg), and "
            f"{farm_info.get('MOP_bags_50kg', 0):.1f} bags of MOP (50kg). "
            f"Follow the 3-split schedule (Basal, Tillering, Flowering) to avoid nutrient runoff."
        )

        return {
            "crop": crop_clean,
            "executive_summary": summary,
            "detailed_explanations": advisories
        }
