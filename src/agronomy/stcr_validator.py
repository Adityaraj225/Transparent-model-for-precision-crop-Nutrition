"""
ICAR STCR (Soil Test Crop Response) Agronomic Validator Module
Implements agronomic equations and safety boundary validation based on
Indian Council of Agricultural Research (ICAR) guidelines.
"""

from typing import Dict, Any, Tuple


class STCRValidator:
    """
    Validates model predictions against ICAR STCR targeted yield equations
    and checks agronomic safety limits.
    """

    # ICAR STCR equation parameters: FN = a*T - b*SN, FP = c*T - d*SP, FK = e*T - f*SK
    # T is Target Yield (q/ha or t/ha), SN/SP/SK are Soil Available N, P, K in kg/ha
    STCR_COEFFICIENTS = {
        "Rice": {
            "default_target_yield": 45.0,  # q/ha
            "a": 4.39, "b": 0.52,
            "c": 2.21, "d": 0.61,
            "e": 2.44, "f": 0.47,
            "max_n": 180.0, "max_p": 90.0, "max_k": 100.0,
            "min_n": 40.0, "min_p": 20.0, "min_k": 20.0
        },
        "Wheat": {
            "default_target_yield": 45.0,  # q/ha
            "a": 4.40, "b": 0.40,
            "c": 3.28, "d": 0.54,
            "e": 2.50, "f": 0.28,
            "max_n": 160.0, "max_p": 80.0, "max_k": 80.0,
            "min_n": 40.0, "min_p": 20.0, "min_k": 15.0
        },
        "Maize": {
            "default_target_yield": 50.0,  # q/ha
            "a": 3.90, "b": 0.45,
            "c": 2.10, "d": 0.50,
            "e": 1.95, "f": 0.22,
            "max_n": 180.0, "max_p": 80.0, "max_k": 70.0,
            "min_n": 50.0, "min_p": 25.0, "min_k": 20.0
        },
        "Cotton": {
            "default_target_yield": 25.0,  # q/ha
            "a": 8.50, "b": 0.48,
            "c": 4.20, "d": 0.55,
            "e": 5.10, "f": 0.35,
            "max_n": 150.0, "max_p": 75.0, "max_k": 75.0,
            "min_n": 40.0, "min_p": 20.0, "min_k": 20.0
        },
        "Sugarcane": {
            "default_target_yield": 1000.0,  # q/ha (100 t/ha)
            "a": 0.32, "b": 0.38,
            "c": 0.18, "d": 0.45,
            "e": 0.22, "f": 0.20,
            "max_n": 250.0, "max_p": 120.0, "max_k": 150.0,
            "min_n": 80.0, "min_p": 40.0, "min_k": 40.0
        },
        "Chickpea": {
            "default_target_yield": 20.0,  # q/ha (Pulse crop, fixes atmospheric N)
            "a": 1.20, "b": 0.20,
            "c": 2.80, "d": 0.60,
            "e": 2.10, "f": 0.30,
            "max_n": 35.0, "max_p": 60.0, "max_k": 40.0,
            "min_n": 15.0, "min_p": 15.0, "min_k": 10.0
        },
        "Potato": {
            "default_target_yield": 250.0,  # q/ha
            "a": 0.85, "b": 0.35,
            "c": 0.65, "d": 0.48,
            "e": 0.90, "f": 0.32,
            "max_n": 180.0, "max_p": 120.0, "max_k": 150.0,
            "min_n": 60.0, "min_p": 40.0, "min_k": 50.0
        }
    }

    @classmethod
    def calculate_stcr_baseline(
        cls, 
        crop: str, 
        soil_n: float, 
        soil_p: float, 
        soil_k: float, 
        soil_ph: float = 7.0, 
        soil_oc: float = 0.55,
        target_yield: float = None
    ) -> Dict[str, float]:
        """
        Calculate theoretical agronomic nutrient requirement based on ICAR STCR equation.
        """
        crop_clean = crop.capitalize() if crop.capitalize() in cls.STCR_COEFFICIENTS else "Wheat"
        params = cls.STCR_COEFFICIENTS[crop_clean]
        T = target_yield if target_yield is not None else params["default_target_yield"]

        # Base STCR formulas
        fn = (params["a"] * T) - (params["b"] * soil_n)
        fp = (params["c"] * T) - (params["d"] * soil_p)
        fk = (params["e"] * T) - (params["f"] * soil_k)

        # Agronomic modulation based on Soil Organic Carbon (SOC)
        # Low SOC (< 0.50%): Mineralization is poor; increase N compensation by 15 kg/ha
        # High SOC (> 0.75%): Healthy mineralization; decrease N by 10 kg/ha
        if soil_oc < 0.50:
            fn += 15.0 * (0.50 - soil_oc) / 0.3
        elif soil_oc > 0.75:
            fn -= min(15.0, 15.0 * (soil_oc - 0.75) / 0.5)

        # Agronomic modulation based on Soil pH
        # Acidic soil (< 6.0): Phosphorus is fixed by Al/Fe; increase P requirement by 15%
        if soil_ph < 6.0:
            fp *= 1.15
        elif soil_ph > 8.0:
            # Calcareous soil, P precipitates with Calcium
            fp *= 1.10

        # Clamp within biological boundaries
        fn = max(params["min_n"], min(params["max_n"], fn))
        fp = max(params["min_p"], min(params["max_p"], fp))
        fk = max(params["min_k"], min(params["max_k"], fk))

        return {
            "stcr_n": round(fn, 2),
            "stcr_p": round(fp, 2),
            "stcr_k": round(fk, 2),
            "target_yield": T,
            "crop": crop_clean
        }

    @classmethod
    def validate_and_clamp(
        cls, 
        crop: str, 
        pred_n: float, 
        pred_p: float, 
        pred_k: float
    ) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """
        Ensures machine learning predictions do not violate biological safety limits.
        """
        crop_clean = crop.capitalize() if crop.capitalize() in cls.STCR_COEFFICIENTS else "Wheat"
        params = cls.STCR_COEFFICIENTS[crop_clean]

        clamped_n = max(params["min_n"], min(params["max_n"], pred_n))
        clamped_p = max(params["min_p"], min(params["max_p"], pred_p))
        clamped_k = max(params["min_k"], min(params["max_k"], pred_k))

        violations = []
        if pred_n > params["max_n"]:
            violations.append(f"Nitrogen ({pred_n:.1f} kg/ha) capped to safety limit ({params['max_n']} kg/ha) to prevent vegetative lodging and nitrate leaching.")
        elif pred_n < params["min_n"]:
            violations.append(f"Nitrogen ({pred_n:.1f} kg/ha) adjusted to minimum starter threshold ({params['min_n']} kg/ha).")

        if pred_p > params["max_p"]:
            violations.append(f"Phosphorus ({pred_p:.1f} kg/ha) capped to maximum agronomic limit ({params['max_p']} kg/ha).")
        if pred_k > params["max_k"]:
            violations.append(f"Potassium ({pred_k:.1f} kg/ha) capped to maximum agronomic limit ({params['max_k']} kg/ha).")

        safe_nutrients = {
            "Nitrogen_N": round(clamped_n, 2),
            "Phosphorus_P": round(clamped_p, 2),
            "Potassium_K": round(clamped_k, 2)
        }

        meta = {
            "crop": crop_clean,
            "is_clamped": len(violations) > 0,
            "warnings": violations
        }

        return safe_nutrients, meta
