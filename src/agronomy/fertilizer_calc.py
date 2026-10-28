"""
Commercial Fertilizer Converter Module
Translates elemental N-P-K nutrient recommendations into commercial fertilizer quantities:
- Urea (46% N)
- DAP (18% N, 46% P2O5)
- MOP / Muriate of Potash (60% K2O)
"""

from typing import Dict, Any
import math


class FertilizerCalculator:
    """
    Translates pure elemental N, P, K requirements (kg/ha) into commercial fertilizer bags.
    """

    HA_TO_ACRE = 2.47105  # 1 Hectare = ~2.471 Acres
    BAG_WEIGHT_KG = 50.0  # Standard commercial fertilizer bag weight in India

    @classmethod
    def calculate_commercial_fertilizers(
        cls, 
        n_rec_kg_ha: float, 
        p_rec_kg_ha: float, 
        k_rec_kg_ha: float, 
        farm_area_acres: float = 1.0
    ) -> Dict[str, Any]:
        """
        Calculate commercial fertilizer quantities needed.
        
        Args:
            n_rec_kg_ha: Recommended Nitrogen in kg/ha
            p_rec_kg_ha: Recommended Phosphorus (P2O5) in kg/ha
            k_rec_kg_ha: Recommended Potassium (K2O) in kg/ha
            farm_area_acres: Area of farm in acres (defaults to 1.0 acre)
            
        Returns:
            Dictionary containing commercial fertilizer quantities, bags, and split application advice.
        """
        # Ensure non-negative nutrient values
        n_req = max(0.0, float(n_rec_kg_ha))
        p_req = max(0.0, float(p_rec_kg_ha))
        k_req = max(0.0, float(k_rec_kg_ha))

        # 1. DAP Calculation (Supplies 46% P2O5 and 18% N)
        if p_req > 0:
            dap_kg_ha = p_req / 0.46
            n_from_dap = dap_kg_ha * 0.18
        else:
            dap_kg_ha = 0.0
            n_from_dap = 0.0

        # 2. Remaining Nitrogen -> Urea (Supplies 46% N)
        n_remaining = max(0.0, n_req - n_from_dap)
        urea_kg_ha = n_remaining / 0.46

        # 3. MOP Calculation (Supplies 60% K2O)
        mop_kg_ha = k_req / 0.60 if k_req > 0 else 0.0

        # Conversion for user's specific farm area in acres
        acre_factor = farm_area_acres / cls.HA_TO_ACRE

        dap_farm_kg = dap_kg_ha * acre_factor
        urea_farm_kg = urea_kg_ha * acre_factor
        mop_farm_kg = mop_kg_ha * acre_factor

        dap_bags = dap_farm_kg / cls.BAG_WEIGHT_KG
        urea_bags = urea_farm_kg / cls.BAG_WEIGHT_KG
        mop_bags = mop_farm_kg / cls.BAG_WEIGHT_KG

        # Split Application Scheduling (Agronomic Standard Practice)
        # Basal: 100% DAP, 100% MOP, 33% Urea applied at sowing
        # Top-dress 1: 33% Urea at active tillering / vegetative stage (30-35 DAS)
        # Top-dress 2: 34% Urea at panicle initiation / flowering (55-60 DAS)
        basal_urea = urea_farm_kg * 0.33
        topdress1_urea = urea_farm_kg * 0.33
        topdress2_urea = urea_farm_kg * 0.34

        return {
            "pure_nutrients_kg_ha": {
                "Nitrogen_N": round(n_req, 2),
                "Phosphorus_P": round(p_req, 2),
                "Potassium_K": round(k_req, 2)
            },
            "commercial_fertilizers_kg_ha": {
                "Urea": round(urea_kg_ha, 2),
                "DAP": round(dap_kg_ha, 2),
                "MOP": round(mop_kg_ha, 2)
            },
            "farm_customized": {
                "farm_area_acres": farm_area_acres,
                "Urea_kg": round(urea_farm_kg, 2),
                "Urea_bags_50kg": round(urea_bags, 2),
                "DAP_kg": round(dap_farm_kg, 2),
                "DAP_bags_50kg": round(dap_bags, 2),
                "MOP_kg": round(mop_farm_kg, 2),
                "MOP_bags_50kg": round(mop_bags, 2)
            },
            "application_schedule": {
                "Basal_at_sowing": {
                    "DAP_kg": round(dap_farm_kg, 2),
                    "MOP_kg": round(mop_farm_kg, 2),
                    "Urea_kg": round(basal_urea, 2),
                    "notes": "Apply full DAP and MOP with 1/3rd Urea at seedbed preparation."
                },
                "Top_Dress_1_Vegetative": {
                    "Urea_kg": round(topdress1_urea, 2),
                    "notes": "Apply 1/3rd Urea at tillering/vegetative stage (approx 25-30 days after sowing)."
                },
                "Top_Dress_2_Flowering": {
                    "Urea_kg": round(topdress2_urea, 2),
                    "notes": "Apply final 1/3rd Urea before panicle/flowering initiation."
                }
            }
        }
