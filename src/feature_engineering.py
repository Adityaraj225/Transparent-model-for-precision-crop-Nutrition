"""
Advanced Agronomic Feature Engineering Module
Computes:
1. GDD (Growing Degree Days): Thermal time accumulation for crops
2. VPD (Vapor Pressure Deficit, kPa): Atmospheric water demand
3. Cyclic Encoding: Sin/Cos periodic seasonal and climatic representations
4. Soil chemical stoichiometric ratios (NP, NK, PK, pH Stress, Total NPK)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List


class AgronomicFeatureEngineer:
    """
    Implements advanced domain features for precision crop nutrition.
    """

    # Base temperatures for Indian crops (T_base in °C)
    CROP_T_BASE = {
        "Rice": 10.0, "Wheat": 4.5, "Maize": 10.0,
        "Cotton": 15.0, "Sugarcane": 12.0, "Potato": 7.0,
        "Tomato": 10.0, "Chickpea": 8.0
    }

    SEASON_MAP = {"Kharif": 0, "Rabi": 1, "Zaid": 2}

    @classmethod
    def calculate_gdd(cls, temp: float, crop: str = "Wheat") -> float:
        """
        Calculates Growing Degree Days (GDD) above biological base temperature:
        GDD = max(0.0, Temp - T_base)
        """
        t_base = cls.CROP_T_BASE.get(crop.capitalize(), 10.0)
        return float(max(0.0, temp - t_base))

    @classmethod
    def calculate_vpd(cls, temp_c: float, relative_humidity_pct: float) -> float:
        """
        Calculates Vapor Pressure Deficit (VPD in kPa):
        e_s(T) = 0.61078 * exp((17.27 * T) / (T + 237.3))
        VPD = e_s(T) * (1 - RH / 100)
        """
        rh_clamped = max(1.0, min(100.0, float(relative_humidity_pct)))
        # Saturation vapor pressure in kPa (Tetens equation)
        es = 0.61078 * np.exp((17.27 * temp_c) / (temp_c + 237.3))
        vpd = es * (1.0 - (rh_clamped / 100.0))
        return float(max(0.01, vpd))

    @classmethod
    def cyclic_encode_season(cls, season: str) -> Dict[str, float]:
        """
        Applies sine/cosine cyclic encoding to agricultural seasons.
        """
        idx = cls.SEASON_MAP.get(season.capitalize(), 0)
        angle = 2.0 * np.pi * idx / 3.0
        return {
            "Season_Sin": float(np.sin(angle)),
            "Season_Cos": float(np.cos(angle))
        }

    @classmethod
    def transform_dataframe(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enriches input dataframe with GDD, VPD, cyclic seasonal encodings, and stoichiometric ratios.
        """
        df_out = df.copy()

        # 1. GDD
        if "Temperature" in df_out.columns:
            crop_col = df_out["Crop_Type"] if "Crop_Type" in df_out.columns else (df_out["Crop"] if "Crop" in df_out.columns else ["Wheat"] * len(df_out))
            gdd_vals = [
                cls.calculate_gdd(temp=t, crop=str(c))
                for t, c in zip(df_out["Temperature"], crop_col)
            ]
            df_out["GDD"] = np.round(gdd_vals, 2)

        # 2. VPD
        if "Temperature" in df_out.columns and "Humidity" in df_out.columns:
            vpd_vals = [
                cls.calculate_vpd(temp_c=t, relative_humidity_pct=h)
                for t, h in zip(df_out["Temperature"], df_out["Humidity"])
            ]
            df_out["VPD_kPa"] = np.round(vpd_vals, 3)

        # 3. Cyclic Season Encoding
        if "Season" in df_out.columns:
            season_sin = []
            season_cos = []
            for s in df_out["Season"]:
                enc = cls.cyclic_encode_season(str(s))
                season_sin.append(enc["Season_Sin"])
                season_cos.append(enc["Season_Cos"])
            df_out["Season_Sin"] = np.round(season_sin, 4)
            df_out["Season_Cos"] = np.round(season_cos, 4)

        # 4. Stoichiometric Ratios
        n_col = "Nitrogen_Level" if "Nitrogen_Level" in df_out.columns else ("Soil_N" if "Soil_N" in df_out.columns else None)
        p_col = "Phosphorus_Level" if "Phosphorus_Level" in df_out.columns else ("Soil_P" if "Soil_P" in df_out.columns else None)
        k_col = "Potassium_Level" if "Potassium_Level" in df_out.columns else ("Soil_K" if "Soil_K" in df_out.columns else None)

        if n_col and p_col:
            df_out["NP_Ratio"] = np.round(df_out[n_col] / (df_out[p_col] + 1e-4), 3)
        if n_col and k_col:
            df_out["NK_Ratio"] = np.round(df_out[n_col] / (df_out[k_col] + 1e-4), 3)
        if p_col and k_col:
            df_out["PK_Ratio"] = np.round(df_out[p_col] / (df_out[k_col] + 1e-4), 3)

        ph_col = "Soil_pH" if "Soil_pH" in df_out.columns else None
        if ph_col:
            df_out["pH_Stress"] = np.round(np.abs(df_out[ph_col] - 6.5), 3)

        if n_col and p_col and k_col:
            df_out["Total_NPK"] = np.round(df_out[n_col] + df_out[p_col] + df_out[k_col], 2)

        return df_out
