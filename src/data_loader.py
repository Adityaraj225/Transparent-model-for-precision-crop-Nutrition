"""
Data Loader and Agronomic Dataset Generator Module
Loads benchmark agricultural soil datasets or synthesizes high-fidelity
ICAR STCR (Soil Test Crop Response) experimental dataset with realistic
soil-weather-crop interaction dynamics.
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from src.agronomy.stcr_validator import STCRValidator


class DataLoader:
    """
    Manages loading, synthetic dataset generation, and splitting of tabular crop nutrition datasets.
    """

    SUPPORTED_CROPS = ["Rice", "Wheat", "Maize", "Cotton", "Sugarcane", "Chickpea", "Potato"]

    def __init__(self, raw_data_path: Optional[str] = None):
        self.raw_data_path = raw_data_path

    @classmethod
    def generate_synthetic_agronomic_dataset(
        cls, 
        n_samples: int = 3500, 
        random_state: int = 42,
        save_path: str = "data/processed/crop_nutrition_dataset.csv"
    ) -> pd.DataFrame:
        """
        Synthesizes a scientifically grounded tabular dataset mapping soil & climatic
        parameters to target N-P-K nutrient recommendations using ICAR STCR guidelines
        augmented with realistic non-linear agronomic noise and climate effects.
        """
        np.random.seed(random_state)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        crops = np.random.choice(cls.SUPPORTED_CROPS, size=n_samples)

        # Soil parameters distributions typical for Indian agricultural soils
        soil_n = np.random.uniform(120.0, 480.0, size=n_samples)    # kg/ha available N
        soil_p = np.random.uniform(8.0, 45.0, size=n_samples)        # kg/ha available P
        soil_k = np.random.uniform(80.0, 360.0, size=n_samples)     # kg/ha available K
        soil_ph = np.random.uniform(5.2, 8.4, size=n_samples)       # Soil pH
        soil_oc = np.random.uniform(0.25, 0.95, size=n_samples)     # Soil Organic Carbon %
        soil_moisture = np.random.uniform(18.0, 52.0, size=n_samples) # % Soil Moisture

        # Weather / Environmental parameters
        temperature = np.random.uniform(16.0, 38.0, size=n_samples) # Celsius
        humidity = np.random.uniform(35.0, 90.0, size=n_samples)    # % Relative Humidity
        rainfall = np.random.uniform(40.0, 260.0, size=n_samples)   # mm seasonal rainfall

        rec_n = []
        rec_p = []
        rec_k = []

        for i in range(n_samples):
            crop = crops[i]
            # Baseline STCR target
            stcr = STCRValidator.calculate_stcr_baseline(
                crop=crop,
                soil_n=soil_n[i],
                soil_p=soil_p[i],
                soil_k=soil_k[i],
                soil_ph=soil_ph[i],
                soil_oc=soil_oc[i]
            )

            base_n = stcr["stcr_n"]
            base_p = stcr["stcr_p"]
            base_k = stcr["stcr_k"]

            # Climatic interaction factors:
            # 1. High rainfall leaching: Increases N demand slightly (+5 to 10%)
            if rainfall[i] > 180.0:
                base_n *= (1.0 + (rainfall[i] - 180.0) / 800.0)

            # 2. Temperature & moisture influence on microbial mineralization:
            # Optimal temp ~25-30C and moisture 30-40% enhances mineralization, reducing fertilizer need
            if 25.0 <= temperature[i] <= 32.0 and 30.0 <= soil_moisture[i] <= 45.0:
                base_n *= 0.96
                base_p *= 0.97

            # Add Gaussian agronomic field variance (noise ~ 3%)
            noise_n = np.random.normal(0, base_n * 0.03)
            noise_p = np.random.normal(0, base_p * 0.03)
            noise_k = np.random.normal(0, base_k * 0.03)

            final_n = base_n + noise_n
            final_p = base_p + noise_p
            final_k = base_k + noise_k

            # Enforce safety bounds
            clamped, _ = STCRValidator.validate_and_clamp(crop, final_n, final_p, final_k)

            rec_n.append(clamped["Nitrogen_N"])
            rec_p.append(clamped["Phosphorus_P"])
            rec_k.append(clamped["Potassium_K"])

        df = pd.DataFrame({
            "Crop": crops,
            "Soil_N": np.round(soil_n, 2),
            "Soil_P": np.round(soil_p, 2),
            "Soil_K": np.round(soil_k, 2),
            "Soil_pH": np.round(soil_ph, 2),
            "Soil_OC": np.round(soil_oc, 3),
            "Soil_Moisture": np.round(soil_moisture, 2),
            "Temperature": np.round(temperature, 2),
            "Humidity": np.round(humidity, 2),
            "Rainfall": np.round(rainfall, 2),
            "Rec_N": np.round(rec_n, 2),
            "Rec_P": np.round(rec_p, 2),
            "Rec_K": np.round(rec_k, 2)
        })

        df.to_csv(save_path, index=False)
        print(f"Generated synthetic agronomic dataset with {len(df)} records at: {save_path}")
        return df

    def load_data(self, dataset_path: str = "data/processed/crop_nutrition_dataset.csv") -> pd.DataFrame:
        """
        Loads dataset from disk or generates it if not present.
        """
        if os.path.exists(dataset_path):
            df = pd.read_csv(dataset_path)
            print(f"Loaded existing dataset from {dataset_path} with {len(df)} samples.")
            return df
        else:
            print(f"Dataset not found at {dataset_path}. Synthesizing new benchmark dataset...")
            return self.generate_synthetic_agronomic_dataset(save_path=dataset_path)
