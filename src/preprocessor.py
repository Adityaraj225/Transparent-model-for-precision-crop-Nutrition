"""
Agronomic Preprocessor & Feature Engineering Module
Transforms soil, crop, and climatic inputs into optimized feature vectors
for multi-target regression and explainable AI inference.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any
from sklearn.preprocessing import OneHotEncoder, RobustScaler


class AgronomicPreprocessor:
    """
    Handles feature engineering, scaling, and encoding for tabular crop nutrition data.
    """

    NUMERICAL_COLS = [
        "Soil_N", "Soil_P", "Soil_K", "Soil_pH", "Soil_OC",
        "Soil_Moisture", "Temperature", "Humidity", "Rainfall"
    ]
    CATEGORICAL_COLS = ["Crop"]
    TARGET_COLS = ["Rec_N", "Rec_P", "Rec_K"]

    def __init__(self, artifact_path: str = "artifacts/scalers/preprocessor.joblib"):
        self.artifact_path = artifact_path
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.scaler = RobustScaler()
        self.is_fitted = False
        self.feature_names: List[str] = []

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derives domain-informed agronomic interaction features.
        """
        df_eng = df.copy()
        # Nutrient balance ratios
        df_eng["NP_Ratio"] = df_eng["Soil_N"] / (df_eng["Soil_P"] + 1e-4)
        df_eng["NK_Ratio"] = df_eng["Soil_N"] / (df_eng["Soil_K"] + 1e-4)
        df_eng["PK_Ratio"] = df_eng["Soil_P"] / (df_eng["Soil_K"] + 1e-4)
        
        # pH stress distance from optimal neutral pH 6.5
        df_eng["pH_Stress"] = np.abs(df_eng["Soil_pH"] - 6.5)
        
        # Moisture-Rainfall water availability index
        df_eng["Water_Index"] = (df_eng["Soil_Moisture"] * df_eng["Rainfall"]) / 1000.0

        return df_eng

    def fit_transform(
        self, 
        df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Fits encoder and scaler on raw dataframe and transforms features and targets.
        """
        os.makedirs(os.path.dirname(self.artifact_path), exist_ok=True)
        
        # Feature engineering
        df_eng = self._engineer_features(df)
        eng_num_cols = self.NUMERICAL_COLS + ["NP_Ratio", "NK_Ratio", "PK_Ratio", "pH_Stress", "Water_Index"]

        # Scale numerical features
        scaled_num = self.scaler.fit_transform(df_eng[eng_num_cols])

        # Encode categorical features
        encoded_cat = self.encoder.fit_transform(df_eng[self.CATEGORICAL_COLS])
        cat_feature_names = list(self.encoder.get_feature_names_out(self.CATEGORICAL_COLS))

        # Combine
        X = np.hstack([scaled_num, encoded_cat])
        self.feature_names = eng_num_cols + cat_feature_names

        # Targets
        y = df_eng[self.TARGET_COLS].values

        self.is_fitted = True
        self.save()
        print(f"Fitted preprocessor with {len(self.feature_names)} features: {self.feature_names}")
        return X, y, self.feature_names

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transforms new incoming test samples using fitted transformers.
        """
        if not self.is_fitted:
            self.load()

        df_eng = self._engineer_features(df)
        eng_num_cols = self.NUMERICAL_COLS + ["NP_Ratio", "NK_Ratio", "PK_Ratio", "pH_Stress", "Water_Index"]

        scaled_num = self.scaler.transform(df_eng[eng_num_cols])
        encoded_cat = self.encoder.transform(df_eng[self.CATEGORICAL_COLS])
        X = np.hstack([scaled_num, encoded_cat])
        return X

    def save(self):
        joblib.dump({
            "scaler": self.scaler,
            "encoder": self.encoder,
            "feature_names": self.feature_names,
            "is_fitted": self.is_fitted
        }, self.artifact_path)

    def load(self):
        if not os.path.exists(self.artifact_path):
            raise FileNotFoundError(f"Preprocessor artifact not found at {self.artifact_path}")
        data = joblib.load(self.artifact_path)
        self.scaler = data["scaler"]
        self.encoder = data["encoder"]
        self.feature_names = data["feature_names"]
        self.is_fitted = data["is_fitted"]
