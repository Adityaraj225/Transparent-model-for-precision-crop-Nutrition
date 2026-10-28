"""
Fertilizer Recommendation Preprocessor & Feature Engineering Module
Handles data transformation, categorical encoding, scaling, and feature engineering
for the 10,000-sample fertilizer recommendation dataset.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any
from sklearn.preprocessing import OneHotEncoder, RobustScaler, LabelEncoder


class FertilizerClassifierPreprocessor:
    """
    Preprocessor tailored for fertilizer product recommendation classification.
    """

    NUMERICAL_COLS = [
        "Soil_pH", "Soil_Moisture", "Organic_Carbon", "Electrical_Conductivity",
        "Nitrogen_Level", "Phosphorus_Level", "Potassium_Level",
        "Temperature", "Humidity", "Rainfall",
        "Fertilizer_Used_Last_Season", "Yield_Last_Season"
    ]

    CATEGORICAL_COLS = [
        "Soil_Type", "Crop_Type", "Crop_Growth_Stage",
        "Season", "Irrigation_Type", "Previous_Crop", "Region"
    ]

    TARGET_COL = "Recommended_Fertilizer"

    def __init__(self, artifact_path: str = "artifacts/scalers/fertilizer_classifier_preprocessor.joblib"):
        self.artifact_path = artifact_path
        self.scaler = RobustScaler()
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
        self.feature_names: List[str] = []
        self.class_names: List[str] = []

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derives agronomic interaction features.
        """
        df_eng = df.copy()
        df_eng["NP_Ratio"] = df_eng["Nitrogen_Level"] / (df_eng["Phosphorus_Level"] + 1e-4)
        df_eng["NK_Ratio"] = df_eng["Nitrogen_Level"] / (df_eng["Potassium_Level"] + 1e-4)
        df_eng["PK_Ratio"] = df_eng["Phosphorus_Level"] / (df_eng["Potassium_Level"] + 1e-4)
        df_eng["pH_Stress"] = np.abs(df_eng["Soil_pH"] - 6.5)
        df_eng["Total_NPK"] = df_eng["Nitrogen_Level"] + df_eng["Phosphorus_Level"] + df_eng["Potassium_Level"]
        return df_eng

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """
        Fits scalers, encoders, and target label encoder, returning X, y, feature_names, class_names.
        """
        os.makedirs(os.path.dirname(self.artifact_path), exist_ok=True)
        df_eng = self._engineer_features(df)

        eng_num_cols = self.NUMERICAL_COLS + ["NP_Ratio", "NK_Ratio", "PK_Ratio", "pH_Stress", "Total_NPK"]
        scaled_num = self.scaler.fit_transform(df_eng[eng_num_cols])

        encoded_cat = self.encoder.fit_transform(df_eng[self.CATEGORICAL_COLS])
        cat_feature_names = list(self.encoder.get_feature_names_out(self.CATEGORICAL_COLS))

        X = np.hstack([scaled_num, encoded_cat])
        self.feature_names = eng_num_cols + cat_feature_names

        # Encode target
        y = self.label_encoder.fit_transform(df[self.TARGET_COL])
        self.class_names = list(self.label_encoder.classes_)

        self.is_fitted = True
        self.save()
        print(f"Fitted fertilizer classifier preprocessor with {len(self.feature_names)} features.")
        print(f"Target classes ({len(self.class_names)}): {self.class_names}")
        return X, y, self.feature_names, self.class_names

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transforms incoming feature records for model inference.
        """
        if not self.is_fitted:
            self.load()

        df_eng = self._engineer_features(df)
        eng_num_cols = self.NUMERICAL_COLS + ["NP_Ratio", "NK_Ratio", "PK_Ratio", "pH_Stress", "Total_NPK"]

        scaled_num = self.scaler.transform(df_eng[eng_num_cols])
        encoded_cat = self.encoder.transform(df_eng[self.CATEGORICAL_COLS])
        X = np.hstack([scaled_num, encoded_cat])
        return X

    def decode_predictions(self, y_pred_indices: np.ndarray) -> np.ndarray:
        """
        Maps numerical predictions back to commercial fertilizer names.
        """
        return self.label_encoder.inverse_transform(y_pred_indices)

    def save(self):
        joblib.dump({
            "scaler": self.scaler,
            "encoder": self.encoder,
            "label_encoder": self.label_encoder,
            "feature_names": self.feature_names,
            "class_names": self.class_names,
            "is_fitted": self.is_fitted
        }, self.artifact_path)

    def load(self):
        if not os.path.exists(self.artifact_path):
            raise FileNotFoundError(f"Preprocessor artifact not found at {self.artifact_path}")
        data = joblib.load(self.artifact_path)
        self.scaler = data["scaler"]
        self.encoder = data["encoder"]
        self.label_encoder = data["label_encoder"]
        self.feature_names = data["feature_names"]
        self.class_names = data["class_names"]
        self.is_fitted = data["is_fitted"]
