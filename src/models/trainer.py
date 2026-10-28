"""
Multi-Target Model Training Engine
Trains and optimizes multiple regression algorithms:
- Random Forest Regressor
- Gradient Boosting Regressor (GBDT)
- XGBoost Regressor
- LightGBM Regressor
for site-specific N, P, K nutrient dosage recommendation.
"""

import os
import time
import joblib
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Optional imports with graceful availability checks
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False


class ModelTrainer:
    """
    Manages multi-target model training, hyperparameter configuration, and checkpointing.
    """

    TARGET_NAMES = ["Rec_N", "Rec_P", "Rec_K"]

    def __init__(self, artifacts_dir: str = "artifacts/saved_models"):
        self.artifacts_dir = artifacts_dir
        os.makedirs(self.artifacts_dir, exist_ok=True)
        self.models: Dict[str, Any] = {}
        self.training_times: Dict[str, float] = {}

    def get_candidate_models(self) -> Dict[str, Any]:
        """
        Initializes the candidate models with production-grade hyperparameters.
        """
        candidates = {
            "Random_Forest": RandomForestRegressor(
                n_estimators=120,
                max_depth=14,
                min_samples_split=4,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=42
            ),
            "Gradient_Boosting": MultiOutputRegressor(
                GradientBoostingRegressor(
                    n_estimators=120,
                    max_depth=5,
                    learning_rate=0.08,
                    subsample=0.85,
                    random_state=42
                )
            )
        }

        if HAS_XGB:
            candidates["XGBoost"] = MultiOutputRegressor(
                xgb.XGBRegressor(
                    n_estimators=120,
                    max_depth=5,
                    learning_rate=0.07,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    random_state=42,
                    n_jobs=-1
                )
            )

        if HAS_LGB:
            candidates["LightGBM"] = MultiOutputRegressor(
                lgb.LGBMRegressor(
                    n_estimators=120,
                    max_depth=6,
                    num_leaves=31,
                    learning_rate=0.07,
                    subsample=0.85,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1
                )
            )

        return candidates

    def train_all(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray
    ) -> Dict[str, Any]:
        """
        Trains all candidate models and logs training duration.
        """
        candidates = self.get_candidate_models()
        trained_models = {}

        for name, model in candidates.items():
            print(f"Training {name} on {X_train.shape[0]} samples with {X_train.shape[1]} features...")
            start_time = time.time()
            model.fit(X_train, y_train)
            duration = time.time() - start_time
            
            trained_models[name] = model
            self.training_times[name] = duration
            
            # Save checkpoint
            model_path = os.path.join(self.artifacts_dir, f"{name}.joblib")
            joblib.dump(model, model_path)
            print(f"  -> {name} trained in {duration:.2f}s and saved to {model_path}")

        self.models = trained_models
        return trained_models

    def save_champion_model(self, champion_name: str, model: Any) -> str:
        """
        Saves the best-evaluated model as champion_model.joblib for production serving.
        """
        champion_path = os.path.join(self.artifacts_dir, "champion_model.joblib")
        meta = {
            "name": champion_name,
            "model": model,
            "target_names": self.TARGET_NAMES,
            "training_time": self.training_times.get(champion_name, 0.0)
        }
        joblib.dump(meta, champion_path)
        print(f"Champion Model '{champion_name}' persisted to {champion_path}")
        return champion_path
