"""
Fertilizer Recommendation Classifier Training Engine
Trains multi-class classification models:
- Random Forest Classifier
- Gradient Boosting Classifier
- XGBoost Classifier
- LightGBM Classifier
on the 10,000-sample fertilizer recommendation dataset.
"""

import os
import time
import joblib
import numpy as np
from typing import Dict, Any, List
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
import lightgbm as lgb


class FertilizerClassifierTrainer:
    """
    Manages multi-class training and serialization for fertilizer recommendation.
    """

    def __init__(self, artifacts_dir: str = "artifacts/saved_models"):
        self.artifacts_dir = artifacts_dir
        os.makedirs(self.artifacts_dir, exist_ok=True)
        self.training_times: Dict[str, float] = {}

    def get_candidate_classifiers(self, num_classes: int = 7) -> Dict[str, Any]:
        """
        Initializes multi-class classification models.
        """
        return {
            "Random_Forest": RandomForestClassifier(
                n_estimators=120,
                max_depth=12,
                min_samples_split=4,
                n_jobs=-1,
                random_state=42
            ),
            "Gradient_Boosting": GradientBoostingClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.08,
                subsample=0.85,
                random_state=42
            ),
            "XGBoost": xgb.XGBClassifier(
                n_estimators=120,
                max_depth=5,
                learning_rate=0.08,
                objective="multi:softprob",
                num_class=num_classes,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=42,
                n_jobs=-1
            ),
            "LightGBM": lgb.LGBMClassifier(
                n_estimators=120,
                max_depth=6,
                num_leaves=31,
                learning_rate=0.08,
                objective="multiclass",
                num_class=num_classes,
                subsample=0.85,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
        }

    def train_all(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray, 
        num_classes: int = 7
    ) -> Dict[str, Any]:
        """
        Trains all classification candidates and saves checkpoints.
        """
        candidates = self.get_candidate_classifiers(num_classes)
        trained_models = {}

        for name, model in candidates.items():
            print(f"Training {name} Classifier on {X_train.shape[0]} samples...")
            start_time = time.time()
            model.fit(X_train, y_train)
            duration = time.time() - start_time

            trained_models[name] = model
            self.training_times[name] = duration

            ckpt_path = os.path.join(self.artifacts_dir, f"classifier_{name}.joblib")
            joblib.dump(model, ckpt_path)
            print(f"  -> {name} trained in {duration:.2f}s and saved to {ckpt_path}")

        return trained_models

    def save_champion_classifier(
        self, 
        champion_name: str, 
        model: Any, 
        class_names: List[str]
    ) -> str:
        """
        Saves the best classification model as champion_fertilizer_classifier.joblib.
        """
        save_path = os.path.join(self.artifacts_dir, "champion_fertilizer_classifier.joblib")
        meta = {
            "champion_name": champion_name,
            "model": model,
            "class_names": class_names,
            "training_time": self.training_times.get(champion_name, 0.0)
        }
        joblib.dump(meta, save_path)
        print(f"Champion Classifier '{champion_name}' persisted to {save_path}")
        return save_path
