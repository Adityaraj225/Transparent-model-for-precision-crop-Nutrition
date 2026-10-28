"""
Phase 1 Master Execution Script: Transparent Model for Precision Crop Nutrition
Executes the complete Phase 1 pipeline:
1. Agronomic Dataset Ingestion & Synthesis
2. Domain Feature Engineering & Preprocessing
3. Multi-Target Training (Random Forest, Gradient Boosting, XGBoost, LightGBM)
4. Comparative Benchmarking Matrix (R², RMSE, MAE, Latency, Interpretability)
5. Champion Model Selection & Serialization
6. Explainable AI Generation (Global SHAP plots & Local Attributions)
7. Natural Language Advisory & Commercial Fertilizer Conversion Demonstration
"""

import os
import sys
import json
import pandas as pd
from sklearn.model_selection import train_test_split

# Ensure workspace root is in python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.data_loader import DataLoader
from src.preprocessor import AgronomicPreprocessor
from src.models.trainer import ModelTrainer
from src.models.benchmark import ModelBenchmark
from src.explainability.shap_explainer import PrecisionShapExplainer
from src.pipeline import PrecisionNutritionPipeline


def main():
    print("=" * 80)
    print("PHASE 1: TRANSPARENT MODEL FOR PRECISION CROP NUTRITION")
    print("School of Computer Science & Engineering - Major Project Phase 1")
    print("=" * 80)

    # -------------------------------------------------------------
    # 1. Dataset Acquisition / Generation
    # -------------------------------------------------------------
    print("\n[Step 1/6] Loading & Validating Agricultural Dataset...")
    data_path = "data/processed/crop_nutrition_dataset.csv"
    loader = DataLoader()
    df = loader.load_data(data_path)
    print(f"Dataset Shape: {df.shape[0]} samples, {df.shape[1]} columns")
    print(f"Crops represented: {df['Crop'].unique().tolist()}")

    # -------------------------------------------------------------
    # 2. Preprocessing & Feature Engineering
    # -------------------------------------------------------------
    print("\n[Step 2/6] Feature Engineering & Agronomic Preprocessing...")
    preprocessor = AgronomicPreprocessor(artifact_path="artifacts/scalers/preprocessor.joblib")
    X, y, feature_names = preprocessor.fit_transform(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"Train split: {X_train.shape[0]} samples | Test split: {X_test.shape[0]} samples")
    print(f"Total Engineered Features: {len(feature_names)}")

    # -------------------------------------------------------------
    # 3. Multi-Target Model Training
    # -------------------------------------------------------------
    print("\n[Step 3/6] Training Multi-Target Candidate Models...")
    trainer = ModelTrainer(artifacts_dir="artifacts/saved_models")
    trained_models = trainer.train_all(X_train, y_train)

    # -------------------------------------------------------------
    # 4. Rigorous Benchmarking & Evaluation
    # -------------------------------------------------------------
    print("\n[Step 4/6] Benchmarking Models (R², RMSE, MAE, Latency, Interpretability)...")
    benchmarker = ModelBenchmark(results_dir="artifacts/benchmark_results")
    df_results, champion_name = benchmarker.evaluate_models(
        models=trained_models,
        X_test=X_test,
        y_test=y_test,
        training_times=trainer.training_times
    )

    champion_model = trained_models[champion_name]
    trainer.save_champion_model(champion_name, champion_model)

    # -------------------------------------------------------------
    # 5. Global Explainable AI (SHAP) Suite
    # -------------------------------------------------------------
    print("\n[Step 5/6] Generating Global Explainable AI (SHAP) Artifacts...")
    shap_explainer = PrecisionShapExplainer(
        model=champion_model,
        feature_names=feature_names,
        artifacts_dir="artifacts/figures"
    )
    plot_paths = shap_explainer.generate_global_summary_plots(X_train, max_samples=400)
    print(f"Generated {len(plot_paths)} global SHAP importance plots.")

    # -------------------------------------------------------------
    # 6. Real-World Field Inference & Natural Language Translation Demo
    # -------------------------------------------------------------
    print("\n[Step 6/6] Executing End-to-End Pipeline & Plain-Language Advisory...")
    pipeline = PrecisionNutritionPipeline(
        champion_model_path="artifacts/saved_models/champion_model.joblib",
        preprocessor_path="artifacts/scalers/preprocessor.joblib"
    )

    # Demonstration Scenario 1: Nitrogen-deficient soil growing Maize
    test_scenario_1 = {
        "Crop": "Maize",
        "Soil_N": 160.0,       # Low N (deficiency < 280)
        "Soil_P": 22.0,        # Medium P
        "Soil_K": 180.0,       # Medium K
        "Soil_pH": 6.8,        # Optimal neutral pH
        "Soil_OC": 0.38,       # Low Organic Carbon (< 0.50%)
        "Soil_Moisture": 32.0, # Normal field capacity %
        "Temperature": 28.5,   # °C
        "Humidity": 65.0,      # %
        "Rainfall": 195.0      # High rainfall (leaching risk)
    }

    result_1 = pipeline.predict(test_scenario_1, farm_area_acres=2.5)

    print("\n" + "=" * 70)
    print("DEMO TEST RESULT: SCENARIO 1 (Maize on 2.5 Acres)")
    print("=" * 70)
    print(f"Target Crop: {result_1['model_metadata']['target_crop']} | Farm Area: {result_1['model_metadata']['farm_area_acres']} Acres")
    print("\n1. PURE NUTRIENTS RECOMMENDED (kg/ha):")
    for k, v in result_1["validated_nutrients_kg_ha"].items():
        print(f"   • {k}: {v} kg/ha")

    print("\n2. COMMERCIAL FERTILIZER PRESCRIPTION (50 kg Bags for 2.5 Acres):")
    farm_rec = result_1["commercial_fertilizers"]["farm_customized"]
    print(f"   • Urea: {farm_rec['Urea_kg']} kg ({farm_rec['Urea_bags_50kg']} Bags)")
    print(f"   • DAP:  {farm_rec['DAP_kg']} kg ({farm_rec['DAP_bags_50kg']} Bags)")
    print(f"   • MOP:  {farm_rec['MOP_kg']} kg ({farm_rec['MOP_bags_50kg']} Bags)")

    print("\n3. FARMER-FRIENDLY NATURAL LANGUAGE EXPLANATION (XAI Translation Layer):")
    advisory = result_1["plain_language_advisory"]
    print(f"   Executive Summary:\n   {advisory['executive_summary']}\n")
    print("   Detailed Feature Reasoning:")
    for nutrient, bullets in advisory["detailed_explanations"].items():
        print(f"   [{nutrient}]")
        for b in bullets:
            print(f"     {b}")

    print("\n" + "=" * 70)
    print("PHASE 1 EXECUTION COMPLETED SUCCESSFULLY!")
    print("All models, scalers, benchmark metrics, and SHAP plots are saved in 'artifacts/'.")
    print("=" * 70)


if __name__ == "__main__":
    main()
