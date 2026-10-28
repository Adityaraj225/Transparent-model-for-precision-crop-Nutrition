"""
Evolved Fertilizer Recommendation & Classification Pipeline Runner
Trains, benchmarks, and explains multi-class models on data/fertilizer_recommendation.csv (10,000 samples)
and links into the unified 2-tier precision crop nutrition engine.
"""

import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.fertilizer_preprocessor import FertilizerClassifierPreprocessor
from src.models.fertilizer_classifier_trainer import FertilizerClassifierTrainer
from src.models.fertilizer_classifier_benchmark import FertilizerClassifierBenchmark
from src.explainability.fertilizer_classifier_shap import FertilizerClassifierShap
from src.pipeline import PrecisionNutritionPipeline


def main():
    print("=" * 80)
    print("EVOLVED MODEL: FERTILIZER RECOMMENDATION CLASSIFIER & 2-TIER PRECISION ENGINE")
    print("Training on: data/fertilizer_recommendation.csv (10,000 records)")
    print("=" * 80)

    dataset_path = "data/fertilizer_recommendation.csv"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Missing dataset at {dataset_path}")

    # 1. Ingestion
    df = pd.read_csv(dataset_path)
    print(f"\n[Step 1/5] Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns.")
    print(f"Target Distribution:\n{df['Recommended_Fertilizer'].value_counts().to_string()}")

    # 2. Preprocessing & Feature Engineering
    print("\n[Step 2/5] Fitting preprocessor and generating engineered interaction features...")
    preprocessor = FertilizerClassifierPreprocessor()
    X, y, feature_names, class_names = preprocessor.fit_transform(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Train split: {X_train.shape[0]} | Test split: {X_test.shape[0]} samples")
    print(f"Total features: {len(feature_names)} | Classes: {class_names}")

    # 3. Model Training
    print("\n[Step 3/5] Training 4 Multi-Class Classifiers (RF, GBDT, XGBoost, LightGBM)...")
    trainer = FertilizerClassifierTrainer(artifacts_dir="artifacts/saved_models")
    trained_models = trainer.train_all(X_train, y_train, num_classes=len(class_names))

    # 4. Benchmarking
    print("\n[Step 4/5] Evaluating performance metrics, latency, and confusion matrix...")
    benchmarker = FertilizerClassifierBenchmark(results_dir="artifacts/benchmark_results")
    df_results, champion_name = benchmarker.evaluate_classifiers(
        models=trained_models,
        X_test=X_test,
        y_test=y_test,
        class_names=class_names,
        training_times=trainer.training_times
    )

    champion_model = trained_models[champion_name]
    trainer.save_champion_classifier(champion_name, champion_model, class_names)

    # 5. SHAP Global Feature Importance
    print("\n[Step 5/5] Generating SHAP explainability artifacts...")
    shap_explainer = FertilizerClassifierShap(
        model=champion_model,
        feature_names=feature_names,
        class_names=class_names,
        artifacts_dir="artifacts/figures"
    )
    shap_explainer.generate_global_summary(X_train, max_samples=400)

    # 6. Integrated 2-Tier Pipeline Demonstration
    print("\n" + "=" * 80)
    print("DEMONSTRATION: UNIFIED 2-TIER PRECISION CROP NUTRITION ENGINE")
    print("=" * 80)

    pipeline = PrecisionNutritionPipeline()

    test_samples = [
        {
            "name": "Case A: Severe Nitrogen Deficit in Cotton (Vegetative Stage)",
            "data": {
                "Soil_Type": "Black", "Crop_Type": "Cotton", "Crop_Growth_Stage": "Vegetative",
                "Season": "Kharif", "Irrigation_Type": "Drip", "Previous_Crop": "Wheat",
                "Region": "Central", "Fertilizer_Used_Last_Season": 180.0, "Yield_Last_Season": 3.8,
                "Soil_pH": 6.8, "Soil_Moisture": 36.0, "Organic_Carbon": 0.42,
                "Electrical_Conductivity": 1.45, "Nitrogen_Level": 35, "Phosphorus_Level": 55,
                "Potassium_Level": 75, "Temperature": 30.0, "Humidity": 60.0, "Rainfall": 110.0
            },
            "acres": 3.0
        },
        {
            "name": "Case B: Acidic Soil Reaction (pH 5.1) in Wheat",
            "data": {
                "Soil_Type": "Silt", "Crop_Type": "Wheat", "Crop_Growth_Stage": "Sowing",
                "Season": "Rabi", "Irrigation_Type": "Canal", "Previous_Crop": "Rice",
                "Region": "North", "Fertilizer_Used_Last_Season": 160.0, "Yield_Last_Season": 4.2,
                "Soil_pH": 5.1, "Soil_Moisture": 38.0, "Organic_Carbon": 0.32,
                "Electrical_Conductivity": 1.55, "Nitrogen_Level": 110, "Phosphorus_Level": 60,
                "Potassium_Level": 80, "Temperature": 22.0, "Humidity": 65.0, "Rainfall": 80.0
            },
            "acres": 2.0
        }
    ]

    for case in test_samples:
        print(f"\n>>> Running Inference: {case['name']} ({case['acres']} Acres)")
        res = pipeline.predict(case["data"], farm_area_acres=case["acres"])

        t1 = res.get("tier1_product_recommendation", {})
        print(f"  [Tier 1 Product Recommendation]: {t1.get('recommended_fertilizer')} (Confidence: {t1.get('confidence_score_percent')}%)")
        print("  Top SHAP Drivers for Product Choice:")
        for d in t1.get("shap_reasoning", {}).get("top_drivers_for_choice", []):
            print(f"    • {d['feature']}: {d['shap_value']:+.3f}")

        t2 = res.get("tier2_dosage_prescription", {})
        bags = t2.get("commercial_bags_allocation", {})
        print(f"  [Tier 2 Quantitative Dosage Allocation for {case['acres']} Acres]:")
        print(f"    • Urea: {bags.get('Urea_bags_50kg', 0)} Bags | DAP: {bags.get('DAP_bags_50kg', 0)} Bags | MOP: {bags.get('MOP_bags_50kg', 0)} Bags")

        adv = t2.get("plain_language_advisory", {})
        print(f"  [Farmer Plain-Language Summary]:\n    \"{adv.get('executive_summary', '')}\"")

    print("\n" + "=" * 80)
    print("EVOLVED MODEL TRAINING AND EVALUATION COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
