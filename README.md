# Transparent Model for Precision Crop Nutrition

**A Major Project by:**  
Aditya Raj, Sonal Singh, Kumar Markandey, Aditya Kumar  
*Department of Computer Science and Engineering, IILM University, Greater Noida*  
*Supervisor:* Dr. Swati Vashisht  

---

## Overview
This repository implements the **2-Tier Precision Crop Nutrition Engine** with **Explainable AI (SHAP & LIME)** and an **Agronomic Natural Language Translation Layer**.

It bridges two essential real-world agricultural problems:
1. **Tier 1 (Product Recommendation Classifier)**: Predicts the primary commercial fertilizer product (`Recommended_Fertilizer`: Urea, DAP, MOP, Compost, Zinc Sulphate, NPK, SSP) trained on 10,000 agricultural records.
2. **Tier 2 (Quantitative Dosage Regressor)**: Recommends site-specific Nitrogen (N), Phosphorus (P), and Potassium (K) quantitative nutrient requirements (kg/ha) backed by ICAR STCR agronomic equations, and calculates exact 50 kg commercial farm bags customized to farm acreage.

---

## Performance Summary

### Tier 1: Commercial Fertilizer Classifier (10,000 samples, 7 classes)
- **Champion Model:** **XGBoost Classifier**
- **Test Accuracy:** **87.75%**
- **Macro F1-Score:** **0.7327**
- **Inference Latency:** **0.0037 ms / sample**

### Tier 2: Multi-Target Quantitative Dosage Regressor (N, P, K kg/ha)
- **Champion Model:** **LightGBM Regressor**
- **Test $R^2$ Score:** **0.9963**
- **Mean Absolute Error (MAE):** **1.87 kg/ha**
- **Inference Latency:** **0.0124 ms / sample**

---

## Project Structure
```text
Precision_Crop_Nutrition/
├── data/
│   ├── fertilizer_recommendation.csv   # 10,000-sample fertilizer classification dataset
│   └── processed/
│       └── crop_nutrition_dataset.csv  # 3,500-sample quantitative dosage dataset
├── src/
│   ├── fertilizer_preprocessor.py      # Preprocessor for fertilizer recommendation classification
│   ├── preprocessor.py                 # Preprocessor for quantitative dosage regression
│   ├── data_loader.py                  # Agronomic STCR dataset loader
│   ├── models/
│   │   ├── fertilizer_classifier_trainer.py   # Multi-class trainer (RF, GBDT, XGB, LGBM)
│   │   ├── fertilizer_classifier_benchmark.py # Multi-class benchmark & confusion matrix
│   │   ├── trainer.py                         # Multi-target regression trainer
│   │   └── benchmark.py                       # Regression benchmark evaluator
│   ├── explainability/
│   │   ├── fertilizer_classifier_shap.py      # SHAP for fertilizer classification
│   │   ├── shap_explainer.py                  # SHAP TreeExplainer for dosage
│   │   ├── lime_explainer.py                  # LIME tabular explainer
│   │   └── nlg_translator.py                  # SHAP-to-natural-language text generator
│   ├── agronomy/
│   │   ├── fertilizer_calc.py                 # NPK -> Urea, DAP, MOP commercial bags
│   │   └── stcr_validator.py                  # ICAR STCR boundary clamping
│   └── pipeline.py                            # Unified 2-Tier Inference Pipeline
├── artifacts/
│   ├── saved_models/                          # Champion checkpoints (.joblib)
│   ├── scalers/                               # Preprocessors (.joblib)
│   ├── benchmark_results/                     # Benchmark tables, charts, confusion matrix
│   └── figures/                               # Global SHAP importance figures
├── run_phase1.py                              # Phase 1 runner (quantitative dosage)
├── run_fertilizer_evolution.py                # Evolved runner (fertilizer recommendation)
└── requirements.txt
```

---

## Quickstart & Commands

### 1. Run Evolved Fertilizer Product Pipeline
Trains and benchmarks on `data/fertilizer_recommendation.csv`, generates SHAP plots, confusion matrix, and runs the 2-tier demonstration:
```bash
.\.venv\Scripts\python.exe run_fertilizer_evolution.py
```

### 2. Run Quantitative Dosage Benchmark
```bash
.\.venv\Scripts\python.exe run_phase1.py
```

### 3. Run Custom Single-Query Prediction
```bash
.\.venv\Scripts\python.exe -c "from src.pipeline import PrecisionNutritionPipeline; p = PrecisionNutritionPipeline(); print(p.predict({'Soil_Type': 'Clay', 'Crop_Type': 'Rice', 'Crop_Growth_Stage': 'Vegetative', 'Season': 'Kharif', 'Nitrogen_Level': 40, 'Phosphorus_Level': 50, 'Potassium_Level': 70, 'Soil_pH': 6.5, 'Organic_Carbon': 0.45}, farm_area_acres=2.5)['tier1_product_recommendation'])"
```
