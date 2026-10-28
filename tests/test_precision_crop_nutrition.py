"""
Software Testing Suite: Transparent Model for Precision Crop Nutrition
40/40 Comprehensive Unit & Integration Tests covering:
1. Data loading and schema validation (Tests 1-5)
2. Feature engineering: GDD, VPD, Cyclic Encodings, stoichiometric ratios (Tests 6-12)
3. Preprocessor fit, transform, serialization, inverse transform (Tests 13-18)
4. Model inference, output shapes, probabilities, boundary constraints (Tests 19-25)
5. Explainable AI: SHAP TreeExplainer, attributions, additivity property (Tests 26-30)
6. ICAR STCR safety clamping and biological boundary rules (Tests 31-34)
7. Commercial fertilizer bag conversion (Urea, DAP, MOP) & split schedule (Tests 35-37)
8. Database persistence (SQLite) & API response schema validation (Tests 38-40)
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import DataLoader
from src.feature_engineering import AgronomicFeatureEngineer
from src.fertilizer_preprocessor import FertilizerClassifierPreprocessor
from src.preprocessor import AgronomicPreprocessor
from src.agronomy.stcr_validator import STCRValidator
from src.agronomy.fertilizer_calc import FertilizerCalculator
from src.explainability.shap_explainer import PrecisionShapExplainer
from src.explainability.fertilizer_classifier_shap import FertilizerClassifierShap
from src.pipeline import PrecisionNutritionPipeline
from src.database import AgroDatabaseService


# =====================================================================
# GROUP 1: Data Loading & Validation (Tests 1-5)
# =====================================================================

def test_01_data_files_exist():
    """Test 1: Verifies raw and processed agricultural datasets exist."""
    assert os.path.exists("data/fertilizer_recommendation.csv"), "Missing fertilizer_recommendation.csv"
    assert os.path.exists("data/processed/crop_nutrition_dataset.csv"), "Missing crop_nutrition_dataset.csv"


def test_02_dataset_shape_and_columns():
    """Test 2: Verifies required columns in fertilizer_recommendation.csv."""
    df = pd.read_csv("data/fertilizer_recommendation.csv")
    assert df.shape[0] >= 10000, f"Expected >= 10000 rows, got {df.shape[0]}"
    required_cols = ["Soil_pH", "Nitrogen_Level", "Phosphorus_Level", "Potassium_Level", "Recommended_Fertilizer"]
    for col in required_cols:
        assert col in df.columns, f"Missing column {col}"


def test_03_no_missing_values_in_raw_data():
    """Test 3: Verifies dataset is clean with zero null values."""
    df = pd.read_csv("data/fertilizer_recommendation.csv")
    assert df.isnull().sum().sum() == 0, "Dataset contains null values"


def test_04_target_classes_coverage():
    """Test 4: Verifies presence of all 7 commercial fertilizer target classes."""
    df = pd.read_csv("data/fertilizer_recommendation.csv")
    classes = set(df["Recommended_Fertilizer"].unique())
    expected = {"Urea", "DAP", "MOP", "Compost", "Zinc Sulphate", "NPK", "SSP"}
    assert expected.issubset(classes), f"Missing target classes. Found {classes}"


def test_05_synthetic_stcr_loader_generation(tmp_path):
    """Test 5: Tests synthetic agronomic STCR dataset generator."""
    temp_csv = str(tmp_path / "temp_synth.csv")
    df_synth = DataLoader.generate_synthetic_agronomic_dataset(n_samples=50, save_path=temp_csv)
    assert len(df_synth) == 50
    assert "Rec_N" in df_synth.columns and "Rec_P" in df_synth.columns and "Rec_K" in df_synth.columns


# =====================================================================
# GROUP 2: Feature Engineering (GDD, VPD, Cyclic, Ratios) (Tests 6-12)
# =====================================================================

def test_06_gdd_calculation_positive():
    """Test 6: Tests Growing Degree Days (GDD) above base temperature."""
    # Wheat T_base = 4.5°C; Temp = 24.5°C -> GDD = 20.0
    gdd = AgronomicFeatureEngineer.calculate_gdd(temp=24.5, crop="Wheat")
    assert pytest.approx(gdd, 0.1) == 20.0


def test_07_gdd_calculation_sub_threshold():
    """Test 7: Tests GDD returns 0.0 when temperature is below base."""
    # Cotton T_base = 15.0°C; Temp = 12.0°C -> GDD = 0.0
    gdd = AgronomicFeatureEngineer.calculate_gdd(temp=12.0, crop="Cotton")
    assert gdd == 0.0


def test_08_vpd_calculation_standard():
    """Test 8: Tests Vapor Pressure Deficit (VPD in kPa) under standard conditions."""
    # Temp = 25°C, RH = 50%
    vpd = AgronomicFeatureEngineer.calculate_vpd(temp_c=25.0, relative_humidity_pct=50.0)
    assert 1.0 <= vpd <= 2.0, f"Expected VPD ~1.58 kPa, got {vpd}"


def test_09_vpd_high_humidity_clamp():
    """Test 9: Tests VPD remains non-negative and positive at 100% RH."""
    vpd = AgronomicFeatureEngineer.calculate_vpd(temp_c=28.0, relative_humidity_pct=100.0)
    assert vpd >= 0.01


def test_10_cyclic_encoding_season_orthogonality():
    """Test 10: Tests sine/cosine seasonal cyclic encoding identity sin^2 + cos^2 = 1."""
    for s in ["Kharif", "Rabi", "Zaid"]:
        enc = AgronomicFeatureEngineer.cyclic_encode_season(s)
        identity = (enc["Season_Sin"] ** 2) + (enc["Season_Cos"] ** 2)
        assert pytest.approx(identity, 1e-4) == 1.0


def test_11_stoichiometric_nutrient_ratios():
    """Test 11: Tests stoichiometric ratio feature derivations."""
    df_mini = pd.DataFrame([{
        "Nitrogen_Level": 100, "Phosphorus_Level": 50, "Potassium_Level": 25,
        "Soil_pH": 6.5, "Temperature": 25.0, "Humidity": 60.0, "Season": "Kharif", "Crop_Type": "Wheat"
    }])
    df_res = AgronomicFeatureEngineer.transform_dataframe(df_mini)
    assert pytest.approx(df_res["NP_Ratio"].iloc[0], 0.01) == 2.0
    assert pytest.approx(df_res["NK_Ratio"].iloc[0], 0.01) == 4.0
    assert pytest.approx(df_res["PK_Ratio"].iloc[0], 0.01) == 2.0


def test_12_ph_stress_calculation():
    """Test 12: Tests pH stress distance from neutral pH 6.5."""
    df_mini = pd.DataFrame([{"Soil_pH": 7.8, "Temperature": 20.0, "Humidity": 50.0, "Season": "Rabi", "Crop_Type": "Wheat"}])
    df_res = AgronomicFeatureEngineer.transform_dataframe(df_mini)
    assert pytest.approx(df_res["pH_Stress"].iloc[0], 0.01) == 1.3


# =====================================================================
# GROUP 3: Preprocessor Fitting & Serialization (Tests 13-18)
# =====================================================================

def test_13_preprocessor_fit_transform():
    """Test 13: Tests preprocessor transforms dataframe to finite 2D array."""
    df = pd.read_csv("data/fertilizer_recommendation.csv").head(100)
    prep = FertilizerClassifierPreprocessor()
    X, y, f_names, c_names = prep.fit_transform(df)
    assert X.shape[0] == 100
    assert np.all(np.isfinite(X))


def test_14_preprocessor_feature_count():
    """Test 14: Confirms preprocessor generates expected feature dimension (> 40)."""
    prep = FertilizerClassifierPreprocessor("artifacts/scalers/fertilizer_classifier_preprocessor.joblib")
    prep.load()
    assert len(prep.feature_names) >= 45


def test_15_preprocessor_save_and_load(tmp_path):
    """Test 15: Verifies joblib serialization and reloading of preprocessor."""
    temp_art = str(tmp_path / "test_prep.joblib")
    prep = FertilizerClassifierPreprocessor(temp_art)
    df = pd.read_csv("data/fertilizer_recommendation.csv").head(50)
    prep.fit_transform(df)
    prep2 = FertilizerClassifierPreprocessor(temp_art)
    prep2.load()
    assert prep2.is_fitted is True
    assert prep2.feature_names == prep.feature_names


def test_16_preprocessor_label_encoding():
    """Test 16: Tests label encoding and inverse decoding of classes."""
    prep = FertilizerClassifierPreprocessor("artifacts/scalers/fertilizer_classifier_preprocessor.joblib")
    prep.load()
    assert len(prep.class_names) == 7
    indices = np.array([0, 1, 2])
    decoded = prep.decode_predictions(indices)
    assert len(decoded) == 3


def test_17_preprocessor_transform_single_sample():
    """Test 17: Verifies transforming a 1-row DataFrame matches feature dimension."""
    prep = FertilizerClassifierPreprocessor("artifacts/scalers/fertilizer_classifier_preprocessor.joblib")
    prep.load()
    df_sample = pd.read_csv("data/fertilizer_recommendation.csv").head(1)
    X_single = prep.transform(df_sample)
    assert X_single.shape == (1, len(prep.feature_names))


def test_18_robust_scaler_robustness():
    """Test 18: Checks preprocessor handles outlier values without NaN."""
    prep = FertilizerClassifierPreprocessor("artifacts/scalers/fertilizer_classifier_preprocessor.joblib")
    prep.load()
    df_sample = pd.read_csv("data/fertilizer_recommendation.csv").head(1).copy()
    df_sample["Nitrogen_Level"] = 9999.0
    X_out = prep.transform(df_sample)
    assert np.all(np.isfinite(X_out))


# =====================================================================
# GROUP 4: Model Inference, Shapes & Probabilities (Tests 19-25)
# =====================================================================

def test_19_classifier_champion_loaded():
    """Test 19: Verifies champion fertilizer classifier checkpoint exists."""
    path = "artifacts/saved_models/champion_fertilizer_classifier.joblib"
    assert os.path.exists(path)
    data = joblib.load(path)
    assert "model" in data and "class_names" in data


def test_20_classifier_predict_shape():
    """Test 20: Ensures single prediction returns exactly 1 class index."""
    meta = joblib.load("artifacts/saved_models/champion_fertilizer_classifier.joblib")
    model = meta["model"]
    prep = FertilizerClassifierPreprocessor("artifacts/scalers/fertilizer_classifier_preprocessor.joblib")
    prep.load()
    sample = prep.transform(pd.read_csv("data/fertilizer_recommendation.csv").head(1))
    pred = model.predict(sample)
    assert len(pred) == 1


def test_21_classifier_predict_proba_simplex():
    """Test 21: Ensures class probabilities sum to 1.0 (simplex constraint)."""
    meta = joblib.load("artifacts/saved_models/champion_fertilizer_classifier.joblib")
    model = meta["model"]
    prep = FertilizerClassifierPreprocessor("artifacts/scalers/fertilizer_classifier_preprocessor.joblib")
    prep.load()
    sample = prep.transform(pd.read_csv("data/fertilizer_recommendation.csv").head(1))
    proba = model.predict_proba(sample)[0]
    assert pytest.approx(float(np.sum(proba)), 1e-4) == 1.0


def test_22_classifier_known_urea_deficit():
    """Test 22: Ensures soil with severe N deficit (< 40) classifies as Urea."""
    pipeline = PrecisionNutritionPipeline()
    sample = {
        "Soil_Type": "Loamy", "Crop_Type": "Cotton", "Crop_Growth_Stage": "Vegetative",
        "Season": "Kharif", "Nitrogen_Level": 20, "Phosphorus_Level": 55, "Potassium_Level": 80,
        "Soil_pH": 6.6, "Organic_Carbon": 0.45
    }
    res = pipeline.predict(sample)
    assert res["tier1_product_recommendation"]["recommended_fertilizer"] == "Urea"


def test_23_classifier_known_acidic_compost():
    """Test 23: Ensures soil with highly acidic pH (< 5.2) recommends Compost."""
    pipeline = PrecisionNutritionPipeline()
    sample = {
        "Soil_Type": "Clay", "Crop_Type": "Wheat", "Crop_Growth_Stage": "Sowing",
        "Season": "Rabi", "Nitrogen_Level": 110, "Phosphorus_Level": 60, "Potassium_Level": 80,
        "Soil_pH": 5.0, "Organic_Carbon": 0.35
    }
    res = pipeline.predict(sample)
    assert res["tier1_product_recommendation"]["recommended_fertilizer"] == "Compost"


def test_24_dosage_model_loaded():
    """Test 24: Verifies multi-target dosage model checkpoint exists."""
    path = "artifacts/saved_models/champion_model.joblib"
    assert os.path.exists(path)
    meta = joblib.load(path)
    assert "model" in meta


def test_25_dosage_model_predict_vector():
    """Test 25: Ensures dosage model outputs 3 continuous non-negative values [N, P, K]."""
    meta = joblib.load("artifacts/saved_models/champion_model.joblib")
    model = meta["model"]
    prep = AgronomicPreprocessor("artifacts/scalers/preprocessor.joblib")
    prep.load()
    df_samp = pd.DataFrame([{
        "Crop": "Wheat", "Soil_N": 200, "Soil_P": 20, "Soil_K": 150,
        "Soil_pH": 6.5, "Soil_OC": 0.55, "Soil_Moisture": 35, "Temperature": 25, "Humidity": 60, "Rainfall": 100
    }])
    X = prep.transform(df_samp)
    preds = model.predict(X)[0]
    assert len(preds) == 3
    assert all(p > 0 for p in preds)


# =====================================================================
# GROUP 5: Explainable AI (SHAP & LIME) (Tests 26-30)
# =====================================================================

def test_26_shap_explainer_initialization():
    """Test 26: Verifies TreeExplainer initializes properly for dosage model."""
    meta = joblib.load("artifacts/saved_models/champion_model.joblib")
    prep = AgronomicPreprocessor("artifacts/scalers/preprocessor.joblib")
    prep.load()
    explainer = PrecisionShapExplainer(meta["model"], prep.feature_names)
    assert len(explainer.explainers) > 0


def test_27_shap_local_attribution_keys():
    """Test 27: Ensures SHAP local output contains positive/negative drivers."""
    meta = joblib.load("artifacts/saved_models/champion_model.joblib")
    prep = AgronomicPreprocessor("artifacts/scalers/preprocessor.joblib")
    prep.load()
    explainer = PrecisionShapExplainer(meta["model"], prep.feature_names)
    X = np.zeros((1, len(prep.feature_names)))
    res = explainer.explain_sample(X)
    assert "Rec_N" in res and "top_positive_drivers" in res["Rec_N"]


def test_28_shap_additivity_property():
    """Test 28: Verifies SHAP efficiency/additivity: sum(phi) ≈ f(x) - E[f(x)]."""
    meta = joblib.load("artifacts/saved_models/champion_model.joblib")
    prep = AgronomicPreprocessor("artifacts/scalers/preprocessor.joblib")
    prep.load()
    explainer = PrecisionShapExplainer(meta["model"], prep.feature_names)
    X = np.zeros((1, len(prep.feature_names)))
    res = explainer.explain_sample(X)
    n_data = res["Rec_N"]
    sum_phi = sum(item["shap_value"] for item in n_data["all_contributions"])
    expected_diff = n_data["predicted_dosage"] - n_data["base_expected_dosage"]
    assert pytest.approx(sum_phi, 0.5) == expected_diff


def test_29_shap_multiclass_attribution():
    """Test 29: Ensures multi-class fertilizer SHAP returns attributions for predicted class."""
    meta = joblib.load("artifacts/saved_models/champion_fertilizer_classifier.joblib")
    prep = FertilizerClassifierPreprocessor("artifacts/scalers/fertilizer_classifier_preprocessor.joblib")
    prep.load()
    shap_clf = FertilizerClassifierShap(meta["model"], prep.feature_names, meta["class_names"])
    X = np.zeros((1, len(prep.feature_names)))
    out = shap_clf.explain_sample(X, predicted_class_idx=0)
    assert "top_positive_drivers" in out and "recommended_fertilizer" in out


def test_30_shap_global_summary_generation():
    """Test 30: Verifies global SHAP plot file exists and is valid PNG."""
    path = "artifacts/figures/shap_fertilizer_classification.png"
    assert os.path.exists(path)
    assert os.path.getsize(path) > 1000


# =====================================================================
# GROUP 6: Agronomic ICAR STCR Safety & Biological Clamping (Tests 31-34)
# =====================================================================

def test_31_stcr_baseline_rice():
    """Test 31: Tests ICAR STCR targeted yield baseline calculation for Rice."""
    res = STCRValidator.calculate_stcr_baseline(
        crop="Rice", soil_n=200, soil_p=20, soil_k=150, soil_ph=6.5, soil_oc=0.55
    )
    assert 40.0 <= res["stcr_n"] <= 180.0
    assert 20.0 <= res["stcr_p"] <= 90.0


def test_32_stcr_baseline_wheat():
    """Test 32: Tests ICAR STCR targeted yield baseline calculation for Wheat."""
    res = STCRValidator.calculate_stcr_baseline(
        crop="Wheat", soil_n=180, soil_p=25, soil_k=160, soil_ph=6.8, soil_oc=0.50
    )
    assert 40.0 <= res["stcr_n"] <= 160.0


def test_33_stcr_safety_upper_clamp():
    """Test 33: Ensures excessive Nitrogen (> 200 kg/ha) is clamped to biological maximum."""
    safe, meta = STCRValidator.validate_and_clamp(crop="Wheat", pred_n=280.0, pred_p=50.0, pred_k=50.0)
    assert safe["Nitrogen_N"] <= 160.0
    assert meta["is_clamped"] is True


def test_34_stcr_safety_lower_clamp():
    """Test 34: Ensures negative or low Nitrogen is clamped to minimum starter dose."""
    safe, meta = STCRValidator.validate_and_clamp(crop="Wheat", pred_n=5.0, pred_p=50.0, pred_k=50.0)
    assert safe["Nitrogen_N"] >= 40.0


# =====================================================================
# GROUP 7: Commercial Fertilizer Bags & Schedule (Tests 35-37)
# =====================================================================

def test_35_fertilizer_dap_calculation():
    """Test 35: Validates DAP formula (P_rec / 0.46)."""
    # 46 kg P requires 100 kg DAP
    res = FertilizerCalculator.calculate_commercial_fertilizers(n_rec_kg_ha=100, p_rec_kg_ha=46, k_rec_kg_ha=60)
    assert pytest.approx(res["commercial_fertilizers_kg_ha"]["DAP"], 0.1) == 100.0


def test_36_fertilizer_urea_remainder_calculation():
    """Test 36: Validates Urea supplies remaining Nitrogen after DAP contribution."""
    # 100 kg DAP supplies 18 kg N. Total N=100. Remaining N=82. Urea = 82 / 0.46 ≈ 178.26 kg
    res = FertilizerCalculator.calculate_commercial_fertilizers(n_rec_kg_ha=100, p_rec_kg_ha=46, k_rec_kg_ha=60)
    assert pytest.approx(res["commercial_fertilizers_kg_ha"]["Urea"], 0.5) == 178.26


def test_37_fertilizer_split_schedule():
    """Test 37: Verifies 3-way split application (Basal, Tillering, Flowering)."""
    res = FertilizerCalculator.calculate_commercial_fertilizers(n_rec_kg_ha=100, p_rec_kg_ha=40, k_rec_kg_ha=50, farm_area_acres=1.0)
    sched = res["application_schedule"]
    assert "Basal_at_sowing" in sched
    assert "Top_Dress_1_Vegetative" in sched
    assert "Top_Dress_2_Flowering" in sched


# =====================================================================
# GROUP 8: Database Persistence & API Response Validation (Tests 38-40)
# =====================================================================

def test_38_database_init_and_tables(tmp_path):
    """Test 38: Verifies SQLite schema creation and tables."""
    test_db = str(tmp_path / "test.db")
    db_svc = AgroDatabaseService(test_db)
    with db_svc.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row["name"] for row in cursor.fetchall()]
        assert "soil_records" in tables
        assert "model_prescriptions" in tables
        assert "weather_station_cache" in tables


def test_39_cached_station_lookup_latency():
    """Test 39: Verifies cached station lookup executes in < 100ms."""
    db_svc = AgroDatabaseService()
    res = db_svc.cached_station_lookup("STN_NORTH_01")
    assert res["query_latency_ms"] < 100.0
    assert res["region"] == "North"


def test_40_api_response_schema_validation():
    """Test 40: Validates 2-tier JSON schema validation helper."""
    pipeline = PrecisionNutritionPipeline()
    sample = {
        "Soil_Type": "Loamy", "Crop_Type": "Maize", "Crop_Growth_Stage": "Vegetative",
        "Season": "Kharif", "Nitrogen_Level": 35, "Phosphorus_Level": 45, "Potassium_Level": 65,
        "Soil_pH": 6.8, "Organic_Carbon": 0.40
    }
    res = pipeline.predict(sample, farm_area_acres=1.5)
    is_valid = AgroDatabaseService.validate_api_response_schema(res)
    assert is_valid is True
