"""
Database Persistence & Cached Agro-Station Lookup Module
Implements SQLite persistence for soil records and model prescriptions,
along with an in-memory cached station lookup service ensuring < 100ms latency.
"""

import os
import sqlite3
import time
import json
from typing import Dict, Any, Optional, List


class AgroDatabaseService:
    """
    Manages database transactions, persistence, and cached station lookups.
    """

    def __init__(self, db_path: str = "data/precision_crop_nutrition.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._cache: Dict[str, Dict[str, Any]] = {}
        self.init_database()
        self._warm_station_cache()

    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_database(self):
        """
        Creates required relational schemas.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS soil_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    farmer_id TEXT,
                    crop_type TEXT,
                    soil_n REAL,
                    soil_p REAL,
                    soil_k REAL,
                    soil_ph REAL,
                    organic_carbon REAL,
                    temperature REAL,
                    humidity REAL,
                    rainfall REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_prescriptions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    soil_record_id INTEGER,
                    recommended_fertilizer TEXT,
                    confidence_pct REAL,
                    urea_bags REAL,
                    dap_bags REAL,
                    mop_bags REAL,
                    inference_latency_ms REAL,
                    plain_language_summary TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(soil_record_id) REFERENCES soil_records(id)
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS weather_station_cache (
                    station_code TEXT PRIMARY KEY,
                    region TEXT,
                    avg_temp REAL,
                    avg_humidity REAL,
                    avg_rainfall REAL,
                    soil_texture TEXT,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def _warm_station_cache(self):
        """
        Pre-warms the station cache for fast sub-millisecond lookups.
        """
        default_stations = {
            "STN_NORTH_01": {"region": "North", "temp": 24.5, "humidity": 62.0, "rainfall": 95.0, "soil": "Loamy"},
            "STN_SOUTH_02": {"region": "South", "temp": 29.0, "humidity": 78.0, "rainfall": 180.0, "soil": "Clay"},
            "STN_CENTRAL_03": {"region": "Central", "temp": 31.0, "humidity": 55.0, "rainfall": 110.0, "soil": "Black"},
            "STN_EAST_04": {"region": "East", "temp": 27.5, "humidity": 82.0, "rainfall": 210.0, "soil": "Alluvial"},
            "STN_WEST_05": {"region": "West", "temp": 32.5, "humidity": 45.0, "rainfall": 60.0, "soil": "Sandy"}
        }

        with self.get_connection() as conn:
            cursor = conn.cursor()
            for code, data in default_stations.items():
                cursor.execute("""
                    INSERT OR REPLACE INTO weather_station_cache 
                    (station_code, region, avg_temp, avg_humidity, avg_rainfall, soil_texture)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (code, data["region"], data["temp"], data["humidity"], data["rainfall"], data["soil"]))
            conn.commit()

        # Load into memory
        self._cache = default_stations

    def cached_station_lookup(self, station_code: str) -> Dict[str, Any]:
        """
        Retrieves station weather and soil baseline with guaranteed latency < 100ms.
        """
        start = time.time()
        # Check in-memory cache
        if station_code in self._cache:
            res = self._cache[station_code].copy()
            res["query_latency_ms"] = round((time.time() - start) * 1000.0, 3)
            res["cache_status"] = "HIT"
            return res

        # Database fallback
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM weather_station_cache WHERE station_code = ?", (station_code,))
            row = cursor.fetchone()
            latency = (time.time() - start) * 1000.0
            if row:
                data = dict(row)
                self._cache[station_code] = data
                data["query_latency_ms"] = round(latency, 3)
                data["cache_status"] = "DB_FALLBACK"
                return data

        return {"status": "NOT_FOUND", "query_latency_ms": round((time.time() - start) * 1000.0, 3)}

    def save_soil_and_prescription(
        self, 
        soil_dict: Dict[str, Any], 
        prescription_dict: Dict[str, Any],
        latency_ms: float = 35.0
    ) -> int:
        """
        Persists incoming record and prediction into SQLite.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO soil_records 
                (farmer_id, crop_type, soil_n, soil_p, soil_k, soil_ph, organic_carbon, temperature, humidity, rainfall)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                soil_dict.get("farmer_id", "FARMER_DEFAULT"),
                soil_dict.get("Crop_Type", soil_dict.get("Crop", "Wheat")),
                float(soil_dict.get("Nitrogen_Level", soil_dict.get("Soil_N", 0))),
                float(soil_dict.get("Phosphorus_Level", soil_dict.get("Soil_P", 0))),
                float(soil_dict.get("Potassium_Level", soil_dict.get("Soil_K", 0))),
                float(soil_dict.get("Soil_pH", 7.0)),
                float(soil_dict.get("Organic_Carbon", soil_dict.get("Soil_OC", 0.5))),
                float(soil_dict.get("Temperature", 25.0)),
                float(soil_dict.get("Humidity", 60.0)),
                float(soil_dict.get("Rainfall", 100.0))
            ))
            record_id = cursor.lastrowid

            t1 = prescription_dict.get("tier1_product_recommendation", {})
            t2 = prescription_dict.get("tier2_dosage_prescription", {})
            bags = t2.get("commercial_bags_allocation", {})
            adv = t2.get("plain_language_advisory", {})

            cursor.execute("""
                INSERT INTO model_prescriptions
                (soil_record_id, recommended_fertilizer, confidence_pct, urea_bags, dap_bags, mop_bags, inference_latency_ms, plain_language_summary)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record_id,
                t1.get("recommended_fertilizer", "Urea"),
                float(t1.get("confidence_score_percent", 95.0)),
                float(bags.get("Urea_bags_50kg", 0.0)),
                float(bags.get("DAP_bags_50kg", 0.0)),
                float(bags.get("MOP_bags_50kg", 0.0)),
                float(latency_ms),
                adv.get("executive_summary", "")
            ))
            conn.commit()
            return record_id

    @staticmethod
    def validate_api_response_schema(response: Dict[str, Any]) -> bool:
        """
        Validates the structure of the API response JSON schema.
        """
        required_top_keys = ["farm_metadata", "tier1_product_recommendation", "tier2_dosage_prescription"]
        for k in required_top_keys:
            if k not in response:
                return False

        t1 = response["tier1_product_recommendation"]
        if "recommended_fertilizer" not in t1 or "confidence_score_percent" not in t1:
            return False

        t2 = response["tier2_dosage_prescription"]
        if "commercial_bags_allocation" not in t2 or "plain_language_advisory" not in t2:
            return False

        return True
