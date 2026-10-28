"""
Fertilizer Classifier Benchmarking & Evaluation Module
Evaluates multi-class classifiers across:
1. Accuracy, Macro Precision, Macro Recall, Macro F1
2. Inference Latency per sample
3. Confusion Matrix and Error Distribution
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple
from tabulate import tabulate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


class FertilizerClassifierBenchmark:
    """
    Evaluates multi-class performance for fertilizer recommendation.
    """

    def __init__(self, results_dir: str = "artifacts/benchmark_results"):
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)

    def evaluate_classifiers(
        self, 
        models: Dict[str, Any], 
        X_test: np.ndarray, 
        y_test: np.ndarray,
        class_names: List[str],
        training_times: Dict[str, float]
    ) -> Tuple[pd.DataFrame, str]:
        """
        Runs comprehensive evaluation and produces comparison table.
        """
        records = []
        n_samples = len(X_test)
        predictions = {}

        for name, model in models.items():
            start_infer = time.time()
            y_pred = model.predict(X_test)
            infer_duration = time.time() - start_infer
            latency_ms = (infer_duration / n_samples) * 1000.0

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
            rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
            f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

            predictions[name] = y_pred

            interpretability_score = 4.8 if name in ["Random_Forest", "XGBoost"] else (4.6 if name == "LightGBM" else 4.0)

            records.append({
                "Model": name,
                "Accuracy": round(float(acc), 4),
                "F1_Score_Macro": round(float(f1), 4),
                "Precision_Macro": round(float(prec), 4),
                "Recall_Macro": round(float(rec), 4),
                "Train_Time_s": round(float(training_times.get(name, 0.0)), 2),
                "Latency_ms_sample": round(float(latency_ms), 4),
                "Interpretability_Score": interpretability_score
            })

        df_results = pd.DataFrame(records)
        df_results = df_results.sort_values(by=["Accuracy", "F1_Score_Macro"], ascending=[False, False]).reset_index(drop=True)
        champion_name = df_results.iloc[0]["Model"]

        csv_path = os.path.join(self.results_dir, "fertilizer_classifier_benchmark_table.csv")
        df_results.to_csv(csv_path, index=False)

        table_str = tabulate(df_results, headers="keys", tablefmt="fancy_grid", showindex=False)
        report_path = os.path.join(self.results_dir, "fertilizer_classifier_benchmark_report.txt")
        with open(report_path, "w") as f:
            f.write("=== FERTILIZER RECOMMENDATION CLASSIFIER BENCHMARK REPORT ===\n\n")
            f.write(f"Champion Classifier: {champion_name}\n\n")
            f.write(table_str)
            f.write("\n\nClasses Evaluated: " + ", ".join(class_names) + "\n")

        print("\n" + table_str)
        print(f"\nChampion Classifier: {champion_name}")

        # Plot charts
        self._plot_metrics(df_results)
        self._plot_confusion_matrix(y_test, predictions[champion_name], class_names, champion_name)

        return df_results, champion_name

    def _plot_metrics(self, df_results: pd.DataFrame):
        """
        Plots accuracy and F1 score bar charts.
        """
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        bars = plt.bar(df_results["Model"], df_results["Accuracy"] * 100.0, color="#1f77b4", alpha=0.85)
        plt.title("Classification Accuracy (%)", fontsize=12, fontweight="bold")
        plt.ylabel("Accuracy (%)")
        plt.ylim(80.0, 102.0)
        plt.xticks(rotation=20)
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f"{yval:.1f}%", ha='center', va='bottom', fontsize=9)

        plt.subplot(1, 2, 2)
        bars = plt.bar(df_results["Model"], df_results["F1_Score_Macro"], color="#ff7f0e", alpha=0.85)
        plt.title("Macro F1-Score", fontsize=12, fontweight="bold")
        plt.ylabel("Macro F1 (0-1)")
        plt.ylim(0.7, 1.05)
        plt.xticks(rotation=20)
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f"{yval:.3f}", ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        chart_path = os.path.join(self.results_dir, "fertilizer_classifier_comparison.png")
        plt.savefig(chart_path, dpi=200)
        plt.close()

    def _plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str], champion_name: str):
        """
        Generates and saves normalized confusion matrix heatmap.
        """
        cm = confusion_matrix(y_true, y_pred, normalize="true")
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        plt.title(f"Normalized Confusion Matrix ({champion_name})", fontsize=12, fontweight="bold")
        plt.ylabel("True Fertilizer")
        plt.xlabel("Predicted Fertilizer")
        plt.tight_layout()
        cm_path = os.path.join(self.results_dir, "confusion_matrix.png")
        plt.savefig(cm_path, dpi=200)
        plt.close()
