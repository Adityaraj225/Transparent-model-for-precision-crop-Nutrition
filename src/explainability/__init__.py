"""
Explainable AI (XAI) package: SHAP, LIME, and Natural Language Translation Layer.
"""
from .shap_explainer import PrecisionShapExplainer
from .lime_explainer import PrecisionLimeExplainer
from .nlg_translator import AgronomicNLGTranslator
from .fertilizer_classifier_shap import FertilizerClassifierShap

__all__ = [
    "PrecisionShapExplainer",
    "PrecisionLimeExplainer",
    "AgronomicNLGTranslator",
    "FertilizerClassifierShap"
]
