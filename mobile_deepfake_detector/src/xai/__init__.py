"""
Explainable AI (XAI) modules
"""

# Note: Only PIAExplainer and PIAVisualizer are currently implemented
# Other XAI modules (GradCAM, SHAP, LIME) are planned for future implementation

from .pia_explainer import PIAExplainer
from .pia_visualizer import PIAVisualizer

# Hybrid Pipeline modules (refactored from hybrid_mmms_pia_explainer.py)
from .hybrid_pipeline import HybridXAIPipeline
from .stage1_scanner import Stage1Scanner
from .stage2_analyzer import Stage2Analyzer
from .unified_feature_extractor import UnifiedFeatureExtractor
from .result_aggregator import ResultAggregator

__all__ = [
    "PIAExplainer",
    "PIAVisualizer",
    # Hybrid Pipeline modules
    "HybridXAIPipeline",
    "Stage1Scanner",
    "Stage2Analyzer",
    "UnifiedFeatureExtractor",
    "ResultAggregator",
]
