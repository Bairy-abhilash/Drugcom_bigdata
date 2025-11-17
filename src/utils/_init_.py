"""
Utilities Module
================

Common utilities for the drug combination therapy application:
- Safety checking
- Metrics calculation
- Visualization helpers
- Configuration management
"""

from src.utils.safety_checker import (
    SafetyChecker,
    InteractionChecker,
    SafetyLevel,
    DrugInteraction
)
from src.utils.metrics import (
    calculate_metrics,
    calculate_mse,
    calculate_mae,
    calculate_rmse,
    calculate_pearson_correlation,
    calculate_spearman_correlation,
    calculate_r2_score,
    MetricsCalculator,
    SynergyMetrics
)

__all__ = [
    # Safety checking
    "SafetyChecker",
    "InteractionChecker",
    "SafetyLevel",
    "DrugInteraction",
    
    # Metrics
    "calculate_metrics",
    "calculate_mse",
    "calculate_mae",
    "calculate_rmse",
    "calculate_pearson_correlation",
    "calculate_spearman_correlation",
    "calculate_r2_score",
    "MetricsCalculator",
    "SynergyMetrics",
]


# Default safety thresholds
DEFAULT_SAFETY_THRESHOLDS = {
    "high_risk_score": 0.8,
    "medium_risk_score": 0.5,
    "low_risk_score": 0.2
}


# Default metric configurations
DEFAULT_METRIC_CONFIG = {
    "primary_metric": "pearson_correlation",
    "regression_metrics": ["mse", "mae", "rmse", "r2", "pearson", "spearman"],
    "classification_metrics": ["accuracy", "precision", "recall", "f1", "auc"]
}


def get_safety_thresholds() -> dict:
    """
    Get default safety threshold configuration.
    
    Returns:
        Dictionary with safety thresholds
    """
    return DEFAULT_SAFETY_THRESHOLDS.copy()


def get_metric_config() -> dict:
    """
    Get default metrics configuration.
    
    Returns:
        Dictionary with metric settings
    """
    return DEFAULT_METRIC_CONFIG.copy()