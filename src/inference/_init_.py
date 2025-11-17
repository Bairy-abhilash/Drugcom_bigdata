"""
Inference Module
================

End-to-end inference pipeline for drug synergy prediction with safety checking
and confidence estimation.
"""

from src.inference.predictor import (
    SynergyPredictor,
    PredictionResult,
    BatchPredictor,
    InferencePipeline
)

__all__ = [
    "SynergyPredictor",
    "PredictionResult",
    "BatchPredictor",
    "InferencePipeline",
]


# Default inference configuration
DEFAULT_INFERENCE_CONFIG = {
    "batch_size": 32,
    "mc_iterations": 50,
    "confidence_threshold": 70.0,
    "safety_check": True,
    "return_top_k": 10,
    "device": "cpu"
}


def get_default_inference_config() -> dict:
    """
    Get default inference configuration.
    
    Returns:
        Dictionary with default settings
    """
    return DEFAULT_INFERENCE_CONFIG.copy()


def create_predictor(
    model_path: str,
    graph_path: str,
    config: dict = None
) -> "SynergyPredictor":
    """
    Factory function to create a configured predictor.
    
    Args:
        model_path: Path to trained model checkpoint
        graph_path: Path to constructed graph
        config: Optional configuration overrides
        
    Returns:
        Configured SynergyPredictor instance
        
    Example:
        >>> predictor = create_predictor(
        ...     'models/checkpoints/best_model.pth',
        ...     'data/processed/graphs/hetero_graph.bin'
        ... )
    """
    if config is None:
        config = DEFAULT_INFERENCE_CONFIG
    else:
        # Merge with defaults
        full_config = DEFAULT_INFERENCE_CONFIG.copy()
        full_config.update(config)
        config = full_config
    
    predictor = SynergyPredictor(
        model_path=model_path,
        graph_path=graph_path,
        **config
    )
    
    return predictor