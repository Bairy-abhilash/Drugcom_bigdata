"""
Models Module
=============

This module contains all neural network models for drug synergy prediction,
including GNN architectures, training utilities, and confidence estimation.

Components:
    - GNNModel: Main Graph Neural Network model (GCN/GraphSAGE)
    - ModelTrainer: Training and validation loop
    - ConfidenceEstimator: MC Dropout-based confidence estimation
"""

from src.models.gnn_model import (
    GNNModel,
    GCNSynergyModel,
    GraphSAGESynergyModel,
    HeteroGNNModel
)
from src.models.trainer import (
    ModelTrainer,
    TrainingConfig,
    EarlyStopping
)
from src.models.confidence_estimator import (
    ConfidenceEstimator,
    MCDropout,
    EnsembleConfidence
)

__all__ = [
    # Base models
    "GNNModel",
    "GCNSynergyModel",
    "GraphSAGESynergyModel",
    "HeteroGNNModel",
    
    # Training
    "ModelTrainer",
    "TrainingConfig",
    "EarlyStopping",
    
    # Confidence estimation
    "ConfidenceEstimator",
    "MCDropout",
    "EnsembleConfidence",
]


# Model registry for easy model selection
MODEL_REGISTRY = {
    "gcn": GCNSynergyModel,
    "graphsage": GraphSAGESynergyModel,
    "hetero_gnn": HeteroGNNModel,
}


def get_model(model_name: str, **kwargs):
    """
    Factory function to get a model by name.
    
    Args:
        model_name: Name of the model ('gcn', 'graphsage', 'hetero_gnn')
        **kwargs: Model-specific arguments
        
    Returns:
        Initialized model instance
        
    Example:
        >>> model = get_model('gcn', in_feats=512, hidden_feats=256, num_layers=3)
    """
    if model_name.lower() not in MODEL_REGISTRY:
        raise ValueError(
            f"Model '{model_name}' not found. "
            f"Available models: {list(MODEL_REGISTRY.keys())}"
        )
    
    return MODEL_REGISTRY[model_name.lower()](**kwargs)


def load_pretrained_model(checkpoint_path: str, model_name: str = "gcn", **kwargs):
    """
    Load a pretrained model from checkpoint.
    
    Args:
        checkpoint_path: Path to the model checkpoint (.pth file)
        model_name: Type of model to load
        **kwargs: Additional model arguments
        
    Returns:
        Loaded model with pretrained weights
        
    Example:
        >>> model = load_pretrained_model('models/checkpoints/best_model.pth', 'gcn')
    """
    import torch
    
    model = get_model(model_name, **kwargs)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model


# Default model configurations
DEFAULT_CONFIGS = {
    "gcn": {
        "in_feats": 512,
        "hidden_feats": 256,
        "num_layers": 3,
        "dropout": 0.3,
        "activation": "relu",
    },
    "graphsage": {
        "in_feats": 512,
        "hidden_feats": 256,
        "num_layers": 3,
        "dropout": 0.3,
        "aggregator_type": "mean",
    },
    "hetero_gnn": {
        "in_feats": {"drug": 512, "target": 256, "disease": 128},
        "hidden_feats": 256,
        "num_layers": 3,
        "dropout": 0.3,
    }
}


def get_default_config(model_name: str) -> dict:
    """
    Get default configuration for a specific model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Dictionary with default configuration parameters
    """
    if model_name.lower() not in DEFAULT_CONFIGS:
        raise ValueError(f"No default config for model '{model_name}'")
    
    return DEFAULT_CONFIGS[model_name.lower()].copy()


# Model metadata
MODEL_METADATA = {
    "gcn": {
        "name": "Graph Convolutional Network",
        "description": "GCN-based synergy prediction model",
        "paper": "Semi-Supervised Classification with Graph Convolutional Networks",
        "year": 2017,
    },
    "graphsage": {
        "name": "GraphSAGE",
        "description": "Inductive representation learning on large graphs",
        "paper": "Inductive Representation Learning on Large Graphs",
        "year": 2017,
    },
    "hetero_gnn": {
        "name": "Heterogeneous GNN",
        "description": "GNN for heterogeneous drug-target-disease graphs",
        "paper": "Custom implementation for drug synergy",
        "year": 2024,
    }
}


def get_model_info(model_name: str) -> dict:
    """
    Get metadata information about a model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Dictionary with model metadata
    """
    if model_name.lower() not in MODEL_METADATA:
        raise ValueError(f"No metadata for model '{model_name}'")
    
    return MODEL_METADATA[model_name.lower()].copy()