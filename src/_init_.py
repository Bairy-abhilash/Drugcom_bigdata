"""
AI-Powered Drug Combination Therapy Application
================================================

A complete system for predicting drug synergy using Graph Neural Networks,
with safety checking and confidence estimation.

Modules:
    - data: Data loading and processing
    - preprocessing: SMILES processing and feature engineering
    - graph: Graph construction using DGL
    - models: GNN models for synergy prediction
    - utils: Safety checking and metrics
    - inference: Prediction pipeline
"""

__version__ = "1.0.0"
__author__ = "Drug Combo Therapy Team"
__license__ = "MIT"

# Core imports for easy access
from src.data.drugcomb_loader import DrugCombLoader
from src.data.drugbank_loader import DrugBankLoader
from src.preprocessing.smiles_processor import SMILESProcessor
from src.preprocessing.feature_engineering import FeatureEngineer
from src.graph.graph_constructor import GraphConstructor
from src.graph.heterograph_builder import HeteroGraphBuilder
from src.models.gnn_model import GNNModel
from src.models.trainer import ModelTrainer
from src.models.confidence_estimator import ConfidenceEstimator
from src.utils.safety_checker import SafetyChecker
from src.utils.metrics import calculate_metrics
from src.inference.predictor import SynergyPredictor

__all__ = [
    # Data loaders
    "DrugCombLoader",
    "DrugBankLoader",
    
    # Preprocessing
    "SMILESProcessor",
    "FeatureEngineer",
    
    # Graph construction
    "GraphConstructor",
    "HeteroGraphBuilder",
    
    # Models
    "GNNModel",
    "ModelTrainer",
    "ConfidenceEstimator",
    
    # Utils
    "SafetyChecker",
    "calculate_metrics",
    
    # Inference
    "SynergyPredictor",
]


def get_version() -> str:
    """Return the current version of the package."""
    return __version__


def setup_logging(level: str = "INFO") -> None:
    """
    Configure logging for the entire package.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    import logging
    import sys
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('drug_combo_therapy.log')
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Drug Combo Therapy v{__version__} initialized")


# Initialize logging by default
setup_logging()