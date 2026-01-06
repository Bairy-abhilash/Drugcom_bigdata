"""
Drug Synergy Prediction Application.

A complete platform for predicting drug combination synergy using
Graph Neural Networks with PostgreSQL database backend.
"""

__version__ = "1.0.0"
__author__ = "Drug Synergy Team"
__description__ = "AI-Powered Drug Combination Therapy Platform"

from app.utils.logger import setup_logger

logger = setup_logger(__name__)
logger.info(f"Initializing Drug Synergy Application v{__version__}")