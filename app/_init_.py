"""
API Module
==========

FastAPI-based REST API for drug synergy prediction service.

Endpoints:
    - POST /predict-synergy: Predict synergy for drug pairs
    - GET /drug-info/{drug_id}: Get drug information
    - GET /disease-list: List available diseases
    - GET /health: Health check endpoint
"""

from api.main import app, get_application

__all__ = [
    "app",
    "get_application",
]

__version__ = "1.0.0"


# API metadata
API_METADATA = {
    "title": "Drug Combination Therapy API",
    "description": (
        "REST API for predicting drug synergy using Graph Neural Networks. "
        "Provides synergy predictions, confidence estimates, and safety checks."
    ),
    "version": __version__,
    "contact": {
        "name": "Drug Combo Therapy Team",
        "email": "support@drugcombotherapy.com"
    },
    "license_info": {
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT"
    }
}


# Default API configuration
DEFAULT_API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "reload": False,
    "workers": 4,
    "log_level": "info",
    "cors_origins": ["*"],
    "max_request_size": 10 * 1024 * 1024,  # 10MB
    "timeout": 60
}


def get_api_config() -> dict:
    """
    Get default API configuration.
    
    Returns:
        Dictionary with API settings
    """
    return DEFAULT_API_CONFIG.copy()


def get_api_metadata() -> dict:
    """
    Get API metadata for documentation.
    
    Returns:
        Dictionary with API metadata
    """
    return API_METADATA.copy()