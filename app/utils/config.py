"""
Configuration management module.
Loads environment variables and YAML config files.
"""

import os
from pathlib import Path
from typing import Any, Dict
import yaml
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Load environment variables
load_dotenv()

class Settings(BaseSettings):
    """Application settings from environment variables."""
    
    # Database
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_NAME: str = "drug_synergy"
    DB_USER: str = "postgres"
    DB_PASSWORD: str = "postgres"
    
    # Application
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    DEBUG: bool = True
    
    # Model
    MODEL_PATH: str = "models/gnn_synergy_model.pth"
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 0.001
    EPOCHS: int = 100
    HIDDEN_DIM: int = 128
    NUM_LAYERS: int = 3
    DROPOUT: float = 0.3
    
    # Features
    DRUG_FEATURE_DIM: int = 2048
    DISEASE_FEATURE_DIM: int = 64
    
    # API
    MAX_PREDICTIONS: int = 100
    TOP_K_RESULTS: int = 10
    
    class Config:
        env_file = ".env"
        case_sensitive = True

    @property
    def database_url(self) -> str:
        """Generate database URL."""
        return f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"


class ConfigLoader:
    """Load YAML configuration files."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = Path(config_path)
        self._config: Dict[str, Any] = {}
        self.load()
    
    def load(self) -> None:
        """Load configuration from YAML file."""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                self._config = yaml.safe_load(f)
        else:
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key (supports dot notation)."""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value


# Global instances
settings = Settings()
config = ConfigLoader()