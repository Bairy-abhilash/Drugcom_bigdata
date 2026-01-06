"""
ETL (Extract, Transform, Load) module.

Scripts for loading data from various sources into PostgreSQL database.
"""

from app.etl.load_drugs import DrugLoader
from app.etl.load_targets import TargetLoader
from app.etl.load_diseases import DiseaseLoader
from app.etl.load_synergy import SynergyLoader

__all__ = [
    'DrugLoader',
    'TargetLoader',
    'DiseaseLoader',
    'SynergyLoader'
]