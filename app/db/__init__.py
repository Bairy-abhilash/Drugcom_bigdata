"""
Database module.

Handles PostgreSQL database connections, models, and queries.
"""

from app.db.connection import db_manager, get_db, init_database
from app.db.models import (
    Drug,
    Target,
    Disease,
    DrugTarget,
    TargetDisease,
    CellLine,
    SynergyScore,
    SideEffect,
    HarmfulCombination,
    PredictionCache
)

__all__ = [
    'db_manager',
    'get_db',
    'init_database',
    'Drug',
    'Target',
    'Disease',
    'DrugTarget',
    'TargetDisease',
    'CellLine',
    'SynergyScore',
    'SideEffect',
    'HarmfulCombination',
    'PredictionCache'
]