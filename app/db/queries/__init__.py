"""
Database query modules.

Provides query interfaces for drugs, diseases, targets, and synergy data.
"""

from app.db.queries.drug_queries import DrugQueries
from app.db.queries.disease_queries import DiseaseQueries
from app.db.queries.synergy_queries import SynergyQueries

__all__ = [
    'DrugQueries',
    'DiseaseQueries',
    'SynergyQueries'
]