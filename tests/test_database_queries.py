"""
Unit tests for database queries.
"""

import pytest
from sqlalchemy.orm import Session

from app.db.connection import db_manager
from app.db.queries.drug_queries import DrugQueries
from app.db.queries.disease_queries import DiseaseQueries
from app.db.queries.synergy_queries import SynergyQueries


class TestDatabaseQueries:
    """Test database query functions."""
    
    @pytest.fixture
    def db_session(self):
        """Database session fixture."""
        with db_manager.session_scope() as session:
            yield session
    
    def test_get_all_drugs(self, db_session):
        """Test getting all drugs."""
        drug_queries = DrugQueries()
        drugs = drug_queries.get_all_drugs(db_session, limit=10)
        
        assert isinstance(drugs, list)
    
    def test_search_drugs(self, db_session):
        """Test drug search."""
        drug_queries = DrugQueries()
        results = drug_queries.search_drugs(db_session, "doxorubicin", limit=5)
        
        assert isinstance(results, list)
    
    def test_get_all_diseases(self, db_session):
        """Test getting all diseases."""
        disease_queries = DiseaseQueries()
        diseases = disease_queries.get_all_diseases(db_session)
        
        assert isinstance(diseases, list)
    
    def test_check_harmful_combination(self, db_session):
        """Test harmful combination check."""
        synergy_queries = SynergyQueries()
        result = synergy_queries.check_harmful_combination(db_session, 1, 2)
        
        # Should return None or HarmfulCombination object
        assert result is None or hasattr(result, 'combination_id')


if __name__ == '__main__':
    pytest.main([__file__])