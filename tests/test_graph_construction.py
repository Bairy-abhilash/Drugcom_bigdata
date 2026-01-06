"""
Unit tests for graph construction.
"""

import pytest
import torch
import dgl
from sqlalchemy.orm import Session

from app.graph.builder import HeterogeneousGraphBuilder
from app.graph.features import FeatureGenerator
from app.db.connection import db_manager
from app.db.models import Drug, Target, Disease


class TestGraphConstruction:
    """Test graph construction module."""
    
    @pytest.fixture
    def db_session(self):
        """Create database session for testing."""
        with db_manager.session_scope() as session:
            yield session
    
    def test_feature_generator_drug(self):
        """Test drug feature generation."""
        generator = FeatureGenerator()
        
        # Test with valid SMILES
        smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
        features = generator.generate_drug_features(smiles)
        
        assert features.shape == (2048,)
        assert features.dtype == 'float32'
        assert features.sum() > 0
    
    def test_feature_generator_invalid_smiles(self):
        """Test with invalid SMILES."""
        generator = FeatureGenerator()
        
        features = generator.generate_drug_features("INVALID")
        
        assert features.shape == (2048,)
        assert features.sum() == 0  # Should return zero vector
    
    def test_graph_builder_initialization(self, db_session):
        """Test graph builder initialization."""
        builder = HeterogeneousGraphBuilder(db_session)
        
        assert builder.db == db_session
        assert builder.feature_generator is not None
    
    def test_build_graph_for_disease(self, db_session):
        """Test building graph for a specific disease."""
        # Skip if no data in database
        disease = db_session.query(Disease).first()
        if not disease:
            pytest.skip("No disease data in database")
        
        builder = HeterogeneousGraphBuilder(db_session)
        graph, metadata = builder.build_graph_for_disease(disease.disease_id)
        
        assert isinstance(graph, dgl.DGLGraph)
        assert 'drug' in graph.ntypes
        assert 'target' in graph.ntypes
        assert 'disease' in graph.ntypes
        
        assert 'drug_ids' in metadata
        assert 'disease_id' in metadata
    
    def test_graph_node_features(self, db_session):
        """Test that graph nodes have features."""
        disease = db_session.query(Disease).first()
        if not disease:
            pytest.skip("No disease data in database")
        
        builder = HeterogeneousGraphBuilder(db_session)
        graph, _ = builder.build_graph_for_disease(disease.disease_id)
        
        # Check drug features
        if graph.num_nodes('drug') > 0:
            assert 'features' in graph.nodes['drug'].data
            drug_features = graph.nodes['drug'].data['features']
            assert drug_features.shape[1] == 2048
        
        # Check target features
        if graph.num_nodes('target') > 0:
            assert 'features' in graph.nodes['target'].data
        
        # Check disease features
        if graph.num_nodes('disease') > 0:
            assert 'features' in graph.nodes['disease'].data


if __name__ == '__main__':
    pytest.main([__file__])