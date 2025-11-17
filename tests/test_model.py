"""
Unit tests for GNN model.
"""
import pytest
import torch
import dgl
import numpy as np
from src.models.gnn_model import HeteroGraphSAGE, SynergyPredictor


class TestHeteroGraphSAGE:
    """Test cases for HeteroGraphSAGE model."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.in_feats = {'drug': 2048, 'target': 128, 'disease': 64}
        self.hidden_dim = 256
        self.out_dim = 128
        self.model = HeteroGraphSAGE(self.in_feats, self.hidden_dim, self.out_dim)
    
    def test_model_initialization(self):
        """Test model initializes correctly."""
        assert self.model is not None
        assert len(self.model.layers) == 2
    
    def test_forward_pass(self):
        """Test forward pass produces correct output shape."""
        # Create dummy graph
        graph_data = {
            ('drug', 'targets', 'target'): (torch.tensor([0, 1]), torch.tensor([0, 1])),
            ('target', 'targeted_by', 'drug'): (torch.tensor([0, 1]), torch.tensor([0, 1])),
            ('drug', 'similar_to', 'drug'): (torch.tensor([0]), torch.tensor([1])),
            ('drug', 'treats', 'disease'): (torch.tensor([0]), torch.tensor([0])),
            ('disease', 'treated_by', 'drug'): (torch.tensor([0]), torch.tensor([0])),
            ('target', 'associated_with', 'disease'): (torch.tensor([0]), torch.tensor([0])),
            ('disease', 'has_target', 'target'): (torch.tensor([0]), torch.tensor([0]))
        }
        g = dgl.heterograph(graph_data)
        
        # Add node features
        node_features = {
            'drug': torch.randn(2, self.in_feats['drug']),
            'target': torch.randn(2, self.in_feats['target']),
            'disease': torch.randn(1, self.in_feats['disease'])
        }
        
        # Forward pass
        output = self.model(g, node_features)
        
        assert 'drug' in output
        assert output['drug'].shape == (2, self.out_dim)
    
    def test_model_training_mode(self):
        """Test model can switch between train and eval modes."""
        self.model.train()
        assert self.model.training
        
        self.model.eval()
        assert not self.model.training


class TestSynergyPredictor:
    """Test cases for SynergyPredictor model."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.in_feats = {'drug': 2048, 'target': 128, 'disease': 64}
        self.model = SynergyPredictor(self.in_feats, use_mc_dropout=True)
    
    def test_synergy_prediction(self):
        """Test synergy prediction."""
        # Create dummy graph
        graph_data = {
            ('drug', 'targets', 'target'): (torch.tensor([0, 1]), torch.tensor([0, 1])),
            ('target', 'targeted_by', 'drug'): (torch.tensor([0, 1]), torch.tensor([0, 1])),
            ('drug', 'similar_to', 'drug'): (torch.tensor([0]), torch.tensor([1])),
            ('drug', 'treats', 'disease'): (torch.tensor([0]), torch.tensor([0])),
            ('disease', 'treated_by', 'drug'): (torch.tensor([0]), torch.tensor([0])),
            ('target', 'associated_with', 'disease'): (torch.tensor([0]), torch.tensor([0])),
            ('disease', 'has_target', 'target'): (torch.tensor([0]), torch.tensor([0]))
        }
        g = dgl.heterograph(graph_data)
        
        # Add features
        g.nodes['drug'].data['feat'] = torch.randn(2, self.in_feats['drug'])
        g.nodes['target'].data['feat'] = torch.randn(2, self.in_feats['target'])
        g.nodes['disease'].data['feat'] = torch.randn(1, self.in_feats['disease'])
        
        # Predict
        drug1_ids = torch.tensor([0])
        drug2_ids = torch.tensor([1])
        
        score = self.model(g, drug1_ids, drug2_ids)
        
        assert score.shape == (1,)
    
    def test_uncertainty_estimation(self):
        """Test MC Dropout uncertainty estimation."""
        # Create dummy graph
        graph_data = {
            ('drug', 'targets', 'target'): (torch.tensor([0, 1]), torch.tensor([0, 1])),
            ('target', 'targeted_by', 'drug'): (torch.tensor([0, 1]), torch.tensor([0, 1])),
            ('drug', 'similar_to', 'drug'): (torch.tensor([0]), torch.tensor([1])),
            ('drug', 'treats', 'disease'): (torch.tensor([0]), torch.tensor([0])),
            ('disease', 'treated_by', 'drug'): (torch.tensor([0]), torch.tensor([0])),
            ('target', 'associated_with', 'disease'): (torch.tensor([0]), torch.tensor([0])),
            ('disease', 'has_target', 'target'): (torch.tensor([0]), torch.tensor([0]))
        }
        g = dgl.heterograph(graph_data)
        
        # Add features
        g.nodes['drug'].data['feat'] = torch.randn(2, self.in_feats['drug'])
        g.nodes['target'].data['feat'] = torch.randn(2, self.in_feats['target'])
        g.nodes['disease'].data['feat'] = torch.randn(1, self.in_feats['disease'])
        
        # Predict with uncertainty
        drug1_ids = torch.tensor([0])
        drug2_ids = torch.tensor([1])
        
        mean, std = self.model.predict_with_uncertainty(g, drug1_ids, drug2_ids, n_samples=10)
        
        assert mean.shape == (1,)
        assert std.shape == (1,)
        assert std.item() >= 0


if __name__ == "__main__":
    pytest.main([__file__])