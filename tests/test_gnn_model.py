"""
Unit tests for GNN model.
"""

import pytest
import torch
import dgl

from app.gnn_model.model import DrugSynergyGNN, HeteroRGCNLayer


class TestGNNModel:
    """Test GNN model components."""
    
    def test_model_initialization(self):
        """Test model initialization."""
        model = DrugSynergyGNN(
            drug_feature_dim=2048,
            target_feature_dim=128,
            disease_feature_dim=64,
            hidden_dim=128,
            num_layers=2
        )
        
        assert model.hidden_dim == 128
        assert model.num_layers == 2
        assert len(model.layers) == 2
    
    def test_model_forward(self):
        """Test forward pass."""
        # Create dummy graph
        graph_data = {
            ('drug', 'targets', 'target'): (torch.tensor([0, 1]), torch.tensor([0, 1])),
            ('target', 'targeted_by', 'drug'): (torch.tensor([0, 1]), torch.tensor([0, 1])),
            ('target', 'associated_with', 'disease'): (torch.tensor([0]), torch.tensor([0])),
            ('disease', 'has_target', 'target'): (torch.tensor([0]), torch.tensor([0]))
        }
        
        g = dgl.heterograph(graph_data, num_nodes_dict={'drug': 2, 'target': 2, 'disease': 1})
        
        # Add features
        node_features = {
            'drug': torch.randn(2, 2048),
            'target': torch.randn(2, 128),
            'disease': torch.randn(1, 64)
        }
        
        model = DrugSynergyGNN(
            drug_feature_dim=2048,
            target_feature_dim=128,
            disease_feature_dim=64,
            hidden_dim=64,
            num_layers=2
        )
        
        # Forward pass
        output = model(g, node_features)
        
        assert 'drug' in output
        assert 'target' in output
        assert 'disease' in output
        assert output['drug'].shape == (2, 64)
    
    def test_synergy_prediction(self):
        """Test synergy prediction."""
        graph_data = {
            ('drug', 'targets', 'target'): (torch.tensor([0, 1]), torch.tensor([0, 1])),
            ('target', 'targeted_by', 'drug'): (torch.tensor([0, 1]), torch.tensor([0, 1])),
            ('target', 'associated_with', 'disease'): (torch.tensor([0]), torch.tensor([0])),
            ('disease', 'has_target', 'target'): (torch.tensor([0]), torch.tensor([0]))
        }
        
        g = dgl.heterograph(graph_data, num_nodes_dict={'drug': 2, 'target': 2, 'disease': 1})
        
        node_features = {
            'drug': torch.randn(2, 2048),
            'target': torch.randn(2, 128),
            'disease': torch.randn(1, 64)
        }
        
        model = DrugSynergyGNN(
            drug_feature_dim=2048,
            target_feature_dim=128,
            disease_feature_dim=64,
            hidden_dim=64,
            num_layers=2
        )
        
        # Predict synergy for drug pair (0, 1)
        drug1_idx = torch.tensor([0])
        drug2_idx = torch.tensor([1])
        
        synergy_score = model.predict_synergy(g, node_features, drug1_idx, drug2_idx)
        
        assert synergy_score.shape == (1,)
        assert torch.isfinite(synergy_score).all()


if __name__ == '__main__':
    pytest.main([__file__])