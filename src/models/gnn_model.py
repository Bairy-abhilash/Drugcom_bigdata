"""
Graph Neural Network models for drug synergy prediction.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GraphConv, SAGEConv, GATConv
from typing import Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HeteroGraphSAGE(nn.Module):
    """GraphSAGE model for heterogeneous graphs."""
    
    def __init__(
        self,
        in_feats: Dict[str, int],
        hidden_dim: int = 256,
        out_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        aggregator_type: str = 'mean'
    ):
        """
        Initialize HeteroGraphSAGE.
        
        Args:
            in_feats: Dictionary mapping node type to input feature dimension
            hidden_dim: Hidden layer dimension
            out_dim: Output embedding dimension
            num_layers: Number of graph convolutional layers
            dropout: Dropout rate
            aggregator_type: Aggregation type ('mean', 'gcn', 'pool', 'lstm')
        """
        super(HeteroGraphSAGE, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Input projection layers for each node type
        self.input_projs = nn.ModuleDict({
            ntype: nn.Linear(in_dim, hidden_dim)
            for ntype, in_dim in in_feats.items()
        })
        
        # GraphSAGE layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_d = hidden_dim if i == 0 else hidden_dim
            out_d = out_dim if i == num_layers - 1 else hidden_dim
            
            self.layers.append(
                dgl.nn.pytorch.HeteroGraphConv({
                    etype: SAGEConv(in_d, out_d, aggregator_type=aggregator_type)
                    for etype in [('drug', 'targets', 'target'),
                                  ('target', 'targeted_by', 'drug'),
                                  ('drug', 'similar_to', 'drug'),
                                  ('drug', 'treats', 'disease'),
                                  ('disease', 'treated_by', 'drug'),
                                  ('target', 'associated_with', 'disease'),
                                  ('disease', 'has_target', 'target')]
                }, aggregate='sum')
            )
        
        self.batch_norms = nn.ModuleList([
            nn.ModuleDict({
                ntype: nn.BatchNorm1d(out_dim if i == num_layers - 1 else hidden_dim)
                for ntype in in_feats.keys()
            })
            for i in range(num_layers)
        ])
    
    def forward(self, g: dgl.DGLHeteroGraph, node_features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            g: DGL heterogeneous graph
            node_features: Dictionary mapping node type to feature tensor
            
        Returns:
            Dictionary mapping node type to embedding tensor
        """
        # Project input features
        h = {ntype: F.relu(self.input_projs[ntype](feat)) 
             for ntype, feat in node_features.items()}
        
        # Apply graph convolutional layers
        for i, (layer, bn_dict) in enumerate(zip(self.layers, self.batch_norms)):
            h = layer(g, h)
            
            # Apply batch normalization and activation
            h = {ntype: bn_dict[ntype](feat) for ntype, feat in h.items() if feat.shape[0] > 1}
            
            if i < self.num_layers - 1:
                h = {ntype: F.relu(feat) for ntype, feat in h.items()}
                h = {ntype: F.dropout(feat, p=self.dropout, training=self.training) 
                     for ntype, feat in h.items()}
        
        return h


class SynergyPredictor(nn.Module):
    """Drug synergy prediction model."""
    
    def __init__(
        self,
        in_feats: Dict[str, int],
        hidden_dim: int = 256,
        embedding_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        use_mc_dropout: bool = True
    ):
        """
        Initialize synergy predictor.
        
        Args:
            in_feats: Dictionary mapping node type to input feature dimension
            hidden_dim: Hidden layer dimension
            embedding_dim: Node embedding dimension
            num_layers: Number of GNN layers
            dropout: Dropout rate
            use_mc_dropout: Whether to use MC dropout for confidence estimation
        """
        super(SynergyPredictor, self).__init__()
        
        self.use_mc_dropout = use_mc_dropout
        self.dropout = dropout
        
        # Graph encoder
        self.gnn = HeteroGraphSAGE(
            in_feats, hidden_dim, embedding_dim, num_layers, dropout
        )
        
        # Synergy prediction MLP
        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(
        self,
        g: dgl.DGLHeteroGraph,
        drug1_ids: torch.Tensor,
        drug2_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict synergy scores for drug pairs.
        
        Args:
            g: DGL heterogeneous graph
            drug1_ids: First drug IDs
            drug2_ids: Second drug IDs
            
        Returns:
            Synergy scores
        """
        # Get node features
        node_features = {
            ntype: g.nodes[ntype].data['feat']
            for ntype in g.ntypes
        }
        
        # Get node embeddings
        embeddings = self.gnn(g, node_features)
        drug_embeddings = embeddings['drug']
        
        # Get embeddings for drug pairs
        drug1_emb = drug_embeddings[drug1_ids]
        drug2_emb = drug_embeddings[drug2_ids]
        
        # Concatenate drug embeddings
        pair_emb = torch.cat([drug1_emb, drug2_emb], dim=1)
        
        # Predict synergy
        synergy_score = self.predictor(pair_emb)
        
        return synergy_score.squeeze()
    
    def predict_with_uncertainty(
        self,
        g: dgl.DGLHeteroGraph,
        drug1_ids: torch.Tensor,
        drug2_ids: torch.Tensor,
        n_samples: int = 20
    ) -> tuple:
        """
        Predict synergy with uncertainty estimation using MC Dropout.
        
        Args:
            g: DGL heterogeneous graph
            drug1_ids: First drug IDs
            drug2_ids: Second drug IDs
            n_samples: Number of MC dropout samples
            
        Returns:
            Tuple of (mean prediction, standard deviation)
        """
        self.train()  # Enable dropout
        
        predictions = []
        for _ in range(n_samples):
            with torch.no_grad():
                pred = self.forward(g, drug1_ids, drug2_ids)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)
        
        return mean_pred, std_pred


# Example usage
if __name__ == "__main__":
    # Create dummy graph and model
    in_feats = {'drug': 2048, 'target': 128, 'disease': 64}
    model = SynergyPredictor(in_feats, hidden_dim=256, embedding_dim=128)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Model: {model}")