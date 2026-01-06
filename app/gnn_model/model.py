"""
Graph Neural Network model for drug synergy prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn.pytorch as dglnn
from typing import Dict, Tuple

from app.utils.logger import setup_logger
from app.utils.config import settings

logger = setup_logger(__name__)


class HeteroRGCNLayer(nn.Module):
    """Heterogeneous Relational Graph Convolution Layer."""
    
    def __init__(self, in_size: Dict[str, int], out_size: int, etypes: list):
        """
        Initialize heterogeneous RGCN layer.
        
        Args:
            in_size: Dictionary mapping node types to input feature dimensions
            out_size: Output feature dimension
            etypes: List of edge types (canonical edge types)
        """
        super(HeteroRGCNLayer, self).__init__()
        
        # Create a GraphConv module for each edge type
        self.conv_dict = nn.ModuleDict()
        for ntype, in_dim in in_size.items():
            self.conv_dict[ntype] = nn.Linear(in_dim, out_size)
        
        # Relational graph convolution for each edge type
        self.edge_conv = nn.ModuleDict()
        for etype in etypes:
            self.edge_conv[etype] = dglnn.GraphConv(out_size, out_size, norm='both')
        
        self.out_size = out_size
    
    def forward(self, g: dgl.DGLGraph, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            g: DGL heterogeneous graph
            inputs: Dictionary of input features for each node type
            
        Returns:
            Dictionary of output features for each node type
        """
        # First transform features to common dimension
        h_dict = {}
        for ntype, h in inputs.items():
            h_dict[ntype] = self.conv_dict[ntype](h)
        
        # Message passing for each edge type
        outputs = {ntype: [] for ntype in h_dict.keys()}
        
        for etype in g.canonical_etypes:
            srctype, etype_name, dsttype = etype
            
            # Create subgraph for this edge type
            subg = g[etype]
            
            if subg.num_edges() > 0:
                # Apply graph convolution
                if srctype in h_dict and dsttype in h_dict:
                    feat = self.edge_conv[etype_name](
                        subg, 
                        (h_dict[srctype], h_dict[dsttype])
                    )
                    outputs[dsttype].append(feat)
        
        # Aggregate messages from different edge types
        h_new = {}
        for ntype in h_dict.keys():
            if len(outputs[ntype]) > 0:
                h_new[ntype] = torch.stack(outputs[ntype], dim=0).mean(dim=0)
            else:
                h_new[ntype] = h_dict[ntype]
        
        return h_new


class HeteroGAT(nn.Module):
    """Heterogeneous Graph Attention Network."""
    
    def __init__(
        self, 
        in_size: Dict[str, int], 
        hidden_size: int, 
        out_size: int,
        num_heads: int = 4,
        etypes: list = None
    ):
        """
        Initialize Heterogeneous GAT.
        
        Args:
            in_size: Dictionary of input dimensions for each node type
            hidden_size: Hidden layer dimension
            out_size: Output dimension
            num_heads: Number of attention heads
            etypes: List of edge types
        """
        super(HeteroGAT, self).__init__()
        
        self.etypes = etypes or []
        
        # Input projection for each node type
        self.input_proj = nn.ModuleDict()
        for ntype, in_dim in in_size.items():
            self.input_proj[ntype] = nn.Linear(in_dim, hidden_size)
        
        # GAT layers for each edge type
        self.gat_layers = nn.ModuleDict()
        for etype in self.etypes:
            self.gat_layers[etype] = dglnn.GATConv(
                hidden_size, 
                hidden_size // num_heads, 
                num_heads=num_heads
            )
        
        # Output projection
        self.output_proj = nn.ModuleDict()
        for ntype in in_size.keys():
            self.output_proj[ntype] = nn.Linear(hidden_size, out_size)
    
    def forward(self, g: dgl.DGLGraph, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        # Input projection
        h = {ntype: F.relu(self.input_proj[ntype](feat)) 
             for ntype, feat in inputs.items()}
        
        # Message passing
        h_out = {ntype: [] for ntype in h.keys()}
        
        for etype in g.canonical_etypes:
            srctype, etype_name, dsttype = etype
            subg = g[etype]
            
            if subg.num_edges() > 0 and etype_name in self.gat_layers:
                # Apply GAT
                feat = self.gat_layers[etype_name](
                    subg,
                    (h[srctype], h[dsttype])
                )
                # Flatten multi-head output
                feat = feat.flatten(1)
                h_out[dsttype].append(feat)
        
        # Aggregate and project
        output = {}
        for ntype in h.keys():
            if len(h_out[ntype]) > 0:
                aggregated = torch.stack(h_out[ntype], dim=0).mean(dim=0)
            else:
                aggregated = h[ntype]
            output[ntype] = self.output_proj[ntype](aggregated)
        
        return output


class DrugSynergyGNN(nn.Module):
    """
    Drug Synergy Prediction Model using Heterogeneous GNN.
    """
    
    def __init__(
        self,
        drug_feature_dim: int,
        target_feature_dim: int,
        disease_feature_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.3,
        use_gat: bool = True,
        num_heads: int = 4
    ):
        """
        Initialize Drug Synergy GNN.
        
        Args:
            drug_feature_dim: Dimension of drug features
            target_feature_dim: Dimension of target features
            disease_feature_dim: Dimension of disease features
            hidden_dim: Hidden layer dimension
            num_layers: Number of GNN layers
            dropout: Dropout rate
            use_gat: Whether to use GAT (True) or RGCN (False)
            num_heads: Number of attention heads (for GAT)
        """
        super(DrugSynergyGNN, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_gat = use_gat
        
        # Input dimensions
        in_size = {
            'drug': drug_feature_dim,
            'target': target_feature_dim,
            'disease': disease_feature_dim
        }
        
        # Edge types
        etypes = ['targets', 'targeted_by', 'associated_with', 'has_target']
        
        # GNN layers
        self.layers = nn.ModuleList()
        
        if use_gat:
            # First layer
            self.layers.append(
                HeteroGAT(in_size, hidden_dim, hidden_dim, num_heads, etypes)
            )
            # Hidden layers
            in_size_hidden = {k: hidden_dim for k in in_size.keys()}
            for _ in range(num_layers - 1):
                self.layers.append(
                    HeteroGAT(in_size_hidden, hidden_dim, hidden_dim, num_heads, etypes)
                )
        else:
            # RGCN layers
            for i in range(num_layers):
                if i == 0:
                    self.layers.append(HeteroRGCNLayer(in_size, hidden_dim, etypes))
                else:
                    in_size_hidden = {k: hidden_dim for k in in_size.keys()}
                    self.layers.append(HeteroRGCNLayer(in_size_hidden, hidden_dim, etypes))
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
        # Drug pair synergy prediction head
        self.synergy_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        logger.info(f"Initialized DrugSynergyGNN with {num_layers} layers, hidden_dim={hidden_dim}")
    
    def forward(
        self, 
        g: dgl.DGLGraph, 
        node_features: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the GNN.
        
        Args:
            g: DGL heterogeneous graph
            node_features: Dictionary of node features
            
        Returns:
            Dictionary of node embeddings
        """
        h = node_features
        
        # Apply GNN layers
        for i, layer in enumerate(self.layers):
            h = layer(g, h)
            
            # Apply activation and dropout (except last layer)
            if i < len(self.layers) - 1:
                h = {k: F.relu(v) for k, v in h.items()}
                h = {k: self.dropout_layer(v) for k, v in h.items()}
        
        return h
    
    def predict_synergy(
        self,
        g: dgl.DGLGraph,
        node_features: Dict[str, torch.Tensor],
        drug1_idx: torch.Tensor,
        drug2_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict synergy score for drug pairs.
        
        Args:
            g: DGL heterogeneous graph
            node_features: Dictionary of node features
            drug1_idx: Indices of first drugs in pairs
            drug2_idx: Indices of second drugs in pairs
            
        Returns:
            Synergy scores for each pair
        """
        # Get node embeddings
        h = self.forward(g, node_features)
        
        # Get drug embeddings
        drug_embeddings = h['drug']
        
        # Get embeddings for drug pairs
        drug1_emb = drug_embeddings[drug1_idx]
        drug2_emb = drug_embeddings[drug2_idx]
        
        # Concatenate pair embeddings
        pair_emb = torch.cat([drug1_emb, drug2_emb], dim=1)
        
        # Predict synergy
        synergy_scores = self.synergy_predictor(pair_emb).squeeze(-1)
        
        return synergy_scores
    
    def predict_synergy_with_uncertainty(
        self,
        g: dgl.DGLGraph,
        node_features: Dict[str, torch.Tensor],
        drug1_idx: torch.Tensor,
        drug2_idx: torch.Tensor,
        num_samples: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict synergy with uncertainty using MC Dropout.
        
        Args:
            g: DGL heterogeneous graph
            node_features: Dictionary of node features
            drug1_idx: Indices of first drugs
            drug2_idx: Indices of second drugs
            num_samples: Number of MC samples
            
        Returns:
            Tuple of (mean predictions, standard deviations)
        """
        self.train()  # Enable dropout
        
        predictions = []
        for _ in range(num_samples):
            with torch.no_grad():
                pred = self.predict_synergy(g, node_features, drug1_idx, drug2_idx)
                predictions.append(pred)
        
        self.eval()
        
        predictions = torch.stack(predictions, dim=0)
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)
        
        return mean_pred, std_pred


class DrugSynergyPredictor:
    """High-level wrapper for drug synergy prediction."""
    
    def __init__(self, model: DrugSynergyGNN, device: str = 'cpu'):
        """
        Initialize predictor.
        
        Args:
            model: DrugSynergyGNN model
            device: Device to run on ('cpu' or 'cuda')
        """
        self.model = model
        self.device = device
        self.model.to(device)
    
    def predict(
        self,
        g: dgl.DGLGraph,
        node_features: Dict[str, torch.Tensor],
        drug_pairs: list,
        with_uncertainty: bool = True
    ) -> list:
        """
        Predict synergy for drug pairs.
        
        Args:
            g: DGL graph
            node_features: Node features
            drug_pairs: List of (drug1_idx, drug2_idx) tuples
            with_uncertainty: Whether to compute uncertainty
            
        Returns:
            List of prediction dictionaries
        """
        self.model.eval()
        
        # Move graph and features to device
        g = g.to(self.device)
        node_features = {k: v.to(self.device) for k, v in node_features.items()}
        
        drug1_indices = torch.tensor([p[0] for p in drug_pairs], dtype=torch.long, device=self.device)
        drug2_indices = torch.tensor([p[1] for p in drug_pairs], dtype=torch.long, device=self.device)
        
        with torch.no_grad():
            if with_uncertainty:
                mean_scores, std_scores = self.model.predict_synergy_with_uncertainty(
                    g, node_features, drug1_indices, drug2_indices
                )
            else:
                mean_scores = self.model.predict_synergy(
                    g, node_features, drug1_indices, drug2_indices
                )
                std_scores = torch.zeros_like(mean_scores)
        
        # Format results
        results = []
        for i, (drug1_idx, drug2_idx) in enumerate(drug_pairs):
            results.append({
                'drug1_idx': drug1_idx,
                'drug2_idx': drug2_idx,
                'synergy_score': float(mean_scores[i]),
                'confidence': float(1.0 / (1.0 + std_scores[i])) if with_uncertainty else 1.0
            })
        
        return results