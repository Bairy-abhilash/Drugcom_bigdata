"""
Inference module for trained GNN model.
"""

import torch
import dgl
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from app.gnn_model.model import DrugSynergyGNN
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class SynergyInference:
    """Inference engine for drug synergy prediction."""
    
    def __init__(
        self,
        model: DrugSynergyGNN,
        model_path: Optional[str] = None,
        device: str = 'cpu'
    ):
        """
        Initialize inference engine.
        
        Args:
            model: DrugSynergyGNN model
            model_path: Path to trained model weights
            device: Device to run inference on
        """
        self.model = model
        self.device = device
        self.model.to(device)
        
        if model_path:
            self.load_model(model_path)
        
        self.model.eval()
    
    def load_model(self, path: str) -> None:
        """Load trained model weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded model from {path}")
    
    def predict_drug_pair(
        self,
        graph: dgl.DGLGraph,
        node_features: Dict[str, torch.Tensor],
        drug1_idx: int,
        drug2_idx: int,
        with_uncertainty: bool = True
    ) -> Dict[str, float]:
        """
        Predict synergy for a single drug pair.
        
        Args:
            graph: DGL heterogeneous graph
            node_features: Dictionary of node features
            drug1_idx: Index of first drug
            drug2_idx: Index of second drug
            with_uncertainty: Whether to compute uncertainty estimate
            
        Returns:
            Dictionary with prediction results
        """
        graph = graph.to(self.device)
        node_features = {k: v.to(self.device) for k, v in node_features.items()}
        
        drug1_tensor = torch.tensor([drug1_idx], dtype=torch.long, device=self.device)
        drug2_tensor = torch.tensor([drug2_idx], dtype=torch.long, device=self.device)
        
        with torch.no_grad():
            if with_uncertainty:
                mean_score, std_score = self.model.predict_synergy_with_uncertainty(
                    graph, node_features, drug1_tensor, drug2_tensor, num_samples=10
                )
                confidence = float(1.0 / (1.0 + std_score[0]))
            else:
                mean_score = self.model.predict_synergy(
                    graph, node_features, drug1_tensor, drug2_tensor
                )
                confidence = 1.0
        
        return {
            'synergy_score': float(mean_score[0]),
            'confidence': confidence
        }
    
    def predict_all_pairs(
        self,
        graph: dgl.DGLGraph,
        node_features: Dict[str, torch.Tensor],
        drug_indices: List[int],
        batch_size: int = 128
    ) -> List[Dict]:
        """
        Predict synergy for all drug pairs.
        
        Args:
            graph: DGL graph
            node_features: Node features
            drug_indices: List of drug indices
            batch_size: Batch size for prediction
            
        Returns:
            List of prediction dictionaries
        """
        graph = graph.to(self.device)
        node_features = {k: v.to(self.device) for k, v in node_features.items()}
        
        # Generate all pairs
        pairs = []
        for i, drug1_idx in enumerate(drug_indices):
            for drug2_idx in drug_indices[i+1:]:
                pairs.append((drug1_idx, drug2_idx))
        
        logger.info(f"Predicting synergy for {len(pairs)} drug pairs")
        
        # Batch prediction
        results = []
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i+batch_size]
            
            drug1_indices = torch.tensor(
                [p[0] for p in batch_pairs], 
                dtype=torch.long, 
                device=self.device
            )
            drug2_indices = torch.tensor(
                [p[1] for p in batch_pairs], 
                dtype=torch.long, 
                device=self.device
            )
            
            with torch.no_grad():
                mean_scores, std_scores = self.model.predict_synergy_with_uncertainty(
                    graph, node_features, drug1_indices, drug2_indices
                )
            
            for j, (drug1_idx, drug2_idx) in enumerate(batch_pairs):
                results.append({
                    'drug1_idx': drug1_idx,
                    'drug2_idx': drug2_idx,
                    'synergy_score': float(mean_scores[j]),
                    'confidence': float(1.0 / (1.0 + std_scores[j]))
                })
        
        return results
    
    def rank_drug_pairs(
        self,
        graph: dgl.DGLGraph,
        node_features: Dict[str, torch.Tensor],
        drug_indices: List[int],
        top_k: int = 10
    ) -> List[Dict]:
        """
        Rank drug pairs by predicted synergy.
        
        Args:
            graph: DGL graph
            node_features: Node features
            drug_indices: List of drug indices
            top_k: Number of top pairs to return
            
        Returns:
            List of top-k ranked predictions
        """
        predictions = self.predict_all_pairs(graph, node_features, drug_indices)
        
        # Sort by synergy score
        ranked = sorted(predictions, key=lambda x: x['synergy_score'], reverse=True)
        
        return ranked[:top_k]