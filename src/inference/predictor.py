"""
Inference pipeline for drug synergy prediction.
"""
import torch
import dgl
import numpy as np
from typing import List, Dict, Tuple
import logging

from src.models.gnn_model import SynergyPredictor
from src.utils.safety_checker import SafetyChecker
from src.preprocessing.smiles_processor import SMILESProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SynergyInferencePipeline:
    """Complete inference pipeline for drug synergy prediction."""
    
    def __init__(
        self,
        model: SynergyPredictor,
        graph: dgl.DGLHeteroGraph,
        drug_info: Dict,
        safety_checker: SafetyChecker,
        device: str = 'cpu'
    ):
        """
        Initialize inference pipeline.
        
        Args:
            model: Trained synergy prediction model
            graph: DGL heterogeneous graph
            drug_info: Dictionary with drug information
            safety_checker: Safety checker instance
            device: Device to run inference on
        """
        self.model = model.to(device)
        self.model.eval()
        self.graph = graph.to(device)
        self.drug_info = drug_info
        self.safety_checker = safety_checker
        self.device = device
    
    def predict_synergy(
        self,
        drug1_id: int,
        drug2_id: int,
        n_mc_samples: int = 20
    ) -> Dict:
        """
        Predict synergy for a drug pair with confidence estimation.
        
        Args:
            drug1_id: First drug ID
            drug2_id: Second drug ID
            n_mc_samples: Number of MC dropout samples for uncertainty
            
        Returns:
            Dictionary with prediction results
        """
        drug1_tensor = torch.tensor([drug1_id], dtype=torch.long).to(self.device)
        drug2_tensor = torch.tensor([drug2_id], dtype=torch.long).to(self.device)
        
        # Predict with uncertainty
        mean_pred, std_pred = self.model.predict_with_uncertainty(
            self.graph, drug1_tensor, drug2_tensor, n_samples=n_mc_samples
        )
        
        # Convert to confidence score (0-100)
        confidence = self._calculate_confidence(std_pred.item())
        
        # Get drug names
        drug1_name = self.drug_info[drug1_id]['name']
        drug2_name = self.drug_info[drug2_id]['name']
        
        # Check safety
        safety_result = self.safety_checker.check_interaction(drug1_name, drug2_name)
        safety_level = self.safety_checker.get_safety_label(safety_result['level'])
        
        return {
            'drug1_id': drug1_id,
            'drug2_id': drug2_id,
            'drug1_name': drug1_name,
            'drug2_name': drug2_name,
            'synergy_score': float(mean_pred.item()),
            'confidence': confidence,
            'uncertainty': float(std_pred.item()),
            'safety_level': safety_level,
            'safety_description': safety_result['description'],
            'smiles1': self.drug_info[drug1_id].get('smiles', 'N/A'),
            'smiles2': self.drug_info[drug2_id].get('smiles', 'N/A')
        }
    
    def predict_top_combinations(
        self,
        disease_id: int,
        top_k: int = 10,
        min_confidence: float = 70.0
    ) -> List[Dict]:
        """
        Predict top synergistic drug combinations for a disease.
        
        Args:
            disease_id: Disease ID
            top_k: Number of top combinations to return
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of prediction dictionaries
        """
        # Get drugs associated with the disease
        disease_drugs = self._get_disease_drugs(disease_id)
        
        if len(disease_drugs) < 2:
            logger.warning(f"Not enough drugs for disease {disease_id}")
            return []
        
        # Generate all drug pairs
        predictions = []
        for i, drug1_id in enumerate(disease_drugs):
            for drug2_id in disease_drugs[i+1:]:
                pred = self.predict_synergy(drug1_id, drug2_id)
                if pred['confidence'] >= min_confidence:
                    predictions.append(pred)
        
        # Sort by synergy score
        predictions.sort(key=lambda x: x['synergy_score'], reverse=True)
        
        return predictions[:top_k]
    
    def _get_disease_drugs(self, disease_id: int) -> List[int]:
        """Get drug IDs associated with a disease."""
        # Get edges from disease to drugs
        if ('disease', 'treated_by', 'drug') in self.graph.canonical_etypes:
            src, dst = self.graph.edges(etype=('disease', 'treated_by', 'drug'))
            mask = src == disease_id
            drug_ids = dst[mask].cpu().numpy().tolist()
            return drug_ids
        return []
    
    def _calculate_confidence(self, uncertainty: float) -> float:
        """
        Calculate confidence score from uncertainty.
        
        Args:
            uncertainty: Standard deviation from MC dropout
            
        Returns:
            Confidence score (0-100)
        """
        # Convert uncertainty to confidence
        # Lower uncertainty -> higher confidence
        # Using exponential decay: confidence = 100 * exp(-k * uncertainty)
        k = 2.0  # Decay factor
        confidence = 100 * np.exp(-k * uncertainty)
        return max(0, min(100, confidence))
    
    @torch.no_grad()
    def batch_predict(
        self,
        drug_pairs: List[Tuple[int, int]],
        batch_size: int = 32
    ) -> List[Dict]:
        """
        Batch prediction for multiple drug pairs.
        
        Args:
            drug_pairs: List of (drug1_id, drug2_id) tuples
            batch_size: Batch size for prediction
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        for i in range(0, len(drug_pairs), batch_size):
            batch_pairs = drug_pairs[i:i+batch_size]
            
            for drug1_id, drug2_id in batch_pairs:
                pred = self.predict_synergy(drug1_id, drug2_id)
                results.append(pred)
        
        return results


# Example usage
if __name__ == "__main__":
    # This would be loaded from saved models
    in_feats = {'drug': 2048, 'target': 128, 'disease': 64}
    model = SynergyPredictor(in_feats)
    
    # Mock drug info
    drug_info = {
        0: {'name': 'Doxorubicin', 'smiles': 'CC1...'},
        1: {'name': 'Cyclophosphamide', 'smiles': 'C1...'}
    }
    
    safety_checker = SafetyChecker()
    
    print("Inference pipeline template created")