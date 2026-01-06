"""
Feature engineering for drug molecules and graphs.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Engineer features for drug synergy prediction."""
    
    def __init__(self):
        """Initialize feature engineer."""
        self.scaler = StandardScaler()
        self.fitted = False
    
    def create_drug_features(
        self,
        fingerprints: np.ndarray,
        descriptors: List[Dict] = None
    ) -> np.ndarray:
        """
        Create drug features from fingerprints and descriptors.
        
        Args:
            fingerprints: Molecular fingerprints (N x D)
            descriptors: List of descriptor dictionaries
            
        Returns:
            Combined feature matrix
        """
        features = [fingerprints]
        
        if descriptors is not None:
            descriptor_matrix = self._descriptors_to_matrix(descriptors)
            features.append(descriptor_matrix)
        
        combined = np.concatenate(features, axis=1)
        logger.info(f"Created drug features: shape {combined.shape}")
        
        return combined
    
    def create_target_features(
        self,
        n_targets: int,
        embedding_dim: int = 128,
        use_pretrained: bool = False
    ) -> np.ndarray:
        """
        Create target protein features.
        
        Args:
            n_targets: Number of targets
            embedding_dim: Embedding dimension
            use_pretrained: Whether to use pretrained embeddings
            
        Returns:
            Target feature matrix
        """
        if use_pretrained:
            # In practice, load from pretrained protein embeddings (e.g., ESM, ProtTrans)
            logger.info("Using pretrained protein embeddings (placeholder)")
            features = np.random.randn(n_targets, embedding_dim)
        else:
            # Random initialization
            features = np.random.randn(n_targets, embedding_dim) * 0.01
        
        logger.info(f"Created target features: shape {features.shape}")
        return features
    
    def create_disease_features(
        self,
        n_diseases: int,
        embedding_dim: int = 64
    ) -> np.ndarray:
        """
        Create disease features.
        
        Args:
            n_diseases: Number of diseases
            embedding_dim: Embedding dimension
            
        Returns:
            Disease feature matrix
        """
        # One-hot encoding or learned embeddings
        features = np.random.randn(n_diseases, embedding_dim) * 0.01
        
        logger.info(f"Created disease features: shape {features.shape}")
        return features
    
    def normalize_features(
        self,
        features: np.ndarray,
        fit: bool = True
    ) -> np.ndarray:
        """
        Normalize features using standard scaling.
        
        Args:
            features: Feature matrix
            fit: Whether to fit the scaler
            
        Returns:
            Normalized features
        """
        if fit:
            normalized = self.scaler.fit_transform(features)
            self.fitted = True
        else:
            if not self.fitted:
                raise ValueError("Scaler not fitted. Set fit=True first.")
            normalized = self.scaler.transform(features)
        
        return normalized
    
    def create_pair_features(
        self,
        drug1_features: np.ndarray,
        drug2_features: np.ndarray,
        method: str = 'concat'
    ) -> np.ndarray:
        """
        Create features for drug pairs.
        
        Args:
            drug1_features: First drug features
            drug2_features: Second drug features
            method: Combination method ('concat', 'add', 'multiply', 'diff')
            
        Returns:
            Pair feature matrix
        """
        if method == 'concat':
            pair_features = np.concatenate([drug1_features, drug2_features], axis=1)
        elif method == 'add':
            pair_features = drug1_features + drug2_features
        elif method == 'multiply':
            pair_features = drug1_features * drug2_features
        elif method == 'diff':
            pair_features = np.abs(drug1_features - drug2_features)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        logger.info(f"Created pair features with method '{method}': shape {pair_features.shape}")
        return pair_features
    
    def _descriptors_to_matrix(self, descriptors: List[Dict]) -> np.ndarray:
        """Convert list of descriptor dictionaries to matrix."""
        if not descriptors:
            return np.array([])
        
        # Get all keys
        keys = list(descriptors[0].keys())
        
        # Create matrix
        matrix = np.zeros((len(descriptors), len(keys)))
        for i, desc in enumerate(descriptors):
            for j, key in enumerate(keys):
                matrix[i, j] = desc.get(key, 0.0)
        
        return matrix
    
    def add_graph_features(
        self,
        node_features: Dict[str, np.ndarray],
        add_degree: bool = True,
        add_clustering: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Add graph-based features to node features.
        
        Args:
            node_features: Dictionary of node type to feature matrix
            add_degree: Whether to add degree features
            add_clustering: Whether to add clustering coefficient
            
        Returns:
            Enhanced node features
        """
        enhanced = {}
        
        for node_type, features in node_features.items():
            enhanced_features = [features]
            
            if add_degree:
                # Placeholder for degree features
                degree_features = np.random.randint(1, 10, (features.shape[0], 1))
                enhanced_features.append(degree_features)
            
            if add_clustering:
                # Placeholder for clustering coefficient
                clustering = np.random.rand(features.shape[0], 1)
                enhanced_features.append(clustering)
            
            enhanced[node_type] = np.concatenate(enhanced_features, axis=1)
        
        return enhanced


# Example usage
if __name__ == "__main__":
    engineer = FeatureEngineer()
    
    # Create dummy features
    fingerprints = np.random.rand(100, 2048)
    descriptors = [{'mw': 300, 'logp': 2.5} for _ in range(100)]
    
    features = engineer.create_drug_features(fingerprints, descriptors)
    print(f"Drug features shape: {features.shape}")
    
    # Normalize
    normalized = engineer.normalize_features(features)
    print(f"Normalized features shape: {normalized.shape}")