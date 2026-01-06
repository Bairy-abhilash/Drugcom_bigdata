"""
Build heterogeneous graphs from processed data.
"""
import dgl
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging

from src.data.drugcomb_loader import DrugCombLoader
from src.preprocessing.smiles_processor import SMILESProcessor
from src.preprocessing.feature_engineering import FeatureEngineer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HeteroGraphBuilder:
    """Build heterogeneous graph from DrugComb data."""
    
    def __init__(
        self,
        drugcomb_loader: DrugCombLoader,
        smiles_processor: SMILESProcessor,
        feature_engineer: FeatureEngineer
    ):
        """
        Initialize graph builder.
        
        Args:
            drugcomb_loader: DrugComb data loader
            smiles_processor: SMILES processor
            feature_engineer: Feature engineer
        """
        self.drugcomb_loader = drugcomb_loader
        self.smiles_processor = smiles_processor
        self.feature_engineer = feature_engineer
    
    def build_graph(
        self,
        similarity_threshold: float = 0.7,
        add_mock_targets: bool = True
    ) -> Tuple[dgl.DGLHeteroGraph, Dict]:
        """
        Build complete heterogeneous graph.
        
        Args:
            similarity_threshold: Threshold for drug similarity edges
            add_mock_targets: Whether to add mock target nodes
            
        Returns:
            Tuple of (graph, metadata)
        """
        # Load and preprocess data
        logger.info("Loading DrugComb data...")
        processed_data = self.drugcomb_loader.preprocess()
        
        # Generate drug features
        logger.info("Generating drug features...")
        drug_features = self._generate_drug_features(processed_data['drug_info'])
        
        # Generate target and disease features
        logger.info("Generating target and disease features...")
        if add_mock_targets:
            n_targets = 50  # Mock number of targets
            target_features = self.feature_engineer.create_target_features(n_targets)
        else:
            n_targets = 0
            target_features = np.array([])
        
        n_diseases = len(processed_data['disease_vocab'])
        disease_features = self.feature_engineer.create_disease_features(n_diseases)
        
        # Build edges
        logger.info("Building graph edges...")
        edges = self._build_edges(
            processed_data,
            drug_features,
            similarity_threshold,
            n_targets
        )
        
        # Create graph
        logger.info("Creating DGL heterograph...")
        graph = self._create_dgl_graph(
            drug_features,
            target_features,
            disease_features,
            edges
        )
        
        # Metadata
        metadata = {
            'n_drugs': len(processed_data['drug_vocab']),
            'n_targets': n_targets,
            'n_diseases': n_diseases,
            'drug_vocab': processed_data['drug_vocab'],
            'disease_vocab': processed_data['disease_vocab'],
            'drug_info': processed_data['drug_info'],
            'disease_info': processed_data['disease_info']
        }
        
        logger.info(f"Graph built: {graph.num_nodes()} nodes, {graph.num_edges()} edges")
        return graph, metadata
    
    def _generate_drug_features(self, drug_info: Dict) -> np.ndarray:
        """Generate features for all drugs."""
        features_list = []
        valid_indices = []
        
        for drug_id in sorted(drug_info.keys()):
            smiles = drug_info[drug_id]['smiles']
            
            if smiles != 'N/A' and not smiles.startswith('ANTIBODY'):
                fp = self.smiles_processor.generate_fingerprint(smiles)
                if fp is not None:
                    features_list.append(fp)
                    valid_indices.append(drug_id)
                else:
                    # Use zero vector for failed SMILES
                    features_list.append(np.zeros(self.smiles_processor.n_bits))
                    valid_indices.append(drug_id)
            else:
                # Use random vector for antibodies/biologics
                features_list.append(np.random.randn(self.smiles_processor.n_bits) * 0.01)
                valid_indices.append(drug_id)
        
        features = np.array(features_list)
        logger.info(f"Generated features for {len(features)} drugs")
        
        return features
    
    def _build_edges(
        self,
        processed_data: Dict,
        drug_features: np.ndarray,
        similarity_threshold: float,
        n_targets: int
    ) -> Dict:
        """Build all edge types."""
        edges = {}
        
        # Drug-Drug similarity edges
        from sklearn.metrics.pairwise import cosine_similarity
        similarity_matrix = cosine_similarity(drug_features)
        
        drug_sim_edges = []
        n_drugs = len(drug_features)
        for i in range(n_drugs):
            for j in range(i + 1, n_drugs):
                if similarity_matrix[i, j] >= similarity_threshold:
                    drug_sim_edges.append((i, j))
                    drug_sim_edges.append((j, i))
        
        edges['drug_similar_to_drug'] = drug_sim_edges
        logger.info(f"Created {len(drug_sim_edges)} drug similarity edges")
        
        # Drug-Target edges (mock)
        if n_targets > 0:
            drug_target_edges = []
            for i in range(n_drugs):
                # Each drug targets 2-5 proteins
                n_targets_per_drug = np.random.randint(2, min(6, n_targets + 1))
                target_ids = np.random.choice(n_targets, n_targets_per_drug, replace=False)
                for target_id in target_ids:
                    drug_target_edges.append((i, target_id))
            
            edges['drug_targets_target'] = drug_target_edges
            logger.info(f"Created {len(drug_target_edges)} drug-target edges")
            
            # Target-Disease edges (mock)
            target_disease_edges = []
            n_diseases = len(processed_data['disease_vocab'])
            for i in range(n_targets):
                disease_id = i % n_diseases
                target_disease_edges.append((i, disease_id))
            
            edges['target_associated_with_disease'] = target_disease_edges
            logger.info(f"Created {len(target_disease_edges)} target