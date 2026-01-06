"""
Construct heterogeneous drug-target-disease graph using DGL.
"""
import dgl
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DrugGraphConstructor:
    """Construct heterogeneous graph for drug synergy prediction."""
    
    def __init__(self):
        """Initialize graph constructor."""
        self.node_types = ['drug', 'target', 'disease']
        self.edge_types = [
            ('drug', 'targets', 'target'),
            ('target', 'targeted_by', 'drug'),
            ('target', 'associated_with', 'disease'),
            ('disease', 'has_target', 'target'),
            ('drug', 'similar_to', 'drug'),
            ('drug', 'treats', 'disease'),
            ('disease', 'treated_by', 'drug')
        ]
    
    def build_heterograph(
        self,
        drug_features: np.ndarray,
        target_features: np.ndarray,
        disease_features: np.ndarray,
        drug_target_edges: List[Tuple[int, int]],
        target_disease_edges: List[Tuple[int, int]],
        drug_similarity_edges: List[Tuple[int, int]],
        drug_disease_edges: List[Tuple[int, int]]
    ) -> dgl.DGLHeteroGraph:
        """
        Build heterogeneous graph.
        
        Args:
            drug_features: Drug node features (N_drugs x feature_dim)
            target_features: Target node features (N_targets x feature_dim)
            disease_features: Disease node features (N_diseases x feature_dim)
            drug_target_edges: List of (drug_id, target_id) tuples
            target_disease_edges: List of (target_id, disease_id) tuples
            drug_similarity_edges: List of (drug_id, drug_id) tuples
            drug_disease_edges: List of (drug_id, disease_id) tuples
            
        Returns:
            DGL heterogeneous graph
        """
        graph_data = {}
        
        # Drug -> Target edges
        if drug_target_edges:
            dt_src, dt_dst = zip(*drug_target_edges)
            graph_data[('drug', 'targets', 'target')] = (torch.tensor(dt_src), torch.tensor(dt_dst))
            graph_data[('target', 'targeted_by', 'drug')] = (torch.tensor(dt_dst), torch.tensor(dt_src))
        
        # Target -> Disease edges
        if target_disease_edges:
            td_src, td_dst = zip(*target_disease_edges)
            graph_data[('target', 'associated_with', 'disease')] = (torch.tensor(td_src), torch.tensor(td_dst))
            graph_data[('disease', 'has_target', 'target')] = (torch.tensor(td_dst), torch.tensor(td_src))
        
        # Drug -> Drug similarity edges
        if drug_similarity_edges:
            dd_src, dd_dst = zip(*drug_similarity_edges)
            graph_data[('drug', 'similar_to', 'drug')] = (torch.tensor(dd_src), torch.tensor(dd_dst))
        
        # Drug -> Disease edges
        if drug_disease_edges:
            dd_src, dd_dst = zip(*drug_disease_edges)
            graph_data[('drug', 'treats', 'disease')] = (torch.tensor(dd_src), torch.tensor(dd_dst))
            graph_data[('disease', 'treated_by', 'drug')] = (torch.tensor(dd_dst), torch.tensor(dd_src))
        
        # Create heterograph
        g = dgl.heterograph(graph_data)
        
        # Add node features
        g.nodes['drug'].data['feat'] = torch.tensor(drug_features, dtype=torch.float32)
        g.nodes['target'].data['feat'] = torch.tensor(target_features, dtype=torch.float32)
        g.nodes['disease'].data['feat'] = torch.tensor(disease_features, dtype=torch.float32)
        
        logger.info(f"Built heterograph with {g.num_nodes()} nodes and {g.num_edges()} edges")
        logger.info(f"Node types: {g.ntypes}")
        logger.info(f"Edge types: {g.canonical_etypes}")
        
        return g
    
    def compute_drug_similarity(
        self,
        drug_features: np.ndarray,
        threshold: float = 0.7
    ) -> List[Tuple[int, int]]:
        """
        Compute drug similarity edges based on feature similarity.
        
        Args:
            drug_features: Drug feature matrix
            threshold: Similarity threshold for edge creation
            
        Returns:
            List of (drug_id, drug_id) tuples
        """
        from sklearn.metrics.pairwise import cosine_similarity
        
        similarity_matrix = cosine_similarity(drug_features)
        edges = []
        
        n_drugs = len(drug_features)
        for i in range(n_drugs):
            for j in range(i + 1, n_drugs):
                if similarity_matrix[i, j] >= threshold:
                    edges.append((i, j))
                    edges.append((j, i))  # Add reverse edge
        
        logger.info(f"Created {len(edges)} drug similarity edges (threshold={threshold})")
        return edges
    
    def add_synergy_labels(
        self,
        g: dgl.DGLHeteroGraph,
        synergy_data: pd.DataFrame
    ) -> dgl.DGLHeteroGraph:
        """
        Add synergy labels to drug pairs.
        
        Args:
            g: Heterogeneous graph
            synergy_data: DataFrame with columns ['drug1_id', 'drug2_id', 'synergy_score']
            
        Returns:
            Graph with synergy edge type added
        """
        drug1_ids = torch.tensor(synergy_data['drug1_id'].values)
        drug2_ids = torch.tensor(synergy_data['drug2_id'].values)
        synergy_scores = torch.tensor(synergy_data['synergy_score'].values, dtype=torch.float32)
        
        # Add synergy edges
        g = dgl.add_edges(
            g,
            drug1_ids,
            drug2_ids,
            etype=('drug', 'synergistic_with', 'drug'),
            data={'label': synergy_scores}
        )
        
        logger.info(f"Added {len(synergy_data)} synergy labels")
        return g


# Example usage
if __name__ == "__main__":
    # Create dummy data
    n_drugs = 100
    n_targets = 50
    n_diseases = 20
    
    drug_features = np.random.rand(n_drugs, 2048)
    target_features = np.random.rand(n_targets, 128)
    disease_features = np.random.rand(n_diseases, 64)
    
    drug_target_edges = [(i, j) for i in range(n_drugs) for j in range(min(3, n_targets))]
    target_disease_edges = [(i, j % n_diseases) for i in range(n_targets)]
    drug_disease_edges = [(i, i % n_diseases) for i in range(n_drugs)]
    
    constructor = DrugGraphConstructor()
    drug_sim_edges = constructor.compute_drug_similarity(drug_features, threshold=0.8)
    
    g = constructor.build_heterograph(
        drug_features, target_features, disease_features,
        drug_target_edges, target_disease_edges, drug_sim_edges, drug_disease_edges
    )
    
    print(f"Graph: {g}")