"""
Heterogeneous graph construction module using DGL.
"""

import dgl
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from sqlalchemy.orm import Session

from app.db.models import Drug, Target, Disease, DrugTarget, TargetDisease
from app.db.queries.drug_queries import DrugQueries
from app.db.queries.disease_queries import DiseaseQueries
from app.graph.features import FeatureGenerator
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class HeterogeneousGraphBuilder:
    """Build heterogeneous graph for drug synergy prediction."""
    
    def __init__(self, db: Session):
        """
        Initialize graph builder.
        
        Args:
            db: Database session
        """
        self.db = db
        self.feature_generator = FeatureGenerator()
        self.drug_queries = DrugQueries()
        self.disease_queries = DiseaseQueries()
    
    def build_graph_for_disease(self, disease_id: int) -> Tuple[dgl.DGLGraph, Dict]:
        """
        Build a heterogeneous graph for a specific disease.
        
        Args:
            disease_id: Disease ID
            
        Returns:
            Tuple of (DGL graph, metadata dictionary)
        """
        logger.info(f"Building graph for disease ID: {disease_id}")
        
        # Get disease-related data
        disease = self.disease_queries.get_disease_by_id(self.db, disease_id)
        if not disease:
            raise ValueError(f"Disease {disease_id} not found")
        
        # Get drugs for this disease
        drugs = self.drug_queries.get_drugs_for_disease(self.db, disease_id)
        logger.info(f"Found {len(drugs)} drugs for disease")
        
        # Get targets
        disease_targets = self.disease_queries.get_disease_targets(self.db, disease_id)
        target_ids = [t['target_id'] for t in disease_targets]
        targets = self.db.query(Target).filter(Target.target_id.in_(target_ids)).all()
        logger.info(f"Found {len(targets)} targets for disease")
        
        # Create node ID mappings
        drug_id_map = {drug.drug_id: idx for idx, drug in enumerate(drugs)}
        target_id_map = {target.target_id: idx for idx, target in enumerate(targets)}
        disease_id_map = {disease.disease_id: 0}  # Single disease node
        
        # Collect edges
        drug_target_edges = self._get_drug_target_edges(drugs, targets, drug_id_map, target_id_map)
        target_disease_edges = self._get_target_disease_edges(disease_id, target_ids, target_id_map)
        
        # Build graph data dictionary
        graph_data = {
            ('drug', 'targets', 'target'): drug_target_edges['drug_target'],
            ('target', 'targeted_by', 'drug'): drug_target_edges['target_drug'],
            ('target', 'associated_with', 'disease'): target_disease_edges['target_disease'],
            ('disease', 'has_target', 'target'): target_disease_edges['disease_target'],
        }
        
        # Create heterogeneous graph
        num_nodes_dict = {
            'drug': len(drugs),
            'target': len(targets),
            'disease': 1
        }
        
        g = dgl.heterograph(graph_data, num_nodes_dict=num_nodes_dict)
        
        # Add node features
        g = self._add_node_features(g, drugs, targets, [disease])
        
        # Metadata
        metadata = {
            'disease_id': disease_id,
            'disease_name': disease.disease_name,
            'num_drugs': len(drugs),
            'num_targets': len(targets),
            'drug_ids': [drug.drug_id for drug in drugs],
            'target_ids': target_ids,
            'drug_id_map': drug_id_map,
            'target_id_map': target_id_map
        }
        
        logger.info(f"Graph built: {g}")
        return g, metadata
    
    def build_full_graph(self, limit_drugs: Optional[int] = None) -> Tuple[dgl.DGLGraph, Dict]:
        """
        Build a complete heterogeneous graph with all drugs, targets, and diseases.
        
        Args:
            limit_drugs: Optional limit on number of drugs to include
            
        Returns:
            Tuple of (DGL graph, metadata dictionary)
        """
        logger.info("Building full heterogeneous graph")
        
        # Get all entities
        drugs = self.drug_queries.get_all_drugs(self.db, limit=limit_drugs or 10000)
        targets = self.db.query(Target).all()
        diseases = self.disease_queries.get_all_diseases(self.db)
        
        logger.info(f"Entities: {len(drugs)} drugs, {len(targets)} targets, {len(diseases)} diseases")
        
        # Create mappings
        drug_id_map = {drug.drug_id: idx for idx, drug in enumerate(drugs)}
        target_id_map = {target.target_id: idx for idx, target in enumerate(targets)}
        disease_id_map = {disease.disease_id: idx for idx, disease in enumerate(diseases)}
        
        # Collect all edges
        drug_ids = [drug.drug_id for drug in drugs]
        target_ids = [target.target_id for target in targets]
        disease_ids = [disease.disease_id for disease in diseases]
        
        drug_target_edges = self._get_drug_target_edges(drugs, targets, drug_id_map, target_id_map)
        target_disease_edges = self._get_all_target_disease_edges(disease_ids, target_id_map, disease_id_map)
        
        # Build graph
        graph_data = {
            ('drug', 'targets', 'target'): drug_target_edges['drug_target'],
            ('target', 'targeted_by', 'drug'): drug_target_edges['target_drug'],
            ('target', 'associated_with', 'disease'): target_disease_edges['target_disease'],
            ('disease', 'has_target', 'target'): target_disease_edges['disease_target'],
        }
        
        num_nodes_dict = {
            'drug': len(drugs),
            'target': len(targets),
            'disease': len(diseases)
        }
        
        g = dgl.heterograph(graph_data, num_nodes_dict=num_nodes_dict)
        g = self._add_node_features(g, drugs, targets, diseases)
        
        metadata = {
            'num_drugs': len(drugs),
            'num_targets': len(targets),
            'num_diseases': len(diseases),
            'drug_id_map': drug_id_map,
            'target_id_map': target_id_map,
            'disease_id_map': disease_id_map
        }
        
        logger.info(f"Full graph built: {g}")
        return g, metadata
    
    def _get_drug_target_edges(
        self, 
        drugs: List[Drug], 
        targets: List[Target],
        drug_id_map: Dict,
        target_id_map: Dict
    ) -> Dict:
        """Get drug-target edges."""
        drug_ids = [drug.drug_id for drug in drugs]
        target_ids = [target.target_id for target in targets]
        
        # Query drug-target relationships
        drug_target_relations = self.db.query(DrugTarget).filter(
            DrugTarget.drug_id.in_(drug_ids),
            DrugTarget.target_id.in_(target_ids)
        ).all()
        
        drug_target_src = []
        drug_target_dst = []
        
        for rel in drug_target_relations:
            if rel.drug_id in drug_id_map and rel.target_id in target_id_map:
                drug_target_src.append(drug_id_map[rel.drug_id])
                drug_target_dst.append(target_id_map[rel.target_id])
        
        return {
            'drug_target': (torch.tensor(drug_target_src), torch.tensor(drug_target_dst)),
            'target_drug': (torch.tensor(drug_target_dst), torch.tensor(drug_target_src))
        }
    
    def _get_target_disease_edges(
        self,
        disease_id: int,
        target_ids: List[int],
        target_id_map: Dict
    ) -> Dict:
        """Get target-disease edges for a specific disease."""
        target_disease_relations = self.db.query(TargetDisease).filter(
            TargetDisease.disease_id == disease_id,
            TargetDisease.target_id.in_(target_ids)
        ).all()
        
        target_disease_src = []
        target_disease_dst = []
        
        for rel in target_disease_relations:
            if rel.target_id in target_id_map:
                target_disease_src.append(target_id_map[rel.target_id])
                target_disease_dst.append(0)  # Single disease node
        
        return {
            'target_disease': (torch.tensor(target_disease_src), torch.tensor(target_disease_dst)),
            'disease_target': (torch.tensor(target_disease_dst), torch.tensor(target_disease_src))
        }
    
    def _get_all_target_disease_edges(
        self,
        disease_ids: List[int],
        target_id_map: Dict,
        disease_id_map: Dict
    ) -> Dict:
        """Get all target-disease edges."""
        target_disease_relations = self.db.query(TargetDisease).filter(
            TargetDisease.disease_id.in_(disease_ids)
        ).all()
        
        target_disease_src = []
        target_disease_dst = []
        
        for rel in target_disease_relations:
            if rel.target_id in target_id_map and rel.disease_id in disease_id_map:
                target_disease_src.append(target_id_map[rel.target_id])
                target_disease_dst.append(disease_id_map[rel.disease_id])
        
        return {
            'target_disease': (torch.tensor(target_disease_src), torch.tensor(target_disease_dst)),
            'disease_target': (torch.tensor(target_disease_dst), torch.tensor(target_disease_src))
        }
    
    def _add_node_features(
        self,
        g: dgl.DGLGraph,
        drugs: List[Drug],
        targets: List[Target],
        diseases: List[Disease]
    ) -> dgl.DGLGraph:
        """Add features to graph nodes."""
        # Drug features
        drug_features = []
        for drug in drugs:
            if drug.smiles:
                feat = self.feature_generator.generate_drug_features(drug.smiles)
            else:
                feat = np.zeros(self.feature_generator.drug_feature_dim)
            drug_features.append(feat)
        
        g.nodes['drug'].data['features'] = torch.FloatTensor(np.array(drug_features))
        
        # Target features
        target_features = []
        for target in targets:
            feat = self.feature_generator.generate_target_features(target)
            target_features.append(feat)
        
        g.nodes['target'].data['features'] = torch.FloatTensor(np.array(target_features))
        
        # Disease features
        disease_features = []
        for disease in diseases:
            feat = self.feature_generator.generate_disease_features(disease)
            disease_features.append(feat)
        
        g.nodes['disease'].data['features'] = torch.FloatTensor(np.array(disease_features))
        
        return g