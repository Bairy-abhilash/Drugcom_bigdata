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
        self.db = db
        self.feature_generator = FeatureGenerator()
        self.drug_queries = DrugQueries()
        self.disease_queries = DiseaseQueries()

    # ------------------------------------------------------------------
    # SAFETY HELPERS
    # ------------------------------------------------------------------

    def _safe_edges(self, src: List[int], dst: List[int], rel_name: str):
        """
        Prevent DGL crashes by skipping empty edge lists.
        """
        if not src or not dst:
            logger.warning(f"Skipping empty edge relation: {rel_name}")
            return None
        return (
            torch.tensor(src, dtype=torch.int64),
            torch.tensor(dst, dtype=torch.int64)
        )

    # ------------------------------------------------------------------
    # PUBLIC GRAPH BUILDERS
    # ------------------------------------------------------------------

    def build_graph_for_disease(self, disease_id: int) -> Tuple[dgl.DGLGraph, Dict]:
        logger.info(f"Building graph for disease ID: {disease_id}")

        disease = self.disease_queries.get_disease_by_id(self.db, disease_id)
        if not disease:
            raise ValueError(f"Disease {disease_id} not found")

        drugs = self.drug_queries.get_drugs_for_disease(self.db, disease_id)
        logger.info(f"Found {len(drugs)} drugs for disease")

        disease_targets = self.disease_queries.get_disease_targets(self.db, disease_id)
        target_ids = [t["target_id"] for t in disease_targets]

        targets = (
            self.db.query(Target)
            .filter(Target.target_id.in_(target_ids))
            .all()
        )
        logger.info(f"Found {len(targets)} targets for disease")

        if not drugs or not targets:
            raise ValueError("Cannot build graph: disease has no drugs or targets")

        drug_id_map = {d.drug_id: i for i, d in enumerate(drugs)}
        target_id_map = {t.target_id: i for i, t in enumerate(targets)}

        drug_target_edges = self._get_drug_target_edges(
            drugs, targets, drug_id_map, target_id_map
        )

        target_disease_edges = self._get_target_disease_edges(
            disease_id, target_ids, target_id_map
        )

        graph_data = {}

        if drug_target_edges["drug_target"]:
            graph_data[("drug", "targets", "target")] = drug_target_edges["drug_target"]
            graph_data[("target", "targeted_by", "drug")] = drug_target_edges["target_drug"]

        if target_disease_edges["target_disease"]:
            graph_data[("target", "associated_with", "disease")] = target_disease_edges["target_disease"]
            graph_data[("disease", "has_target", "target")] = target_disease_edges["disease_target"]

        if not graph_data:
            raise ValueError("Graph has no valid edges — cannot build DGL graph")

        num_nodes_dict = {
            "drug": len(drugs),
            "target": len(targets),
            "disease": 1,
        }

        g = dgl.heterograph(graph_data, num_nodes_dict=num_nodes_dict)
        g = self._add_node_features(g, drugs, targets, [disease])

        metadata = {
            "disease_id": disease_id,
            "disease_name": disease.disease_name,
            "num_drugs": len(drugs),
            "num_targets": len(targets),
            "drug_ids": [d.drug_id for d in drugs],
            "target_ids": target_ids,
            "drug_id_map": drug_id_map,
            "target_id_map": target_id_map,
        }

        logger.info(f"Graph built successfully: {g}")
        return g, metadata

    # ------------------------------------------------------------------

    def build_full_graph(self, limit_drugs: Optional[int] = None) -> Tuple[dgl.DGLGraph, Dict]:
        logger.info("Building full heterogeneous graph")

        drugs = self.drug_queries.get_all_drugs(self.db, limit=limit_drugs or 10000)
        targets = self.db.query(Target).all()
        diseases = self.disease_queries.get_all_diseases(self.db)

        if not drugs or not targets or not diseases:
            raise ValueError("Cannot build full graph: missing entities")

        drug_id_map = {d.drug_id: i for i, d in enumerate(drugs)}
        target_id_map = {t.target_id: i for i, t in enumerate(targets)}
        disease_id_map = {d.disease_id: i for i, d in enumerate(diseases)}

        drug_target_edges = self._get_drug_target_edges(
            drugs, targets, drug_id_map, target_id_map
        )

        target_disease_edges = self._get_all_target_disease_edges(
            list(disease_id_map.keys()),
            target_id_map,
            disease_id_map,
        )

        graph_data = {}

        if drug_target_edges["drug_target"]:
            graph_data[("drug", "targets", "target")] = drug_target_edges["drug_target"]
            graph_data[("target", "targeted_by", "drug")] = drug_target_edges["target_drug"]

        if target_disease_edges["target_disease"]:
            graph_data[("target", "associated_with", "disease")] = target_disease_edges["target_disease"]
            graph_data[("disease", "has_target", "target")] = target_disease_edges["disease_target"]

        if not graph_data:
            raise ValueError("Full graph has no valid edges")

        num_nodes_dict = {
            "drug": len(drugs),
            "target": len(targets),
            "disease": len(diseases),
        }

        g = dgl.heterograph(graph_data, num_nodes_dict=num_nodes_dict)
        g = self._add_node_features(g, drugs, targets, diseases)

        metadata = {
            "num_drugs": len(drugs),
            "num_targets": len(targets),
            "num_diseases": len(diseases),
            "drug_id_map": drug_id_map,
            "target_id_map": target_id_map,
            "disease_id_map": disease_id_map,
        }

        logger.info(f"Full graph built successfully: {g}")
        return g, metadata

    # ------------------------------------------------------------------
    # EDGE BUILDERS
    # ------------------------------------------------------------------

    def _get_drug_target_edges(self, drugs, targets, drug_id_map, target_id_map):
        drug_ids = [d.drug_id for d in drugs]
        target_ids = [t.target_id for t in targets]

        relations = (
            self.db.query(DrugTarget)
            .filter(
                DrugTarget.drug_id.in_(drug_ids),
                DrugTarget.target_id.in_(target_ids),
            )
            .all()
        )

        src, dst = [], []
        for r in relations:
            src.append(drug_id_map[r.drug_id])
            dst.append(target_id_map[r.target_id])

        return {
            "drug_target": self._safe_edges(src, dst, "drug-target"),
            "target_drug": self._safe_edges(dst, src, "target-drug"),
        }

    # ------------------------------------------------------------------

    def _get_target_disease_edges(self, disease_id, target_ids, target_id_map):
        relations = (
            self.db.query(TargetDisease)
            .filter(
                TargetDisease.disease_id == disease_id,
                TargetDisease.target_id.in_(target_ids),
            )
            .all()
        )

        src, dst = [], []
        for r in relations:
            src.append(target_id_map[r.target_id])
            dst.append(0)

        return {
            "target_disease": self._safe_edges(src, dst, "target-disease"),
            "disease_target": self._safe_edges(dst, src, "disease-target"),
        }

    # ------------------------------------------------------------------

    def _get_all_target_disease_edges(self, disease_ids, target_id_map, disease_id_map):
        relations = (
            self.db.query(TargetDisease)
            .filter(TargetDisease.disease_id.in_(disease_ids))
            .all()
        )

        src, dst = [], []
        for r in relations:
            if r.target_id in target_id_map and r.disease_id in disease_id_map:
                src.append(target_id_map[r.target_id])
                dst.append(disease_id_map[r.disease_id])

        return {
            "target_disease": self._safe_edges(src, dst, "target-disease"),
            "disease_target": self._safe_edges(dst, src, "disease-target"),
        }

    # ------------------------------------------------------------------
    # FEATURE ATTACHMENT
    # ------------------------------------------------------------------

    def _add_node_features(self, g, drugs, targets, diseases):
        drug_feats = []
        for d in drugs:
            if d.smiles:
                drug_feats.append(
                    self.feature_generator.generate_drug_features(d.smiles)
                )
            else:
                drug_feats.append(
                    np.zeros(self.feature_generator.drug_feature_dim)
                )

        g.nodes["drug"].data["features"] = torch.tensor(
            np.array(drug_feats), dtype=torch.float32
        )

        target_feats = [
            self.feature_generator.generate_target_features(t)
            for t in targets
        ]
        g.nodes["target"].data["features"] = torch.tensor(
            np.array(target_feats), dtype=torch.float32
        )

        disease_feats = [
            self.feature_generator.generate_disease_features(d)
            for d in diseases
        ]
        g.nodes["disease"].data["features"] = torch.tensor(
            np.array(disease_feats), dtype=torch.float32
        )

        return g
