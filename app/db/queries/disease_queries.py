"""
Database queries for disease-related operations.
"""

from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.db.models import Disease, CellLine, TargetDisease, Target
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class DiseaseQueries:
    """Disease-related database queries."""
    
    @staticmethod
    def get_disease_by_id(db: Session, disease_id: int) -> Optional[Disease]:
        """Get disease by ID."""
        return db.query(Disease).filter(Disease.disease_id == disease_id).first()
    
    @staticmethod
    def get_disease_by_name(db: Session, disease_name: str) -> Optional[Disease]:
        """Get disease by name (case-insensitive)."""
        return db.query(Disease).filter(
            func.lower(Disease.disease_name) == func.lower(disease_name)
        ).first()
    
    @staticmethod
    def get_all_diseases(db: Session) -> List[Disease]:
        """Get all diseases."""
        return db.query(Disease).all()
    
    @staticmethod
    def get_disease_cell_lines(db: Session, disease_id: int) -> List[CellLine]:
        """Get all cell lines for a disease."""
        return db.query(CellLine).filter(
            CellLine.disease_id == disease_id
        ).all()
    
    @staticmethod
    def get_disease_targets(db: Session, disease_id: int) -> List[Dict[str, Any]]:
        """Get all targets associated with a disease."""
        results = db.query(
            Target.target_id,
            Target.target_name,
            Target.gene_name,
            TargetDisease.association_score,
            TargetDisease.evidence_type
        ).join(
            TargetDisease, Target.target_id == TargetDisease.target_id
        ).filter(
            TargetDisease.disease_id == disease_id
        ).all()
        
        return [
            {
                'target_id': r.target_id,
                'target_name': r.target_name,
                'gene_name': r.gene_name,
                'association_score': r.association_score,
                'evidence_type': r.evidence_type
            }
            for r in results
        ]
    
    @staticmethod
    def search_diseases(db: Session, search_term: str, limit: int = 50) -> List[Disease]:
        """Search diseases by name or type."""
        search_pattern = f"%{search_term}%"
        return db.query(Disease).filter(
            func.lower(Disease.disease_name).like(func.lower(search_pattern)) |
            func.lower(Disease.disease_type).like(func.lower(search_pattern))
        ).limit(limit).all()
    
    @staticmethod
    def insert_disease(db: Session, disease_data: Dict[str, Any]) -> Disease:
        """Insert a new disease."""
        disease = Disease(**disease_data)
        db.add(disease)
        db.commit()
        db.refresh(disease)
        logger.info(f"Inserted disease: {disease.disease_name} (ID: {disease.disease_id})")
        return disease
    
    @staticmethod
    def bulk_insert_diseases(db: Session, diseases_data: List[Dict[str, Any]]) -> int:
        """Bulk insert diseases."""
        diseases = [Disease(**data) for data in diseases_data]
        db.bulk_save_objects(diseases)
        db.commit()
        logger.info(f"Bulk inserted {len(diseases)} diseases")
        return len(diseases)