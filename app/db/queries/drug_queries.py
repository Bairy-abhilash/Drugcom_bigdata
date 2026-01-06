"""
Database queries for drug-related operations.
"""

from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import func, or_

from app.db.models import Drug, DrugTarget, Target, SideEffect
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class DrugQueries:
    """Drug-related database queries."""
    
    @staticmethod
    def get_drug_by_id(db: Session, drug_id: int) -> Optional[Drug]:
        """
        Get drug by ID.
        
        Args:
            db: Database session
            drug_id: Drug ID
            
        Returns:
            Drug instance or None
        """
        return db.query(Drug).filter(Drug.drug_id == drug_id).first()
    
    @staticmethod
    def get_drug_by_name(db: Session, drug_name: str) -> Optional[Drug]:
        """
        Get drug by name (case-insensitive).
        
        Args:
            db: Database session
            drug_name: Drug name
            
        Returns:
            Drug instance or None
        """
        return db.query(Drug).filter(
            func.lower(Drug.drug_name) == func.lower(drug_name)
        ).first()
    
    @staticmethod
    def get_all_drugs(db: Session, limit: int = 1000) -> List[Drug]:
        """
        Get all drugs.
        
        Args:
            db: Database session
            limit: Maximum number of results
            
        Returns:
            List of Drug instances
        """
        return db.query(Drug).limit(limit).all()
    
    @staticmethod
    def search_drugs(db: Session, search_term: str, limit: int = 50) -> List[Drug]:
        """
        Search drugs by name.
        
        Args:
            db: Database session
            search_term: Search term
            limit: Maximum number of results
            
        Returns:
            List of matching Drug instances
        """
        search_pattern = f"%{search_term}%"
        return db.query(Drug).filter(
            Drug.drug_name.ilike(search_pattern)
        ).limit(limit).all()
    
    @staticmethod
    def get_drugs_with_smiles(db: Session) -> List[Drug]:
        """
        Get all drugs that have SMILES strings.
        
        Args:
            db: Database session
            
        Returns:
            List of Drug instances with SMILES
        """
        return db.query(Drug).filter(Drug.smiles.isnot(None)).all()
    
    @staticmethod
    def get_drug_targets(db: Session, drug_id: int) -> List[Dict[str, Any]]:
        """
        Get all targets for a specific drug.
        
        Args:
            db: Database session
            drug_id: Drug ID
            
        Returns:
            List of target information dictionaries
        """
        results = db.query(
            Target.target_id,
            Target.target_name,
            Target.gene_name,
            DrugTarget.interaction_type,
            DrugTarget.binding_affinity
        ).join(
            DrugTarget, Target.target_id == DrugTarget.target_id
        ).filter(
            DrugTarget.drug_id == drug_id
        ).all()
        
        return [
            {
                'target_id': r.target_id,
                'target_name': r.target_name,
                'gene_name': r.gene_name,
                'interaction_type': r.interaction_type,
                'binding_affinity': r.binding_affinity
            }
            for r in results
        ]
    
    @staticmethod
    def get_drug_side_effects(db: Session, drug_id: int) -> List[SideEffect]:
        """
        Get side effects for a drug.
        
        Args:
            db: Database session
            drug_id: Drug ID
            
        Returns:
            List of SideEffect instances
        """
        return db.query(SideEffect).filter(
            SideEffect.drug_id == drug_id
        ).all()
    
    @staticmethod
    def get_drugs_by_target(db: Session, target_id: int) -> List[Drug]:
        """
        Get all drugs targeting a specific target.
        
        Args:
            db: Database session
            target_id: Target ID
            
        Returns:
            List of Drug instances
        """
        return db.query(Drug).join(
            DrugTarget, Drug.drug_id == DrugTarget.drug_id
        ).filter(
            DrugTarget.target_id == target_id
        ).all()
    
    @staticmethod
    def get_drugs_for_disease(db: Session, disease_id: int) -> List[Drug]:
        """
        Get drugs associated with a disease through targets.
        
        Args:
            db: Database session
            disease_id: Disease ID
            
        Returns:
            List of Drug instances
        """
        from app.db.models import TargetDisease
        
        return db.query(Drug).join(
            DrugTarget, Drug.drug_id == DrugTarget.drug_id
        ).join(
            Target, DrugTarget.target_id == Target.target_id
        ).join(
            TargetDisease, Target.target_id == TargetDisease.target_id
        ).filter(
            TargetDisease.disease_id == disease_id
        ).distinct().all()
    
    @staticmethod
    def insert_drug(db: Session, drug_data: Dict[str, Any]) -> Drug:
        """
        Insert a new drug into the database.
        
        Args:
            db: Database session
            drug_data: Dictionary containing drug information
            
        Returns:
            Created Drug instance
        """
        drug = Drug(**drug_data)
        db.add(drug)
        db.commit()
        db.refresh(drug)
        logger.info(f"Inserted drug: {drug.drug_name} (ID: {drug.drug_id})")
        return drug
    
    @staticmethod
    def bulk_insert_drugs(db: Session, drugs_data: List[Dict[str, Any]]) -> int:
        """
        Bulk insert drugs.
        
        Args:
            db: Database session
            drugs_data: List of drug data dictionaries
            
        Returns:
            Number of drugs inserted
        """
        drugs = [Drug(**data) for data in drugs_data]
        db.bulk_save_objects(drugs)
        db.commit()
        logger.info(f"Bulk inserted {len(drugs)} drugs")
        return len(drugs)