"""
Database queries for synergy-related operations.
"""

from typing import List, Optional, Dict, Any, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_

from app.db.models import SynergyScore, Drug, CellLine, HarmfulCombination
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class SynergyQueries:
    """Synergy-related database queries."""
    
    @staticmethod
    def get_synergy_scores(
        db: Session,
        drug1_id: Optional[int] = None,
        drug2_id: Optional[int] = None,
        cell_line_id: Optional[int] = None,
        min_score: Optional[float] = None
    ) -> List[SynergyScore]:
        """
        Get synergy scores with optional filters.
        
        Args:
            db: Database session
            drug1_id: First drug ID
            drug2_id: Second drug ID
            cell_line_id: Cell line ID
            min_score: Minimum synergy score
            
        Returns:
            List of SynergyScore instances
        """
        query = db.query(SynergyScore)
        
        if drug1_id is not None and drug2_id is not None:
            query = query.filter(
                or_(
                    and_(SynergyScore.drug1_id == drug1_id, SynergyScore.drug2_id == drug2_id),
                    and_(SynergyScore.drug1_id == drug2_id, SynergyScore.drug2_id == drug1_id)
                )
            )
        elif drug1_id is not None:
            query = query.filter(
                or_(SynergyScore.drug1_id == drug1_id, SynergyScore.drug2_id == drug1_id)
            )
        elif drug2_id is not None:
            query = query.filter(
                or_(SynergyScore.drug1_id == drug2_id, SynergyScore.drug2_id == drug2_id)
            )
        
        if cell_line_id is not None:
            query = query.filter(SynergyScore.cell_line_id == cell_line_id)
        
        if min_score is not None:
            query = query.filter(SynergyScore.synergy_score >= min_score)
        
        return query.all()
    
    @staticmethod
    def get_avg_synergy_for_pair(
        db: Session,
        drug1_id: int,
        drug2_id: int
    ) -> Optional[float]:
        """
        Get average synergy score for a drug pair across all cell lines.
        
        Args:
            db: Database session
            drug1_id: First drug ID
            drug2_id: Second drug ID
            
        Returns:
            Average synergy score or None
        """
        result = db.query(func.avg(SynergyScore.synergy_score)).filter(
            or_(
                and_(SynergyScore.drug1_id == drug1_id, SynergyScore.drug2_id == drug2_id),
                and_(SynergyScore.drug1_id == drug2_id, SynergyScore.drug2_id == drug1_id)
            )
        ).scalar()
        
        return float(result) if result is not None else None
    
    @staticmethod
    def get_top_synergistic_pairs(
        db: Session,
        disease_id: Optional[int] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get top synergistic drug pairs.
        
        Args:
            db: Database session
            disease_id: Optional disease ID to filter by
            limit: Number of results to return
            
        Returns:
            List of dictionaries with drug pair information
        """
        query = db.query(
            SynergyScore.drug1_id,
            SynergyScore.drug2_id,
            func.avg(SynergyScore.synergy_score).label('avg_synergy'),
            func.count(SynergyScore.synergy_id).label('num_experiments')
        ).group_by(
            SynergyScore.drug1_id,
            SynergyScore.drug2_id
        )
        
        if disease_id is not None:
            query = query.join(
                CellLine, SynergyScore.cell_line_id == CellLine.cell_line_id
            ).filter(
                CellLine.disease_id == disease_id
            )
        
        results = query.order_by(
            func.avg(SynergyScore.synergy_score).desc()
        ).limit(limit).all()
        
        return [
            {
                'drug1_id': r.drug1_id,
                'drug2_id': r.drug2_id,
                'avg_synergy': float(r.avg_synergy),
                'num_experiments': r.num_experiments
            }
            for r in results
        ]
    
    @staticmethod
    def check_harmful_combination(
        db: Session,
        drug1_id: int,
        drug2_id: int
    ) -> Optional[HarmfulCombination]:
        """
        Check if a drug combination is harmful.
        
        Args:
            db: Database session
            drug1_id: First drug ID
            drug2_id: Second drug ID
            
        Returns:
            HarmfulCombination instance if exists, None otherwise
        """
        # Ensure consistent ordering
        if drug1_id > drug2_id:
            drug1_id, drug2_id = drug2_id, drug1_id
        
        return db.query(HarmfulCombination).filter(
            and_(
                HarmfulCombination.drug1_id == drug1_id,
                HarmfulCombination.drug2_id == drug2_id
            )
        ).first()
    
    @staticmethod
    def get_all_harmful_combinations(db: Session) -> List[HarmfulCombination]:
        """Get all harmful drug combinations."""
        return db.query(HarmfulCombination).all()
    
    @staticmethod
    def insert_synergy_score(db: Session, synergy_data: Dict[str, Any]) -> SynergyScore:
        """Insert a new synergy score."""
        synergy = SynergyScore(**synergy_data)
        db.add(synergy)
        db.commit()
        db.refresh(synergy)
        return synergy
    
    @staticmethod
    def bulk_insert_synergy_scores(
        db: Session,
        synergy_data: List[Dict[str, Any]]
    ) -> int:
        """Bulk insert synergy scores."""
        scores = [SynergyScore(**data) for data in synergy_data]
        db.bulk_save_objects(scores)
        db.commit()
        logger.info(f"Bulk inserted {len(scores)} synergy scores")
        return len(scores)
    
    @staticmethod
    def insert_harmful_combination(
        db: Session,
        combination_data: Dict[str, Any]
    ) -> HarmfulCombination:
        """Insert a harmful drug combination."""
        # Ensure consistent ordering
        if combination_data['drug1_id'] > combination_data['drug2_id']:
            combination_data['drug1_id'], combination_data['drug2_id'] = \
                combination_data['drug2_id'], combination_data['drug1_id']
        
        combination = HarmfulCombination(**combination_data)
        db.add(combination)
        db.commit()
        db.refresh(combination)
        logger.info(f"Inserted harmful combination: {combination.drug1_id}, {combination.drug2_id}")
        return combination