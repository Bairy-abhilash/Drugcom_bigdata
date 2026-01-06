"""
Safety checker module for drug combinations.
"""

from typing import Dict, Any, Optional
from sqlalchemy.orm import Session

from app.db.queries.synergy_queries import SynergyQueries
from app.db.queries.drug_queries import DrugQueries
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class SafetyChecker:
    """Check safety of drug combinations."""
    
    def __init__(self, db: Session):
        """
        Initialize safety checker.
        
        Args:
            db: Database session
        """
        self.db = db
        self.synergy_queries = SynergyQueries()
        self.drug_queries = DrugQueries()
    
    def check_combination_safety(
        self,
        drug1_id: int,
        drug2_id: int
    ) -> Dict[str, Any]:
        """
        Check if a drug combination is safe.
        
        Args:
            drug1_id: First drug ID
            drug2_id: Second drug ID
            
        Returns:
            Dictionary containing safety information:
            {
                'is_safe': bool,
                'severity': str or None,
                'interaction_type': str or None,
                'description': str or None,
                'mechanism': str or None,
                'clinical_effect': str or None
            }
        """
        # Check harmful combinations database
        harmful = self.synergy_queries.check_harmful_combination(
            self.db, drug1_id, drug2_id
        )
        
        if harmful:
            logger.warning(
                f"Harmful combination detected: Drug {drug1_id} + Drug {drug2_id} "
                f"(Severity: {harmful.severity_level})"
            )
            
            return {
                'is_safe': False,
                'severity': harmful.severity_level,
                'interaction_type': harmful.interaction_type,
                'description': harmful.description,
                'mechanism': harmful.mechanism,
                'clinical_effect': harmful.clinical_effect
            }
        
        # If not in harmful combinations, assume safe
        return {
            'is_safe': True,
            'severity': None,
            'interaction_type': None,
            'description': 'No known harmful interactions',
            'mechanism': None,
            'clinical_effect': None
        }
    
    def get_safety_score(self, drug1_id: int, drug2_id: int) -> float:
        """
        Get a numerical safety score (0-1, higher is safer).
        
        Args:
            drug1_id: First drug ID
            drug2_id: Second drug ID
            
        Returns:
            Safety score between 0 and 1
        """
        safety_info = self.check_combination_safety(drug1_id, drug2_id)
        
        if safety_info['is_safe']:
            return 1.0
        
        # Map severity to score
        severity_scores = {
            'mild': 0.7,
            'moderate': 0.4,
            'severe': 0.2,
            'contraindicated': 0.0
        }
        
        severity = safety_info['severity']
        return severity_scores.get(severity.lower() if severity else '', 0.5)
    
    def get_drug_interaction_warnings(self, drug_id: int) -> Dict[str, Any]:
        """
        Get all interaction warnings for a specific drug.
        
        Args:
            drug_id: Drug ID
            
        Returns:
            Dictionary with warnings information
        """
        # Query all harmful combinations involving this drug
        harmful_combinations = self.db.query(
            self.db.models.HarmfulCombination
        ).filter(
            or_(
                self.db.models.HarmfulCombination.drug1_id == drug_id,
                self.db.models.HarmfulCombination.drug2_id == drug_id
            )
        ).all()
        
        warnings = []
        for combo in harmful_combinations:
            other_drug_id = combo.drug2_id if combo.drug1_id == drug_id else combo.drug1_id
            other_drug = self.drug_queries.get_drug_by_id(self.db, other_drug_id)
            
            warnings.append({
                'other_drug_id': other_drug_id,
                'other_drug_name': other_drug.drug_name if other_drug else 'Unknown',
                'severity': combo.severity_level,
                'interaction_type': combo.interaction_type,
                'description': combo.description
            })
        
        return {
            'drug_id': drug_id,
            'num_warnings': len(warnings),
            'warnings': warnings
        }