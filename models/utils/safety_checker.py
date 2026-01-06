"""
Drug safety interaction checker using DrugBank data.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SafetyChecker:
    """Check drug combination safety using interaction database."""
    
    def __init__(self, interaction_db_path: str = None):
        """
        Initialize safety checker.
        
        Args:
            interaction_db_path: Path to interaction database CSV
        """
        self.interactions = {}
        self.severity_levels = ['safe', 'minor', 'moderate', 'major', 'contraindicated']
        
        if interaction_db_path:
            self.load_interactions(interaction_db_path)
        else:
            # Load mock interactions for demo
            self.load_mock_interactions()
    
    def load_interactions(self, path: str):
        """Load drug interactions from CSV file."""
        try:
            df = pd.read_csv(path)
            for _, row in df.iterrows():
                drug1 = row['drug1_name'].lower()
                drug2 = row['drug2_name'].lower()
                severity = row['severity'].lower()
                description = row.get('description', '')
                
                key = tuple(sorted([drug1, drug2]))
                self.interactions[key] = {
                    'severity': severity,
                    'description': description
                }
            logger.info(f"Loaded {len(self.interactions)} drug interactions")
        except Exception as e:
            logger.error(f"Error loading interactions: {e}")
            self.load_mock_interactions()
    
    def load_mock_interactions(self):
        """Load mock drug interactions for demonstration."""
        mock_data = [
            ('warfarin', 'aspirin', 'major', 'Increased bleeding risk'),
            ('methotrexate', 'nsaids', 'major', 'Increased methotrexate toxicity'),
            ('digoxin', 'amiodarone', 'moderate', 'Increased digoxin levels'),
            ('simvastatin', 'amlodipine', 'moderate', 'Increased statin levels'),
            ('ace_inhibitors', 'potassium', 'moderate', 'Hyperkalemia risk'),
            ('paclitaxel', 'carboplatin', 'minor', 'Monitor for neutropenia'),
            ('trastuzumab', 'doxorubicin', 'moderate', 'Cardiac toxicity risk'),
            ('bevacizumab', 'sunitinib', 'major', 'Severe hypertension risk'),
        ]
        
        for drug1, drug2, severity, desc in mock_data:
            key = tuple(sorted([drug1, drug2]))
            self.interactions[key] = {
                'severity': severity,
                'description': desc
            }
        
        logger.info(f"Loaded {len(self.interactions)} mock interactions")
    
    def check_interaction(
        self,
        drug1_name: str,
        drug2_name: str
    ) -> Dict[str, str]:
        """
        Check interaction between two drugs.
        
        Args:
            drug1_name: First drug name
            drug2_name: Second drug name
            
        Returns:
            Dictionary with 'level' and 'description'
        """
        drug1 = drug1_name.lower().strip()
        drug2 = drug2_name.lower().strip()
        
        key = tuple(sorted([drug1, drug2]))
        
        if key in self.interactions:
            interaction = self.interactions[key]
            return {
                'level': interaction['severity'],
                'description': interaction['description'],
                'warning': True
            }
        
        # Check for partial matches (e.g., drug classes)
        for (d1, d2), interaction in self.interactions.items():
            if (d1 in drug1 or d1 in drug2) and (d2 in drug1 or d2 in drug2):
                return {
                    'level': interaction['severity'],
                    'description': interaction['description'],
                    'warning': True
                }
        
        return {
            'level': 'safe',
            'description': 'No known significant interactions',
            'warning': False
        }
    
    def batch_check(
        self,
        drug_pairs: List[Tuple[str, str]]
    ) -> List[Dict]:
        """
        Check multiple drug pairs.
        
        Args:
            drug_pairs: List of (drug1, drug2) tuples
            
        Returns:
            List of interaction dictionaries
        """
        results = []
        for drug1, drug2 in drug_pairs:
            result = self.check_interaction(drug1, drug2)
            result['drug1'] = drug1
            result['drug2'] = drug2
            results.append(result)
        
        return results
    
    def get_safety_label(self, severity: str) -> str:
        """
        Convert severity to simple safety label.
        
        Args:
            severity: Severity level
            
        Returns:
            Safety label ('safe', 'caution', 'danger')
        """
        if severity in ['safe', 'minor']:
            return 'safe'
        elif severity in ['moderate']:
            return 'caution'
        else:
            return 'danger'
    
    def get_safety_score(self, severity: str) -> float:
        """
        Convert severity to numerical score (0-1, higher is safer).
        
        Args:
            severity: Severity level
            
        Returns:
            Safety score
        """
        score_map = {
            'safe': 1.0,
            'minor': 0.8,
            'moderate': 0.5,
            'major': 0.2,
            'contraindicated': 0.0
        }
        return score_map.get(severity.lower(), 0.5)


# Example usage
if __name__ == "__main__":
    checker = SafetyChecker()
    
    # Test single interaction
    result = checker.check_interaction('Warfarin', 'Aspirin')
    print(f"Interaction: {result}")
    
    # Test batch
    pairs = [
        ('Paclitaxel', 'Carboplatin'),
        ('Trastuzumab', 'Doxorubicin'),
        ('Aspirin', 'Ibuprofen')
    ]
    results = checker.batch_check(pairs)
    for r in results:
        print(f"{r['drug1']} + {r['drug2']}: {r['level']} - {r['description']}")