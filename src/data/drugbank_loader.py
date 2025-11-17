"""
DrugBank interaction database loader.
"""
import pandas as pd
from typing import Dict, List
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DrugBankLoader:
    """Load DrugBank drug interaction data."""
    
    def __init__(self, data_path: str = None):
        """
        Initialize DrugBank loader.
        
        Args:
            data_path: Path to DrugBank interactions CSV
        """
        self.data_path = Path(data_path) if data_path else None
        self.interactions = None
    
    def load_interactions(self) -> pd.DataFrame:
        """
        Load drug interactions from CSV.
        
        Returns:
            DataFrame with drug interactions
        """
        if self.data_path and self.data_path.exists():
            try:
                self.interactions = pd.read_csv(self.data_path)
                logger.info(f"Loaded {len(self.interactions)} drug interactions")
                return self.interactions
            except Exception as e:
                logger.error(f"Error loading DrugBank data: {e}")
        
        logger.warning("Creating mock interaction data")
        return self._create_mock_interactions()
    
    def _create_mock_interactions(self) -> pd.DataFrame:
        """Create mock drug interaction data."""
        mock_data = {
            'drug1_name': [
                'Warfarin', 'Methotrexate', 'Digoxin', 'Simvastatin', 'ACE_inhibitors',
                'Paclitaxel', 'Trastuzumab', 'Bevacizumab', 'Cisplatin', 'Doxorubicin',
                'Cyclophosphamide', 'Tamoxifen', 'Imatinib', 'Rituximab', 'Carboplatin'
            ],
            'drug2_name': [
                'Aspirin', 'NSAIDs', 'Amiodarone', 'Amlodipine', 'Potassium',
                'Carboplatin', 'Doxorubicin', 'Sunitinib', 'Pemetrexed', 'Cyclophosphamide',
                'Doxorubicin', 'Letrozole', 'Dasatinib', 'Bendamustine', 'Paclitaxel'
            ],
            'severity': [
                'major', 'major', 'moderate', 'moderate', 'moderate',
                'minor', 'moderate', 'major', 'moderate', 'minor',
                'minor', 'minor', 'moderate', 'moderate', 'minor'
            ],
            'description': [
                'Increased bleeding risk',
                'Increased methotrexate toxicity',
                'Increased digoxin levels',
                'Increased statin levels',
                'Hyperkalemia risk',
                'Monitor for neutropenia',
                'Cardiac toxicity risk',
                'Severe hypertension risk',
                'Increased nephrotoxicity',
                'Standard combination therapy',
                'Standard combination therapy',
                'No significant interaction',
                'Monitor QTc interval',
                'Monitor for myelosuppression',
                'Standard platinum-taxane regimen'
            ]
        }
        
        self.interactions = pd.DataFrame(mock_data)
        logger.info(f"Created {len(self.interactions)} mock interactions")
        return self.interactions
    
    def get_interaction(self, drug1: str, drug2: str) -> Dict:
        """
        Get interaction between two drugs.
        
        Args:
            drug1: First drug name
            drug2: Second drug name
            
        Returns:
            Dictionary with interaction info
        """
        if self.interactions is None:
            self.load_interactions()
        
        drug1 = drug1.lower()
        drug2 = drug2.lower()
        
        # Check both orderings
        mask1 = (self.interactions['drug1_name'].str.lower() == drug1) & \
                (self.interactions['drug2_name'].str.lower() == drug2)
        mask2 = (self.interactions['drug1_name'].str.lower() == drug2) & \
                (self.interactions['drug2_name'].str.lower() == drug1)
        
        result = self.interactions[mask1 | mask2]
        
        if not result.empty:
            row = result.iloc[0]
            return {
                'drug1': row['drug1_name'],
                'drug2': row['drug2_name'],
                'severity': row['severity'],
                'description': row['description'],
                'has_interaction': True
            }
        
        return {
            'drug1': drug1,
            'drug2': drug2,
            'severity': 'safe',
            'description': 'No known significant interactions',
            'has_interaction': False
        }
    
    def get_all_interactions_for_drug(self, drug_name: str) -> List[Dict]:
        """
        Get all interactions for a specific drug.
        
        Args:
            drug_name: Drug name
            
        Returns:
            List of interaction dictionaries
        """
        if self.interactions is None:
            self.load_interactions()
        
        drug_name = drug_name.lower()
        
        mask1 = self.interactions['drug1_name'].str.lower() == drug_name
        mask2 = self.interactions['drug2_name'].str.lower() == drug_name
        
        results = self.interactions[mask1 | mask2]
        
        interactions = []
        for _, row in results.iterrows():
            interactions.append({
                'partner_drug': row['drug2_name'] if row['drug1_name'].lower() == drug_name else row['drug1_name'],
                'severity': row['severity'],
                'description': row['description']
            })
        
        return interactions


# Example usage
if __name__ == "__main__":
    loader = DrugBankLoader()
    interactions = loader.load_interactions()
    print(f"Loaded {len(interactions)} interactions")
    
    result = loader.get_interaction('Warfarin', 'Aspirin')
    print(f"Interaction: {result}")