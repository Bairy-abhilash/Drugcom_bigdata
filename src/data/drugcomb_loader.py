"""
DrugComb dataset loader and preprocessor.
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DrugCombLoader:
    """Load and preprocess DrugComb synergy dataset."""
    
    def __init__(self, data_path: str):
        """
        Initialize DrugComb loader.
        
        Args:
            data_path: Path to DrugComb CSV file
        """
        self.data_path = Path(data_path)
        self.data = None
        self.drug_vocab = {}
        self.disease_vocab = {}
        self.target_vocab = {}
        
    def load_data(self) -> pd.DataFrame:
        """
        Load DrugComb data from CSV.
        
        Returns:
            DataFrame with drug combination data
        """
        try:
            self.data = pd.read_csv(self.data_path)
            logger.info(f"Loaded {len(self.data)} drug combinations")
            return self.data
        except FileNotFoundError:
            logger.warning(f"DrugComb file not found at {self.data_path}. Creating mock data.")
            return self._create_mock_data()
    
    def _create_mock_data(self) -> pd.DataFrame:
        """Create mock DrugComb data for demonstration."""
        mock_data = {
            'drug1_name': ['Doxorubicin', 'Paclitaxel', 'Cisplatin', 'Tamoxifen', 'Trastuzumab'] * 20,
            'drug2_name': ['Cyclophosphamide', 'Carboplatin', 'Pemetrexed', 'Letrozole', 'Pertuzumab'] * 20,
            'drug1_smiles': [
                'CC1=C2C(C(=O)C3(C(CC4C(C3C(C(C2(C)C)(CC1OC(=O)C)O)O)(CO4)OC(=O)C)O)C)OC(=O)C',
                'CC1=C2C(C(=O)C3(C(CC4C(C3C(C(C2(C)C)(CC1OC(=O)C(C(C5=CC=CC=C5)NC(=O)C6=CC=CC=C6)O)O)OC(=O)C7=CC=CC=C7)(CO4)OC(=O)C)O)C)OC(=O)C',
                'N.Cl[Pt](Cl)(N)N',
                'CCC(=C(CC)C1=CC=C(C=C1)OCCN(C)C)C2=CC=CC=C2',
                'ANTIBODY_1'
            ] * 20,
            'drug2_smiles': [
                'C1CNP(=O)(OC1)N(CCCl)CCCl',
                'C(C(=O)O)N1C(=O)N(C(=O)N([Pt]1(N)(N)))',
                'CCC1=C(C(=O)NC2=NC(=CC(=N2)C3=CC=C(C=C3)N(C)C=O)C)NC=N1',
                'CC12CCC3C(C1CCC2O)CCC4=C3C=CC(=C4)O',
                'ANTIBODY_2'
            ] * 20,
            'synergy_score': np.random.uniform(0.5, 0.95, 100),
            'disease': ['Breast Cancer'] * 20 + ['Lung Cancer'] * 20 + ['Leukemia'] * 20 + 
                       ['Colorectal Cancer'] * 20 + ['Melanoma'] * 20,
            'cell_line': ['MCF7'] * 100,
            'tissue': ['breast'] * 100
        }
        
        self.data = pd.DataFrame(mock_data)
        logger.info(f"Created mock data with {len(self.data)} combinations")
        return self.data
    
    def preprocess(self) -> Dict:
        """
        Preprocess data for model training.
        
        Returns:
            Dictionary containing processed data
        """
        if self.data is None:
            self.load_data()
        
        # Build vocabularies
        self._build_vocabularies()
        
        # Convert to indices
        drug1_indices = [self.drug_vocab[name] for name in self.data['drug1_name']]
        drug2_indices = [self.drug_vocab[name] for name in self.data['drug2_name']]
        disease_indices = [self.disease_vocab[name] for name in self.data['disease']]
        
        # Normalize synergy scores to [0, 1]
        synergy_scores = self.data['synergy_score'].values
        if synergy_scores.max() > 1.0:
            synergy_scores = synergy_scores / synergy_scores.max()
        
        processed_data = {
            'drug1_ids': np.array(drug1_indices),
            'drug2_ids': np.array(drug2_indices),
            'disease_ids': np.array(disease_indices),
            'synergy_scores': synergy_scores,
            'drug_vocab': self.drug_vocab,
            'disease_vocab': self.disease_vocab,
            'drug_info': self._build_drug_info(),
            'disease_info': self._build_disease_info()
        }
        
        logger.info(f"Preprocessed data: {len(self.drug_vocab)} drugs, {len(self.disease_vocab)} diseases")
        return processed_data
    
    def _build_vocabularies(self):
        """Build drug and disease vocabularies."""
        # Drug vocabulary
        all_drugs = set(self.data['drug1_name'].unique()) | set(self.data['drug2_name'].unique())
        self.drug_vocab = {drug: idx for idx, drug in enumerate(sorted(all_drugs))}
        
        # Disease vocabulary
        all_diseases = self.data['disease'].unique()
        self.disease_vocab = {disease: idx for idx, disease in enumerate(sorted(all_diseases))}
        
        logger.info(f"Built vocabularies: {len(self.drug_vocab)} drugs, {len(self.disease_vocab)} diseases")
    
    def _build_drug_info(self) -> Dict:
        """Build drug information dictionary."""
        drug_info = {}
        
        for drug_name in self.drug_vocab.keys():
            drug_id = self.drug_vocab[drug_name]
            
            # Get SMILES from data
            smiles1_match = self.data[self.data['drug1_name'] == drug_name]
            smiles2_match = self.data[self.data['drug2_name'] == drug_name]
            
            if not smiles1_match.empty:
                smiles = smiles1_match.iloc[0]['drug1_smiles']
            elif not smiles2_match.empty:
                smiles = smiles2_match.iloc[0]['drug2_smiles']
            else:
                smiles = 'N/A'
            
            drug_info[drug_id] = {
                'name': drug_name,
                'smiles': smiles,
                'id': drug_id
            }
        
        return drug_info
    
    def _build_disease_info(self) -> Dict:
        """Build disease information dictionary."""
        disease_info = {}
        
        for disease_name in self.disease_vocab.keys():
            disease_id = self.disease_vocab[disease_name]
            disease_info[disease_id] = {
                'name': disease_name,
                'id': disease_id
            }
        
        return disease_info
    
    def split_data(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        random_seed: int = 42
    ) -> Tuple[Dict, Dict, Dict]:
        """
        Split data into train/val/test sets.
        
        Args:
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            random_seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_data, val_data, test_data) dictionaries
        """
        processed = self.preprocess()
        
        np.random.seed(random_seed)
        n_samples = len(processed['drug1_ids'])
        indices = np.random.permutation(n_samples)
        
        train_end = int(train_ratio * n_samples)
        val_end = train_end + int(val_ratio * n_samples)
        
        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]
        
        def create_split(idx):
            return {
                'drug1_ids': processed['drug1_ids'][idx],
                'drug2_ids': processed['drug2_ids'][idx],
                'disease_ids': processed['disease_ids'][idx],
                'synergy_scores': processed['synergy_scores'][idx]
            }
        
        train_data = create_split(train_idx)
        val_data = create_split(val_idx)
        test_data = create_split(test_idx)
        
        logger.info(f"Split data: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
        
        return train_data, val_data, test_data
    
    def get_drug_disease_associations(self) -> List[Tuple[int, int]]:
        """
        Get drug-disease associations.
        
        Returns:
            List of (drug_id, disease_id) tuples
        """
        associations = set()
        
        for _, row in self.data.iterrows():
            drug1_id = self.drug_vocab[row['drug1_name']]
            drug2_id = self.drug_vocab[row['drug2_name']]
            disease_id = self.disease_vocab[row['disease']]
            
            associations.add((drug1_id, disease_id))
            associations.add((drug2_id, disease_id))
        
        return list(associations)


# Example usage
if __name__ == "__main__":
    loader = DrugCombLoader("data/raw/drugcomb.csv")
    data = loader.load_data()
    print(f"Loaded {len(data)} combinations")
    
    processed = loader.preprocess()
    print(f"Drugs: {len(processed['drug_vocab'])}")
    print(f"Diseases: {len(processed['disease_vocab'])}")