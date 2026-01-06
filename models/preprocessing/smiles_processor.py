"""
SMILES processing and molecular fingerprint generation using RDKit.
"""
from typing import List, Optional, Tuple
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, Descriptors
from rdkit.Chem import rdMolDescriptors
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SMILESProcessor:
    """Process SMILES strings and generate molecular fingerprints."""
    
    def __init__(self, fingerprint_type: str = "morgan", radius: int = 2, n_bits: int = 2048):
        """
        Initialize SMILES processor.
        
        Args:
            fingerprint_type: Type of fingerprint ('morgan', 'maccs', 'topological')
            radius: Radius for Morgan fingerprints
            n_bits: Number of bits for fingerprint
        """
        self.fingerprint_type = fingerprint_type
        self.radius = radius
        self.n_bits = n_bits
        
    def smiles_to_mol(self, smiles: str) -> Optional[Chem.Mol]:
        """Convert SMILES string to RDKit molecule object."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(f"Failed to parse SMILES: {smiles}")
            return mol
        except Exception as e:
            logger.error(f"Error parsing SMILES {smiles}: {e}")
            return None
    
    def generate_fingerprint(self, smiles: str) -> Optional[np.ndarray]:
        """
        Generate molecular fingerprint from SMILES.
        
        Args:
            smiles: SMILES string
            
        Returns:
            Fingerprint as numpy array or None if failed
        """
        mol = self.smiles_to_mol(smiles)
        if mol is None:
            return None
        
        try:
            if self.fingerprint_type == "morgan":
                fp = AllChem.GetMorganFingerprintAsBitVect(
                    mol, self.radius, nBits=self.n_bits
                )
            elif self.fingerprint_type == "maccs":
                fp = MACCSkeys.GenMACCSKeys(mol)
            elif self.fingerprint_type == "topological":
                fp = Chem.RDKFingerprint(mol, fpSize=self.n_bits)
            else:
                raise ValueError(f"Unknown fingerprint type: {self.fingerprint_type}")
            
            return np.array(fp)
        except Exception as e:
            logger.error(f"Error generating fingerprint for {smiles}: {e}")
            return None
    
    def calculate_descriptors(self, smiles: str) -> Optional[dict]:
        """Calculate molecular descriptors."""
        mol = self.smiles_to_mol(smiles)
        if mol is None:
            return None
        
        try:
            descriptors = {
                'molecular_weight': Descriptors.MolWt(mol),
                'logp': Descriptors.MolLogP(mol),
                'num_h_donors': Descriptors.NumHDonors(mol),
                'num_h_acceptors': Descriptors.NumHAcceptors(mol),
                'tpsa': Descriptors.TPSA(mol),
                'num_rotatable_bonds': Descriptors.NumRotatableBonds(mol),
                'num_aromatic_rings': Descriptors.NumAromaticRings(mol),
                'qed': Descriptors.qed(mol)
            }
            return descriptors
        except Exception as e:
            logger.error(f"Error calculating descriptors for {smiles}: {e}")
            return None
    
    def batch_process(self, smiles_list: List[str]) -> Tuple[np.ndarray, List[int]]:
        """
        Process multiple SMILES strings.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            Tuple of (fingerprints array, valid indices)
        """
        fingerprints = []
        valid_indices = []
        
        for idx, smiles in enumerate(smiles_list):
            fp = self.generate_fingerprint(smiles)
            if fp is not None:
                fingerprints.append(fp)
                valid_indices.append(idx)
        
        if fingerprints:
            return np.array(fingerprints), valid_indices
        else:
            return np.array([]), []
    
    def is_valid_smiles(self, smiles: str) -> bool:
        """Check if SMILES string is valid."""
        mol = self.smiles_to_mol(smiles)
        return mol is not None


# Example usage
if __name__ == "__main__":
    processor = SMILESProcessor(fingerprint_type="morgan", radius=2, n_bits=2048)
    
    # Test SMILES
    test_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
    
    fp = processor.generate_fingerprint(test_smiles)
    print(f"Fingerprint shape: {fp.shape if fp is not None else 'None'}")
    
    descriptors = processor.calculate_descriptors(test_smiles)
    print(f"Descriptors: {descriptors}")