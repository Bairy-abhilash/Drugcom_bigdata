"""
Feature generation for graph nodes.
"""

import numpy as np
import torch
from typing import Dict, List, Optional
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from sqlalchemy.orm import Session

from app.db.models import Drug, Disease, Target
from app.utils.logger import setup_logger
from app.utils.config import settings

logger = setup_logger(__name__)


class FeatureGenerator:
    """Generate node features for the heterogeneous graph."""
    
    def __init__(self, drug_feature_dim: int = None, disease_feature_dim: int = None):
        """
        Initialize feature generator.
        
        Args:
            drug_feature_dim: Dimension of drug features
            disease_feature_dim: Dimension of disease features
        """
        self.drug_feature_dim = drug_feature_dim or settings.DRUG_FEATURE_DIM
        self.disease_feature_dim = disease_feature_dim or settings.DISEASE_FEATURE_DIM
    
    def generate_drug_features(self, smiles: str) -> np.ndarray:
        """
        Generate drug features from SMILES using RDKit molecular fingerprints.
        
        Args:
            smiles: SMILES string
            
        Returns:
            Feature vector as numpy array
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(f"Invalid SMILES: {smiles}")
                return np.zeros(self.drug_feature_dim)
            
            # Generate Morgan fingerprint (ECFP)
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol, 
                radius=2, 
                nBits=self.drug_feature_dim
            )
            
            # Convert to numpy array
            arr = np.zeros((self.drug_feature_dim,), dtype=np.float32)
            Chem.DataStructs.ConvertToNumpyArray(fp, arr)
            
            return arr
            
        except Exception as e:
            logger.error(f"Error generating drug features for SMILES {smiles}: {e}")
            return np.zeros(self.drug_feature_dim)
    
    def generate_drug_features_with_descriptors(self, smiles: str) -> np.ndarray:
        """
        Generate drug features combining fingerprints and molecular descriptors.
        
        Args:
            smiles: SMILES string
            
        Returns:
            Feature vector as numpy array
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return np.zeros(self.drug_feature_dim)
            
            # Morgan fingerprint
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            fp_arr = np.zeros((2048,), dtype=np.float32)
            Chem.DataStructs.ConvertToNumpyArray(fp, fp_arr)
            
            # Molecular descriptors
            descriptors = np.array([
                Descriptors.MolWt(mol),
                Descriptors.MolLogP(mol),
                Descriptors.NumHDonors(mol),
                Descriptors.NumHAcceptors(mol),
                Descriptors.TPSA(mol),
                Descriptors.NumRotatableBonds(mol),
                Descriptors.NumAromaticRings(mol),
            ], dtype=np.float32)
            
            # Normalize descriptors
            descriptors = (descriptors - descriptors.mean()) / (descriptors.std() + 1e-8)
            
            # Combine features
            if self.drug_feature_dim == 2048:
                return fp_arr
            else:
                # Pad or truncate to desired dimension
                combined = np.concatenate([fp_arr, descriptors])
                if len(combined) > self.drug_feature_dim:
                    return combined[:self.drug_feature_dim]
                else:
                    return np.pad(combined, (0, self.drug_feature_dim - len(combined)))
            
        except Exception as e:
            logger.error(f"Error generating enhanced drug features: {e}")
            return np.zeros(self.drug_feature_dim)
    
    def generate_disease_features(self, disease: Disease) -> np.ndarray:
        """
        Generate disease features from disease information.
        
        Args:
            disease: Disease model instance
            
        Returns:
            Feature vector as numpy array
        """
        features = np.zeros(self.disease_feature_dim, dtype=np.float32)
        
        # One-hot encode disease type
        disease_types = [
            'carcinoma', 'lymphoma', 'leukemia', 'sarcoma', 
            'melanoma', 'glioma', 'neuroblastoma', 'other'
        ]
        
        if disease.disease_type:
            disease_type_lower = disease.disease_type.lower()
            for idx, dtype in enumerate(disease_types):
                if dtype in disease_type_lower:
                    if idx < self.disease_feature_dim:
                        features[idx] = 1.0
                    break
        
        # One-hot encode tissue type (next positions)
        tissue_types = [
            'lung', 'breast', 'colon', 'liver', 'brain', 
            'blood', 'skin', 'bone', 'other'
        ]
        
        if disease.tissue_type:
            tissue_type_lower = disease.tissue_type.lower()
            for idx, ttype in enumerate(tissue_types):
                if ttype in tissue_type_lower:
                    offset = len(disease_types)
                    if offset + idx < self.disease_feature_dim:
                        features[offset + idx] = 1.0
                    break
        
        return features
    
    def generate_target_features(self, target: Target) -> np.ndarray:
        """
        Generate target features.
        
        Args:
            target: Target model instance
            
        Returns:
            Feature vector as numpy array
        """
        # Simple one-hot encoding for target types
        target_feature_dim = 128
        features = np.zeros(target_feature_dim, dtype=np.float32)
        
        target_types = [
            'enzyme', 'receptor', 'transporter', 'carrier',
            'ion channel', 'kinase', 'gpcr', 'other'
        ]
        
        if target.target_type:
            target_type_lower = target.target_type.lower()
            for idx, ttype in enumerate(target_types):
                if ttype in target_type_lower:
                    if idx < target_feature_dim:
                        features[idx] = 1.0
                    break
        
        return features
    
    def batch_generate_drug_features(
        self, 
        db: Session, 
        drug_ids: List[int]
    ) -> torch.Tensor:
        """
        Generate features for multiple drugs.
        
        Args:
            db: Database session
            drug_ids: List of drug IDs
            
        Returns:
            Tensor of shape (num_drugs, drug_feature_dim)
        """
        features = []
        
        for drug_id in drug_ids:
            drug = db.query(Drug).filter(Drug.drug_id == drug_id).first()
            if drug and drug.smiles:
                feat = self.generate_drug_features(drug.smiles)
            else:
                feat = np.zeros(self.drug_feature_dim)
            features.append(feat)
        
        return torch.FloatTensor(np.array(features))
    
    def batch_generate_disease_features(
        self, 
        db: Session, 
        disease_ids: List[int]
    ) -> torch.Tensor:
        """
        Generate features for multiple diseases.
        
        Args:
            db: Database session
            disease_ids: List of disease IDs
            
        Returns:
            Tensor of shape (num_diseases, disease_feature_dim)
        """
        features = []
        
        for disease_id in disease_ids:
            disease = db.query(Disease).filter(Disease.disease_id == disease_id).first()
            if disease:
                feat = self.generate_disease_features(disease)
            else:
                feat = np.zeros(self.disease_feature_dim)
            features.append(feat)
        
        return torch.FloatTensor(np.array(features))