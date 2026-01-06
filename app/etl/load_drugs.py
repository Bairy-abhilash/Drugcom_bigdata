"""
ETL script to load drug data into the database.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any
from sqlalchemy.orm import Session

from app.db.connection import db_manager
from app.db.models import Drug, DrugTarget, Target, SideEffect
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class DrugLoader:
    """Load drug data from various sources."""
    
    def __init__(self, db: Session):
        self.db = db
    
    def load_from_drugbank_csv(self, csv_path: str) -> int:
        """
        Load drugs from DrugBank CSV export.
        
        Expected columns: drug_name, smiles, inchi, drugbank_id, molecular_formula,
                         molecular_weight, mechanism_of_action, description
        
        Args:
            csv_path: Path to CSV file
            
        Returns:
            Number of drugs loaded
        """
        logger.info(f"Loading drugs from {csv_path}")
        
        df = pd.read_csv(csv_path)
        
        drugs_to_insert = []
        for _, row in df.iterrows():
            drug_data = {
                'drug_name': row.get('drug_name', ''),
                'smiles': row.get('smiles', None),
                'inchi': row.get('inchi', None),
                'molecular_formula': row.get('molecular_formula', None),
                'molecular_weight': float(row['molecular_weight']) if pd.notna(row.get('molecular_weight')) else None,
                'drugbank_id': row.get('drugbank_id', None),
                'pubchem_cid': row.get('pubchem_cid', None),
                'mechanism_of_action': row.get('mechanism_of_action', None),
                'description': row.get('description', None)
            }
            drugs_to_insert.append(drug_data)
        
        # Bulk insert
        drugs = [Drug(**data) for data in drugs_to_insert]
        self.db.bulk_save_objects(drugs)
        self.db.commit()
        
        logger.info(f"Loaded {len(drugs)} drugs")
        return len(drugs)
    
    def load_sample_drugs(self) -> int:
        """Load sample drugs for testing."""
        sample_drugs = [
            {
                'drug_name': 'Doxorubicin',
                'smiles': 'CC1C(C(CC(O1)OC2CC(CC3=C(C4=C(C(=C23)O)C(=O)C5=C(C4=O)C=CC=C5OC)O)(C(=O)CO)O)N)O',
                'drugbank_id': 'DB00997',
                'molecular_weight': 543.52,
                'mechanism_of_action': 'DNA intercalation and topoisomerase II inhibition',
                'description': 'Anthracycline antibiotic used in cancer chemotherapy'
            },
            {
                'drug_name': 'Paclitaxel',
                'smiles': 'CC1=C2C(C(=O)C3(C(CC4C(C3C(C(C2(C)C)(CC1OC(=O)C(C(C5=CC=CC=C5)NC(=O)C6=CC=CC=C6)O)O)OC(=O)C7=CC=CC=C7)(CO4)OC(=O)C)O)C)OC(=O)C',
                'drugbank_id': 'DB01229',
                'molecular_weight': 853.91,
                'mechanism_of_action': 'Microtubule stabilization',
                'description': 'Mitotic inhibitor used in cancer chemotherapy'
            },
            {
                'drug_name': 'Cisplatin',
                'smiles': 'N.N.Cl[Pt]Cl',
                'drugbank_id': 'DB00515',
                'molecular_weight': 300.05,
                'mechanism_of_action': 'DNA crosslinking',
                'description': 'Platinum-based chemotherapy drug'
            },
            {
                'drug_name': 'Imatinib',
                'smiles': 'CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5',
                'drugbank_id': 'DB00619',
                'molecular_weight': 493.60,
                'mechanism_of_action': 'Tyrosine kinase inhibitor',
                'description': 'Targeted therapy for chronic myeloid leukemia'
            },
            {
                'drug_name': 'Erlotinib',
                'smiles': 'C1=CC(=C(C=C1NC2=NC=CC(=N2)NC3=CC=CC(=C3)C#C)OCCOC)OCCOC',
                'drugbank_id': 'DB00530',
                'molecular_weight': 393.44,
                'mechanism_of_action': 'EGFR tyrosine kinase inhibitor',
                'description': 'Used for treatment of non-small cell lung cancer'
            },
            {
                'drug_name': 'Gefitinib',
                'smiles': 'COC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4',
                'drugbank_id': 'DB00317',
                'molecular_weight': 446.90,
                'mechanism_of_action': 'EGFR tyrosine kinase inhibitor',
                'description': 'Treatment for non-small cell lung cancer'
            },
            {
                'drug_name': 'Tamoxifen',
                'smiles': 'CCC(=C(C1=CC=CC=C1)C2=CC=C(C=C2)OCCN(C)C)C3=CC=CC=C3',
                'drugbank_id': 'DB00675',
                'molecular_weight': 371.51,
                'mechanism_of_action': 'Selective estrogen receptor modulator',
                'description': 'Breast cancer treatment and prevention'
            },
            {
                'drug_name': 'Methotrexate',
                'smiles': 'CN(CC1=CN=C2C(=N1)C(=NC(=N2)N)N)C3=CC=C(C=C3)C(=O)NC(CCC(=O)O)C(=O)O',
                'drugbank_id': 'DB00563',
                'molecular_weight': 454.44,
                'mechanism_of_action': 'Dihydrofolate reductase inhibitor',
                'description': 'Antimetabolite and antifolate drug'
            },
            {
                'drug_name': '5-Fluorouracil',
                'smiles': 'C1=C(C(=O)NC(=O)N1)F',
                'drugbank_id': 'DB00544',
                'molecular_weight': 130.08,
                'mechanism_of_action': 'Thymidylate synthase inhibitor',
                'description': 'Pyrimidine analog used in cancer treatment'
            },
            {
                'drug_name': 'Gemcitabine',
                'smiles': 'C1=CN(C(=O)N=C1N)C2C(C(C(O2)CO)O)(F)F',
                'drugbank_id': 'DB00441',
                'molecular_weight': 263.20,
                'mechanism_of_action': 'Nucleoside analog that inhibits DNA synthesis',
                'description': 'Chemotherapy medication for various cancers'
            }
        ]
        
        drugs = [Drug(**data) for data in sample_drugs]
        self.db.bulk_save_objects(drugs)
        self.db.commit()
        
        logger.info(f"Loaded {len(sample_drugs)} sample drugs")
        return len(sample_drugs)
    
    def load_drug_targets(self, csv_path: str) -> int:
        """
        Load drug-target relationships.
        
        Expected columns: drug_name, target_name, interaction_type, binding_affinity
        
        Args:
            csv_path: Path to CSV file
            
        Returns:
            Number of relationships loaded
        """
        logger.info(f"Loading drug-target relationships from {csv_path}")
        
        df = pd.read_csv(csv_path)
        
        relationships = []
        for _, row in df.iterrows():
            # Find drug and target
            drug = self.db.query(Drug).filter(
                Drug.drug_name == row['drug_name']
            ).first()
            
            target = self.db.query(Target).filter(
                Target.target_name == row['target_name']
            ).first()
            
            if drug and target:
                relationships.append({
                    'drug_id': drug.drug_id,
                    'target_id': target.target_id,
                    'interaction_type': row.get('interaction_type', 'unknown'),
                    'binding_affinity': float(row['binding_affinity']) if pd.notna(row.get('binding_affinity')) else None,
                    'source': row.get('source', 'manual')
                })
        
        # Bulk insert
        drug_targets = [DrugTarget(**data) for data in relationships]
        self.db.bulk_save_objects(drug_targets)
        self.db.commit()
        
        logger.info(f"Loaded {len(drug_targets)} drug-target relationships")
        return len(drug_targets)
    
    def load_sample_drug_targets(self) -> int:
        """Load sample drug-target relationships."""
        # Get existing drugs and targets
        doxorubicin = self.db.query(Drug).filter(Drug.drug_name == 'Doxorubicin').first()
        paclitaxel = self.db.query(Drug).filter(Drug.drug_name == 'Paclitaxel').first()
        cisplatin = self.db.query(Drug).filter(Drug.drug_name == 'Cisplatin').first()
        imatinib = self.db.query(Drug).filter(Drug.drug_name == 'Imatinib').first()
        erlotinib = self.db.query(Drug).filter(Drug.drug_name == 'Erlotinib').first()
        
        top2a = self.db.query(Target).filter(Target.gene_name == 'TOP2A').first()
        tubb = self.db.query(Target).filter(Target.gene_name == 'TUBB').first()
        egfr = self.db.query(Target).filter(Target.gene_name == 'EGFR').first()
        bcr_abl = self.db.query(Target).filter(Target.gene_name == 'BCR-ABL1').first()
        
        relationships = []
        
        if doxorubicin and top2a:
            relationships.append({
                'drug_id': doxorubicin.drug_id,
                'target_id': top2a.target_id,
                'interaction_type': 'inhibitor',
                'binding_affinity': 8.5,
                'source': 'DrugBank'
            })
        
        if paclitaxel and tubb:
            relationships.append({
                'drug_id': paclitaxel.drug_id,
                'target_id': tubb.target_id,
                'interaction_type': 'stabilizer',
                'binding_affinity': 9.2,
                'source': 'DrugBank'
            })
        
        if imatinib and bcr_abl:
            relationships.append({
                'drug_id': imatinib.drug_id,
                'target_id': bcr_abl.target_id,
                'interaction_type': 'inhibitor',
                'binding_affinity': 9.8,
                'source': 'DrugBank'
            })
        
        if erlotinib and egfr:
            relationships.append({
                'drug_id': erlotinib.drug_id,
                'target_id': egfr.target_id,
                'interaction_type': 'inhibitor',
                'binding_affinity': 8.7,
                'source': 'DrugBank'
            })
        
        if relationships:
            drug_targets = [DrugTarget(**data) for data in relationships]
            self.db.bulk_save_objects(drug_targets)
            self.db.commit()
            logger.info(f"Loaded {len(drug_targets)} sample drug-target relationships")
            return len(drug_targets)
        
        return 0


def main():
    """Main ETL function for drugs."""
    with db_manager.session_scope() as db:
        loader = DrugLoader(db)
        
        # Load sample drugs
        num_drugs = loader.load_sample_drugs()
        
        # Load sample drug-targets
        num_relationships = loader.load_sample_drug_targets()
        
        logger.info(f"Drug ETL completed: {num_drugs} drugs, {num_relationships} relationships loaded")


if __name__ == '__main__':
    main()