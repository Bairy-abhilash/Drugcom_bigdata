"""
ETL script to load disease and cell line data.
"""

import pandas as pd
from sqlalchemy.orm import Session

from app.db.connection import db_manager
from app.db.models import Disease, CellLine, TargetDisease, Target
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class DiseaseLoader:
    """Load disease and cell line data."""
    
    def __init__(self, db: Session):
        self.db = db
    
    def load_sample_diseases(self) -> int:
        """Load sample diseases."""
        sample_diseases = [
            {
                'disease_name': 'Lung Adenocarcinoma',
                'disease_type': 'carcinoma',
                'tissue_type': 'lung',
                'description': 'A type of non-small cell lung cancer'
            },
            {
                'disease_name': 'Chronic Myeloid Leukemia',
                'disease_type': 'leukemia',
                'tissue_type': 'blood',
                'description': 'Cancer of white blood cells characterized by BCR-ABL fusion'
            },
            {
                'disease_name': 'Breast Carcinoma',
                'disease_type': 'carcinoma',
                'tissue_type': 'breast',
                'description': 'Malignant tumor of breast tissue'
            },
            {
                'disease_name': 'Colorectal Adenocarcinoma',
                'disease_type': 'carcinoma',
                'tissue_type': 'colon',
                'description': 'Cancer of the colon or rectum'
            },
            {
                'disease_name': 'Glioblastoma',
                'disease_type': 'glioma',
                'tissue_type': 'brain',
                'description': 'Aggressive brain tumor'
            }
        ]
        
        diseases = [Disease(**data) for data in sample_diseases]
        self.db.bulk_save_objects(diseases)
        self.db.commit()
        
        logger.info(f"Loaded {len(sample_diseases)} sample diseases")
        return len(sample_diseases)
    
    def load_sample_cell_lines(self) -> int:
        """Load sample cell lines."""
        # Get diseases
        lung_cancer = self.db.query(Disease).filter(
            Disease.disease_name == 'Lung Adenocarcinoma'
        ).first()
        
        cml = self.db.query(Disease).filter(
            Disease.disease_name == 'Chronic Myeloid Leukemia'
        ).first()
        
        breast_cancer = self.db.query(Disease).filter(
            Disease.disease_name == 'Breast Carcinoma'
        ).first()
        
        sample_cell_lines = [
            {
                'cell_line_name': 'A549',
                'disease_id': lung_cancer.disease_id if lung_cancer else None,
                'tissue_origin': 'lung',
                'ccle_name': 'A549_LUNG',
                'description': 'Lung adenocarcinoma cell line'
            },
            {
                'cell_line_name': 'K562',
                'disease_id': cml.disease_id if cml else None,
                'tissue_origin': 'blood',
                'ccle_name': 'K562_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE',
                'description': 'Chronic myeloid leukemia cell line'
            },
            {
                'cell_line_name': 'MCF7',
                'disease_id': breast_cancer.disease_id if breast_cancer else None,
                'tissue_origin': 'breast',
                'ccle_name': 'MCF7_BREAST',
                'description': 'Breast adenocarcinoma cell line'
            }
        ]
        
        cell_lines = [CellLine(**data) for data in sample_cell_lines]
        self.db.bulk_save_objects(cell_lines)
        self.db.commit()
        
        logger.info(f"Loaded {len(sample_cell_lines)} sample cell lines")
        return len(sample_cell_lines)
    
    def load_target_disease_associations(self) -> int:
        """Create sample target-disease associations."""
        # Get entities
        egfr = self.db.query(Target).filter(Target.gene_name == 'EGFR').first()
        bcr_abl = self.db.query(Target).filter(Target.gene_name == 'BCR-ABL1').first()
        
        lung_cancer = self.db.query(Disease).filter(
            Disease.disease_name == 'Lung Adenocarcinoma'
        ).first()
        cml = self.db.query(Disease).filter(
            Disease.disease_name == 'Chronic Myeloid Leukemia'
        ).first()
        
        associations = []
        
        if egfr and lung_cancer:
            associations.append({
                'target_id': egfr.target_id,
                'disease_id': lung_cancer.disease_id,
                'association_score': 0.9,
                'evidence_type': 'genetic',
                'source': 'literature'
            })
        
        if bcr_abl and cml:
            associations.append({
                'target_id': bcr_abl.target_id,
                'disease_id': cml.disease_id,
                'association_score': 1.0,
                'evidence_type': 'genetic',
                'source': 'literature'
            })
        
        target_diseases = [TargetDisease(**data) for data in associations]
        self.db.bulk_save_objects(target_diseases)
        self.db.commit()
        
        logger.info(f"Loaded {len(target_diseases)} target-disease associations")
        return len(target_diseases)


def main():
    """Main ETL function for diseases."""
    with db_manager.session_scope() as db:
        loader = DiseaseLoader(db)
        
        num_diseases = loader.load_sample_diseases()
        num_cell_lines = loader.load_sample_cell_lines()
        num_associations = loader.load_target_disease_associations()
        
        logger.info(
            f"Disease ETL completed: {num_diseases} diseases, "
            f"{num_cell_lines} cell lines, {num_associations} associations"
        )


if __name__ == '__main__':
    main()