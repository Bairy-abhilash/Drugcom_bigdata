"""
ETL script to load biological target data.
"""

import pandas as pd
from sqlalchemy.orm import Session

from app.db.connection import db_manager
from app.db.models import Target
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class TargetLoader:
    """Load biological target data."""
    
    def __init__(self, db: Session):
        self.db = db
    
    def load_sample_targets(self) -> int:
        """Load sample targets for testing."""
        sample_targets = [
            {
                'target_name': 'DNA topoisomerase 2-alpha',
                'gene_name': 'TOP2A',
                'uniprot_id': 'P11388',
                'target_type': 'enzyme',
                'description': 'Nuclear enzyme involved in DNA replication'
            },
            {
                'target_name': 'Tubulin beta chain',
                'gene_name': 'TUBB',
                'uniprot_id': 'P07437',
                'target_type': 'structural protein',
                'description': 'Component of microtubules'
            },
            {
                'target_name': 'Epidermal growth factor receptor',
                'gene_name': 'EGFR',
                'uniprot_id': 'P00533',
                'target_type': 'receptor tyrosine kinase',
                'description': 'Receptor for EGF family proteins'
            },
            {
                'target_name': 'BCR-ABL fusion protein',
                'gene_name': 'BCR-ABL1',
                'uniprot_id': 'P11274',
                'target_type': 'kinase',
                'description': 'Constitutively active tyrosine kinase'
            },
            {
                'target_name': 'Vascular endothelial growth factor receptor 2',
                'gene_name': 'KDR',
                'uniprot_id': 'P35968',
                'target_type': 'receptor tyrosine kinase',
                'description': 'VEGF receptor involved in angiogenesis'
            }
        ]
        
        targets = [Target(**data) for data in sample_targets]
        self.db.bulk_save_objects(targets)
        self.db.commit()
        
        logger.info(f"Loaded {len(sample_targets)} sample targets")
        return len(sample_targets)


def main():
    """Main ETL function for targets."""
    with db_manager.session_scope() as db:
        loader = TargetLoader(db)
        num_targets = loader.load_sample_targets()
        logger.info(f"Target ETL completed: {num_targets} targets loaded")


if __name__ == '__main__':
    main()