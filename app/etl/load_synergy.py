"""
ETL script to load drug synergy data.
"""

import pandas as pd
import numpy as np
from sqlalchemy.orm import Session

from app.db.connection import db_manager
from app.db.models import Drug, CellLine, SynergyScore, HarmfulCombination
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class SynergyLoader:
    """Load drug synergy data."""
    
    def __init__(self, db: Session):
        self.db = db
    
    def load_sample_synergy_scores(self) -> int:
        """Load sample synergy scores."""
        # Get drugs and cell lines
        doxorubicin = self.db.query(Drug).filter(Drug.drug_name == 'Doxorubicin').first()
        paclitaxel = self.db.query(Drug).filter(Drug.drug_name == 'Paclitaxel').first()
        cisplatin = self.db.query(Drug).filter(Drug.drug_name == 'Cisplatin').first()
        imatinib = self.db.query(Drug).filter(Drug.drug_name == 'Imatinib').first()
        
        a549 = self.db.query(CellLine).filter(CellLine.cell_line_name == 'A549').first()
        k562 = self.db.query(CellLine).filter(CellLine.cell_line_name == 'K562').first()
        mcf7 = self.db.query(CellLine).filter(CellLine.cell_line_name == 'MCF7').first()
        
        sample_synergies = []
        
        if doxorubicin and paclitaxel and a549:
            sample_synergies.append({
                'drug1_id': doxorubicin.drug_id,
                'drug2_id': paclitaxel.drug_id,
                'cell_line_id': a549.cell_line_id,
                'synergy_score': 25.5,
                'zip_score': 23.2,
                'source': 'DrugComb',
                'study_id': 'study_001'
            })
        
        if doxorubicin and cisplatin and a549:
            sample_synergies.append({
                'drug1_id': doxorubicin.drug_id,
                'drug2_id': cisplatin.drug_id,
                'cell_line_id': a549.cell_line_id,
                'synergy_score': 18.3,
                'zip_score': 17.1,
                'source': 'DrugComb',
                'study_id': 'study_002'
            })
        
        if paclitaxel and cisplatin and mcf7:
            sample_synergies.append({
                'drug1_id': paclitaxel.drug_id,
                'drug2_id': cisplatin.drug_id,
                'cell_line_id': mcf7.cell_line_id,
                'synergy_score': 31.2,
                'zip_score': 29.8,
                'source': 'DrugComb',
                'study_id': 'study_003'
            })
        
        synergy_scores = [SynergyScore(**data) for data in sample_synergies]
        self.db.bulk_save_objects(synergy_scores)
        self.db.commit()
        
        logger.info(f"Loaded {len(synergy_scores)} sample synergy scores")
        return len(synergy_scores)
    
    def load_sample_harmful_combinations(self) -> int:
        """Load sample harmful drug combinations."""
        # This would typically come from drug interaction databases
        sample_harmful = []
        
        # Add some example harmful combinations
        # In practice, this would come from databases like DrugBank DDI
        
        logger.info(f"Loaded {len(sample_harmful)} harmful combinations")
        return len(sample_harmful)


def main():
    """Main ETL function for synergy data."""
    with db_manager.session_scope() as db:
        loader = SynergyLoader(db)
        
        num_synergies = loader.load_sample_synergy_scores()
        num_harmful = loader.load_sample_harmful_combinations()
        
        logger.info(
            f"Synergy ETL completed: {num_synergies} synergy scores, "
            f"{num_harmful} harmful combinations"
        )


if __name__ == '__main__':
    main()