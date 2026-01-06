"""
Main ETL script to load all data into the database.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.db.connection import init_database, db_manager
from app.etl.load_drugs import DrugLoader
from app.etl.load_targets import TargetLoader
from app.etl.load_diseases import DiseaseLoader
from app.etl.load_synergy import SynergyLoader
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


def run_full_etl():
    """Run complete ETL pipeline."""
    logger.info("=" * 50)
    logger.info("Starting ETL Pipeline")
    logger.info("=" * 50)
    
    try:
        # Initialize database schema
        logger.info("Initializing database schema...")
        init_database()
        
        with db_manager.session_scope() as db:
            # Load targets first (referenced by drugs)
            logger.info("\n1. Loading biological targets...")
            target_loader = TargetLoader(db)
            num_targets = target_loader.load_sample_targets()
            
            # Load drugs
            logger.info("\n2. Loading drugs...")
            drug_loader = DrugLoader(db)
            num_drugs = drug_loader.load_sample_drugs()
            
            # Load diseases and cell lines
            logger.info("\n3. Loading diseases and cell lines...")
            disease_loader = DiseaseLoader(db)
            num_diseases = disease_loader.load_sample_diseases()
            num_cell_lines = disease_loader.load_sample_cell_lines()
            num_associations = disease_loader.load_target_disease_associations()
            
            # Load synergy data
            logger.info("\n4. Loading synergy scores...")
            synergy_loader = SynergyLoader(db)
            num_synergies = synergy_loader.load_sample_synergy_scores()
            num_harmful = synergy_loader.load_sample_harmful_combinations()
        
        logger.info("\n" + "=" * 50)
        logger.info("ETL Pipeline Completed Successfully!")
        logger.info("=" * 50)
        logger.info(f"Summary:")
        logger.info(f"  - Targets: {num_targets}")
        logger.info(f"  - Drugs: {num_drugs}")
        logger.info(f"  - Diseases: {num_diseases}")
        logger.info(f"  - Cell Lines: {num_cell_lines}")
        logger.info(f"  - Target-Disease Associations: {num_associations}")
        logger.info(f"  - Synergy Scores: {num_synergies}")
        logger.info(f"  - Harmful Combinations: {num_harmful}")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"ETL Pipeline failed: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    run_full_etl()