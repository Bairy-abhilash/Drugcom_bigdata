"""
FastAPI routes for drug synergy prediction API.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel, Field

from app.db.connection import get_db
from app.db.queries.drug_queries import DrugQueries
from app.db.queries.disease_queries import DiseaseQueries
from app.db.queries.synergy_queries import SynergyQueries
from app.utils.safety_checker import SafetyChecker
from app.utils.logger import setup_logger
from app.graph.builder import HeterogeneousGraphBuilder
from app.gnn_model.inference import SynergyInference
from app.gnn_model.model import DrugSynergyGNN
from app.utils.config import settings
import torch

logger = setup_logger(__name__)

# Create router
router = APIRouter()

# Pydantic models for request/response
class DrugInfo(BaseModel):
    drug_id: int
    drug_name: str
    smiles: Optional[str]
    drugbank_id: Optional[str]
    mechanism_of_action: Optional[str]
    description: Optional[str]

class DiseaseInfo(BaseModel):
    disease_id: int
    disease_name: str
    disease_type: Optional[str]
    tissue_type: Optional[str]
    description: Optional[str]

class SafetyInfo(BaseModel):
    is_safe: bool
    severity: Optional[str]
    interaction_type: Optional[str]
    description: Optional[str]
    mechanism: Optional[str]
    clinical_effect: Optional[str]

class SynergyPrediction(BaseModel):
    drug1_id: int
    drug1_name: str
    drug2_id: int
    drug2_name: str
    synergy_score: float
    confidence: float
    safety_flag: str
    safety_info: SafetyInfo

class PredictionResponse(BaseModel):
    disease_id: int
    disease_name: str
    num_predictions: int
    predictions: List[SynergyPrediction]

class TargetInfo(BaseModel):
    target_id: int
    target_name: str
    gene_name: Optional[str]
    target_type: Optional[str]

# Global model instance (loaded once)
_model_cache = {}

def get_or_load_model():
    """Load GNN model (cached)."""
    if 'model' not in _model_cache:
        try:
            model = DrugSynergyGNN(
                drug_feature_dim=settings.DRUG_FEATURE_DIM,
                target_feature_dim=128,
                disease_feature_dim=settings.DISEASE_FEATURE_DIM,
                hidden_dim=settings.HIDDEN_DIM,
                num_layers=settings.NUM_LAYERS,
                dropout=settings.DROPOUT
            )
            
            # Load trained weights if available
            if Path(settings.MODEL_PATH).exists():
                inference_engine = SynergyInference(
                    model=model,
                    model_path=settings.MODEL_PATH,
                    device='cpu'
                )
                _model_cache['model'] = inference_engine
                logger.info("Loaded trained model")
            else:
                logger.warning("No trained model found, using untrained model")
                _model_cache['model'] = SynergyInference(model=model, device='cpu')
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise HTTPException(status_code=500, detail="Model loading failed")
    
    return _model_cache['model']


# Routes

@router.get("/")
async def root():
    """API root endpoint."""
    return {
        "message": "Drug Synergy Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict?disease_id={id}",
            "drug": "/drug/{id}",
            "disease": "/disease/{id}",
            "diseases": "/diseases",
            "drugs": "/drugs"
        }
    }

@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

@router.get("/drugs", response_model=List[DrugInfo])
async def get_drugs(
    limit: int = Query(50, le=500),
    search: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Get list of drugs.
    
    Args:
        limit: Maximum number of results
        search: Optional search term
        db: Database session
        
    Returns:
        List of drugs
    """
    drug_queries = DrugQueries()
    
    if search:
        drugs = drug_queries.search_drugs(db, search, limit)
    else:
        drugs = drug_queries.get_all_drugs(db, limit)
    
    return [
        DrugInfo(
            drug_id=drug.drug_id,
            drug_name=drug.drug_name,
            smiles=drug.smiles,
            drugbank_id=drug.drugbank_id,
            mechanism_of_action=drug.mechanism_of_action,
            description=drug.description
        )
        for drug in drugs
    ]

@router.get("/drug/{drug_id}", response_model=DrugInfo)
async def get_drug(drug_id: int, db: Session = Depends(get_db)):
    """
    Get drug by ID.
    
    Args:
        drug_id: Drug ID
        db: Database session
        
    Returns:
        Drug information
    """
    drug_queries = DrugQueries()
    drug = drug_queries.get_drug_by_id(db, drug_id)
    
    if not drug:
        raise HTTPException(status_code=404, detail=f"Drug {drug_id} not found")
    
    return DrugInfo(
        drug_id=drug.drug_id,
        drug_name=drug.drug_name,
        smiles=drug.smiles,
        drugbank_id=drug.drugbank_id,
        mechanism_of_action=drug.mechanism_of_action,
        description=drug.description
    )

@router.get("/drug/{drug_id}/targets", response_model=List[TargetInfo])
async def get_drug_targets(drug_id: int, db: Session = Depends(get_db)):
    """Get targets for a drug."""
    drug_queries = DrugQueries()
    targets = drug_queries.get_drug_targets(db, drug_id)
    
    return [
        TargetInfo(
            target_id=t['target_id'],
            target_name=t['target_name'],
            gene_name=t['gene_name'],
            target_type=None
        )
        for t in targets
    ]

@router.get("/diseases", response_model=List[DiseaseInfo])
async def get_diseases(db: Session = Depends(get_db)):
    """Get list of all diseases."""
    disease_queries = DiseaseQueries()
    diseases = disease_queries.get_all_diseases(db)
    
    return [
        DiseaseInfo(
            disease_id=disease.disease_id,
            disease_name=disease.disease_name,
            disease_type=disease.disease_type,
            tissue_type=disease.tissue_type,
            description=disease.description
        )
        for disease in diseases
    ]

@router.get("/disease/{disease_id}", response_model=DiseaseInfo)
async def get_disease(disease_id: int, db: Session = Depends(get_db)):
    """Get disease by ID."""
    disease_queries = DiseaseQueries()
    disease = disease_queries.get_disease_by_id(db, disease_id)
    
    if not disease:
        raise HTTPException(status_code=404, detail=f"Disease {disease_id} not found")
    
    return DiseaseInfo(
        disease_id=disease.disease_id,
        disease_name=disease.disease_name,
        disease_type=disease.disease_type,
        tissue_type=disease.tissue_type,
        description=disease.description
    )

@router.get("/predict", response_model=PredictionResponse)
async def predict_synergy(
    disease_id: int,
    top_k: int = Query(10, le=100),
    db: Session = Depends(get_db)
):
    """
    Predict drug synergy for a disease.
    
    Args:
        disease_id: Disease ID
        top_k: Number of top predictions to return
        db: Database session
        
    Returns:
        Ranked drug combination predictions
    """
    logger.info(f"Predicting synergy for disease {disease_id}")
    
    # Validate disease exists
    disease_queries = DiseaseQueries()
    disease = disease_queries.get_disease_by_id(db, disease_id)
    
    if not disease:
        raise HTTPException(status_code=404, detail=f"Disease {disease_id} not found")
    
    try:
        # Build graph for disease
        graph_builder = HeterogeneousGraphBuilder(db)
        graph, metadata = graph_builder.build_graph_for_disease(disease_id)
        
        # Get node features
        node_features = {
            'drug': graph.nodes['drug'].data['features'],
            'target': graph.nodes['target'].data['features'],
            'disease': graph.nodes['disease'].data['features']
        }
        
        # Get model
        inference_engine = get_or_load_model()
        
        # Get drug indices from metadata
        drug_indices = list(range(metadata['num_drugs']))
        
        # Predict synergy for all pairs
        predictions = inference_engine.rank_drug_pairs(
            graph, node_features, drug_indices, top_k=top_k
        )
        
        # Get drug information and safety checks
        drug_queries = DrugQueries()
        safety_checker = SafetyChecker(db)
        
        formatted_predictions = []
        for pred in predictions:
            # Map indices back to drug IDs
            drug1_id = metadata['drug_ids'][pred['drug1_idx']]
            drug2_id = metadata['drug_ids'][pred['drug2_idx']]
            
            # Get drug names
            drug1 = drug_queries.get_drug_by_id(db, drug1_id)
            drug2 = drug_queries.get_drug_by_id(db, drug2_id)
            
            # Check safety
            safety_info = safety_checker.check_combination_safety(drug1_id, drug2_id)
            
            formatted_predictions.append(
                SynergyPrediction(
                    drug1_id=drug1_id,
                    drug1_name=drug1.drug_name if drug1 else "Unknown",
                    drug2_id=drug2_id,
                    drug2_name=drug2.drug_name if drug2 else "Unknown",
                    synergy_score=pred['synergy_score'],
                    confidence=pred['confidence'],
                    safety_flag="Safe" if safety_info['is_safe'] else "Harmful",
                    safety_info=SafetyInfo(**safety_info)
                )
            )
        
        return PredictionResponse(
            disease_id=disease_id,
            disease_name=disease.disease_name,
            num_predictions=len(formatted_predictions),
            predictions=formatted_predictions
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.get("/safety/{drug1_id}/{drug2_id}", response_model=SafetyInfo)
async def check_safety(
    drug1_id: int,
    drug2_id: int,
    db: Session = Depends(get_db)
):
    """Check safety of drug combination."""
    safety_checker = SafetyChecker(db)
    safety_info = safety_checker.check_combination_safety(drug1_id, drug2_id)
    
    return SafetyInfo(**safety_info)