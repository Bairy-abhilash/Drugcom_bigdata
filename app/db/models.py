"""
SQLAlchemy ORM models for database tables.
"""

from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, Text, DateTime, 
    ForeignKey, CheckConstraint, UniqueConstraint
)
from sqlalchemy.orm import relationship
from app.db.connection import Base


class Drug(Base):
    """Drug entity model."""
    
    __tablename__ = 'drugs'
    
    drug_id = Column(Integer, primary_key=True, autoincrement=True)
    drug_name = Column(String(255), nullable=False)
    smiles = Column(Text)
    inchi = Column(Text)
    molecular_formula = Column(String(100))
    molecular_weight = Column(Float)
    drugbank_id = Column(String(50), unique=True)
    pubchem_cid = Column(String(50))
    mechanism_of_action = Column(Text)
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    drug_targets = relationship("DrugTarget", back_populates="drug", cascade="all, delete-orphan")
    side_effects = relationship("SideEffect", back_populates="drug", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Drug(id={self.drug_id}, name='{self.drug_name}')>"


class Target(Base):
    """Biological target model."""
    
    __tablename__ = 'targets'
    
    target_id = Column(Integer, primary_key=True, autoincrement=True)
    target_name = Column(String(255), nullable=False)
    gene_name = Column(String(100))
    uniprot_id = Column(String(50), unique=True)
    target_type = Column(String(100))
    organism = Column(String(100), default='Homo sapiens')
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    drug_targets = relationship("DrugTarget", back_populates="target")
    target_diseases = relationship("TargetDisease", back_populates="target")
    
    def __repr__(self):
        return f"<Target(id={self.target_id}, name='{self.target_name}')>"


class Disease(Base):
    """Disease/cancer type model."""
    
    __tablename__ = 'diseases'
    
    disease_id = Column(Integer, primary_key=True, autoincrement=True)
    disease_name = Column(String(255), nullable=False)
    disease_type = Column(String(100))
    tissue_type = Column(String(100))
    icd_code = Column(String(50))
    mesh_id = Column(String(50))
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    cell_lines = relationship("CellLine", back_populates="disease")
    target_diseases = relationship("TargetDisease", back_populates="disease")
    
    def __repr__(self):
        return f"<Disease(id={self.disease_id}, name='{self.disease_name}')>"


class DrugTarget(Base):
    """Drug-Target interaction model."""
    
    __tablename__ = 'drug_targets'
    __table_args__ = (
        UniqueConstraint('drug_id', 'target_id', name='uq_drug_target'),
    )
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    drug_id = Column(Integer, ForeignKey('drugs.drug_id', ondelete='CASCADE'))
    target_id = Column(Integer, ForeignKey('targets.target_id', ondelete='CASCADE'))
    interaction_type = Column(String(100))
    binding_affinity = Column(Float)
    source = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    drug = relationship("Drug", back_populates="drug_targets")
    target = relationship("Target", back_populates="drug_targets")
    
    def __repr__(self):
        return f"<DrugTarget(drug_id={self.drug_id}, target_id={self.target_id})>"


class TargetDisease(Base):
    """Target-Disease association model."""
    
    __tablename__ = 'target_diseases'
    __table_args__ = (
        UniqueConstraint('target_id', 'disease_id', name='uq_target_disease'),
    )
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    target_id = Column(Integer, ForeignKey('targets.target_id', ondelete='CASCADE'))
    disease_id = Column(Integer, ForeignKey('diseases.disease_id', ondelete='CASCADE'))
    association_score = Column(Float)
    evidence_type = Column(String(100))
    source = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    target = relationship("Target", back_populates="target_diseases")
    disease = relationship("Disease", back_populates="target_diseases")
    
    def __repr__(self):
        return f"<TargetDisease(target_id={self.target_id}, disease_id={self.disease_id})>"


class CellLine(Base):
    """Cell line model."""
    
    __tablename__ = 'cell_lines'
    
    cell_line_id = Column(Integer, primary_key=True, autoincrement=True)
    cell_line_name = Column(String(255), nullable=False, unique=True)
    disease_id = Column(Integer, ForeignKey('diseases.disease_id'))
    tissue_origin = Column(String(100))
    organism = Column(String(100), default='Homo sapiens')
    ccle_name = Column(String(255))
    cosmic_id = Column(String(50))
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    disease = relationship("Disease", back_populates="cell_lines")
    synergy_scores = relationship("SynergyScore", back_populates="cell_line")
    
    def __repr__(self):
        return f"<CellLine(id={self.cell_line_id}, name='{self.cell_line_name}')>"


class SynergyScore(Base):
    """Drug combination synergy score model."""
    
    __tablename__ = 'synergy_scores'
    __table_args__ = (
        CheckConstraint('drug1_id != drug2_id', name='check_different_drugs'),
    )
    
    synergy_id = Column(Integer, primary_key=True, autoincrement=True)
    drug1_id = Column(Integer, ForeignKey('drugs.drug_id', ondelete='CASCADE'))
    drug2_id = Column(Integer, ForeignKey('drugs.drug_id', ondelete='CASCADE'))
    cell_line_id = Column(Integer, ForeignKey('cell_lines.cell_line_id', ondelete='CASCADE'))
    synergy_score = Column(Float, nullable=False)
    loewe_score = Column(Float)
    bliss_score = Column(Float)
    zip_score = Column(Float)
    hsa_score = Column(Float)
    source = Column(String(100))
    study_id = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    cell_line = relationship("CellLine", back_populates="synergy_scores")
    
    def __repr__(self):
        return f"<SynergyScore(drugs={self.drug1_id},{self.drug2_id}, score={self.synergy_score})>"


class SideEffect(Base):
    """Drug side effect model."""
    
    __tablename__ = 'side_effects'
    
    side_effect_id = Column(Integer, primary_key=True, autoincrement=True)
    drug_id = Column(Integer, ForeignKey('drugs.drug_id', ondelete='CASCADE'))
    effect_name = Column(String(255), nullable=False)
    severity = Column(String(50))
    frequency = Column(String(50))
    umls_id = Column(String(50))
    meddra_id = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    drug = relationship("Drug", back_populates="side_effects")
    
    def __repr__(self):
        return f"<SideEffect(drug_id={self.drug_id}, effect='{self.effect_name}')>"


class HarmfulCombination(Base):
    """Harmful drug combination model."""
    
    __tablename__ = 'harmful_combinations'
    __table_args__ = (
        CheckConstraint('drug1_id < drug2_id', name='check_drug_order'),
        UniqueConstraint('drug1_id', 'drug2_id', name='uq_harmful_combination'),
    )
    
    combination_id = Column(Integer, primary_key=True, autoincrement=True)
    drug1_id = Column(Integer, ForeignKey('drugs.drug_id', ondelete='CASCADE'))
    drug2_id = Column(Integer, ForeignKey('drugs.drug_id', ondelete='CASCADE'))
    interaction_type = Column(String(100))
    severity_level = Column(String(50))
    description = Column(Text)
    mechanism = Column(Text)
    clinical_effect = Column(Text)
    source = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<HarmfulCombination(drugs={self.drug1_id},{self.drug2_id}, severity='{self.severity_level}')>"


class PredictionCache(Base):
    """Model prediction cache for performance optimization."""
    
    __tablename__ = 'prediction_cache'
    __table_args__ = (
        UniqueConstraint('disease_id', 'drug1_id', 'drug2_id', 'model_version', 
                        name='uq_prediction_cache'),
    )
    
    cache_id = Column(Integer, primary_key=True, autoincrement=True)
    disease_id = Column(Integer, ForeignKey('diseases.disease_id', ondelete='CASCADE'))
    drug1_id = Column(Integer, ForeignKey('drugs.drug_id', ondelete='CASCADE'))
    drug2_id = Column(Integer, ForeignKey('drugs.drug_id', ondelete='CASCADE'))
    predicted_synergy = Column(Float, nullable=False)
    confidence_score = Column(Float)
    model_version = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime)
    
    def __repr__(self):
        return f"<PredictionCache(disease={self.disease_id}, drugs={self.drug1_id},{self.drug2_id})>"