-- Drug Synergy Database Schema
-- PostgreSQL 13+

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Drugs table
CREATE TABLE IF NOT EXISTS drugs (
    drug_id SERIAL PRIMARY KEY,
    drug_name VARCHAR(255) NOT NULL,
    smiles TEXT,
    inchi TEXT,
    molecular_formula VARCHAR(100),
    molecular_weight FLOAT,
    drugbank_id VARCHAR(50) UNIQUE,
    pubchem_cid VARCHAR(50),
    mechanism_of_action TEXT,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index on drug_name for faster searches
CREATE INDEX idx_drugs_name ON drugs(drug_name);
CREATE INDEX idx_drugs_drugbank_id ON drugs(drugbank_id);

-- Targets table
CREATE TABLE IF NOT EXISTS targets (
    target_id SERIAL PRIMARY KEY,
    target_name VARCHAR(255) NOT NULL,
    gene_name VARCHAR(100),
    uniprot_id VARCHAR(50) UNIQUE,
    target_type VARCHAR(100),
    organism VARCHAR(100) DEFAULT 'Homo sapiens',
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_targets_gene_name ON targets(gene_name);
CREATE INDEX idx_targets_uniprot_id ON targets(uniprot_id);

-- Diseases table
CREATE TABLE IF NOT EXISTS diseases (
    disease_id SERIAL PRIMARY KEY,
    disease_name VARCHAR(255) NOT NULL,
    disease_type VARCHAR(100),
    tissue_type VARCHAR(100),
    icd_code VARCHAR(50),
    mesh_id VARCHAR(50),
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_diseases_name ON diseases(disease_name);
CREATE INDEX idx_diseases_type ON diseases(disease_type);

-- Drug-Target relationships
CREATE TABLE IF NOT EXISTS drug_targets (
    id SERIAL PRIMARY KEY,
    drug_id INTEGER REFERENCES drugs(drug_id) ON DELETE CASCADE,
    target_id INTEGER REFERENCES targets(target_id) ON DELETE CASCADE,
    interaction_type VARCHAR(100),
    binding_affinity FLOAT,
    source VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(drug_id, target_id)
);

CREATE INDEX idx_drug_targets_drug ON drug_targets(drug_id);
CREATE INDEX idx_drug_targets_target ON drug_targets(target_id);

-- Target-Disease relationships
CREATE TABLE IF NOT EXISTS target_diseases (
    id SERIAL PRIMARY KEY,
    target_id INTEGER REFERENCES targets(target_id) ON DELETE CASCADE,
    disease_id INTEGER REFERENCES diseases(disease_id) ON DELETE CASCADE,
    association_score FLOAT,
    evidence_type VARCHAR(100),
    source VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(target_id, disease_id)
);

CREATE INDEX idx_target_diseases_target ON target_diseases(target_id);
CREATE INDEX idx_target_diseases_disease ON target_diseases(disease_id);

-- Cell lines table
CREATE TABLE IF NOT EXISTS cell_lines (
    cell_line_id SERIAL PRIMARY KEY,
    cell_line_name VARCHAR(255) NOT NULL UNIQUE,
    disease_id INTEGER REFERENCES diseases(disease_id),
    tissue_origin VARCHAR(100),
    organism VARCHAR(100) DEFAULT 'Homo sapiens',
    ccle_name VARCHAR(255),
    cosmic_id VARCHAR(50),
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_cell_lines_disease ON cell_lines(disease_id);
CREATE INDEX idx_cell_lines_name ON cell_lines(cell_line_name);

-- Synergy scores table
CREATE TABLE IF NOT EXISTS synergy_scores (
    synergy_id SERIAL PRIMARY KEY,
    drug1_id INTEGER REFERENCES drugs(drug_id) ON DELETE CASCADE,
    drug2_id INTEGER REFERENCES drugs(drug_id) ON DELETE CASCADE,
    cell_line_id INTEGER REFERENCES cell_lines(cell_line_id) ON DELETE CASCADE,
    synergy_score FLOAT NOT NULL,
    loewe_score FLOAT,
    bliss_score FLOAT,
    zip_score FLOAT,
    hsa_score FLOAT,
    source VARCHAR(100),
    study_id VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CHECK (drug1_id != drug2_id)
);

CREATE INDEX idx_synergy_drug1 ON synergy_scores(drug1_id);
CREATE INDEX idx_synergy_drug2 ON synergy_scores(drug2_id);
CREATE INDEX idx_synergy_cell_line ON synergy_scores(cell_line_id);
CREATE INDEX idx_synergy_score ON synergy_scores(synergy_score);

-- Side effects table
CREATE TABLE IF NOT EXISTS side_effects (
    side_effect_id SERIAL PRIMARY KEY,
    drug_id INTEGER REFERENCES drugs(drug_id) ON DELETE CASCADE,
    effect_name VARCHAR(255) NOT NULL,
    severity VARCHAR(50),
    frequency VARCHAR(50),
    umls_id VARCHAR(50),
    meddra_id VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_side_effects_drug ON side_effects(drug_id);
CREATE INDEX idx_side_effects_name ON side_effects(effect_name);

-- Harmful drug combinations table
CREATE TABLE IF NOT EXISTS harmful_combinations (
    combination_id SERIAL PRIMARY KEY,
    drug1_id INTEGER REFERENCES drugs(drug_id) ON DELETE CASCADE,
    drug2_id INTEGER REFERENCES drugs(drug_id) ON DELETE CASCADE,
    interaction_type VARCHAR(100),
    severity_level VARCHAR(50), -- 'mild', 'moderate', 'severe', 'contraindicated'
    description TEXT,
    mechanism TEXT,
    clinical_effect TEXT,
    source VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CHECK (drug1_id < drug2_id), -- Ensure consistent ordering
    UNIQUE(drug1_id, drug2_id)
);

CREATE INDEX idx_harmful_drug1 ON harmful_combinations(drug1_id);
CREATE INDEX idx_harmful_drug2 ON harmful_combinations(drug2_id);
CREATE INDEX idx_harmful_severity ON harmful_combinations(severity_level);

-- Model predictions cache (optional, for performance)
CREATE TABLE IF NOT EXISTS prediction_cache (
    cache_id SERIAL PRIMARY KEY,
    disease_id INTEGER REFERENCES diseases(disease_id) ON DELETE CASCADE,
    drug1_id INTEGER REFERENCES drugs(drug_id) ON DELETE CASCADE,
    drug2_id INTEGER REFERENCES drugs(drug_id) ON DELETE CASCADE,
    predicted_synergy FLOAT NOT NULL,
    confidence_score FLOAT,
    model_version VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,
    UNIQUE(disease_id, drug1_id, drug2_id, model_version)
);

CREATE INDEX idx_cache_disease ON prediction_cache(disease_id);
CREATE INDEX idx_cache_expiry ON prediction_cache(expires_at);

-- Create views for common queries

-- View: Drug pairs with existing synergy data
CREATE OR REPLACE VIEW drug_synergy_summary AS
SELECT 
    d1.drug_name as drug1_name,
    d2.drug_name as drug2_name,
    cl.cell_line_name,
    dis.disease_name,
    AVG(ss.synergy_score) as avg_synergy,
    COUNT(*) as num_experiments,
    MAX(ss.synergy_score) as max_synergy,
    MIN(ss.synergy_score) as min_synergy
FROM synergy_scores ss
JOIN drugs d1 ON ss.drug1_id = d1.drug_id
JOIN drugs d2 ON ss.drug2_id = d2.drug_id
JOIN cell_lines cl ON ss.cell_line_id = cl.cell_line_id
LEFT JOIN diseases dis ON cl.disease_id = dis.disease_id
GROUP BY d1.drug_name, d2.drug_name, cl.cell_line_name, dis.disease_name;

-- View: Drugs with their targets
CREATE OR REPLACE VIEW drug_target_summary AS
SELECT 
    d.drug_id,
    d.drug_name,
    t.target_id,
    t.target_name,
    t.gene_name,
    dt.interaction_type,
    dt.binding_affinity
FROM drugs d
JOIN drug_targets dt ON d.drug_id = dt.drug_id
JOIN targets t ON dt.target_id = t.target_id;

-- View: Disease-related information
CREATE OR REPLACE VIEW disease_info AS
SELECT 
    dis.disease_id,
    dis.disease_name,
    dis.disease_type,
    COUNT(DISTINCT cl.cell_line_id) as num_cell_lines,
    COUNT(DISTINCT td.target_id) as num_associated_targets
FROM diseases dis
LEFT JOIN cell_lines cl ON dis.disease_id = cl.disease_id
LEFT JOIN target_diseases td ON dis.disease_id = td.disease_id
GROUP BY dis.disease_id, dis.disease_name, dis.disease_type;

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger for drugs table
CREATE TRIGGER update_drugs_updated_at BEFORE UPDATE ON drugs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Grant permissions (adjust as needed)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO drug_synergy_user;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO drug_synergy_user;