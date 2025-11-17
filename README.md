# 🧬 AI-Powered Drug Combination Therapy System

A complete Graph Neural Network-based system for predicting drug synergy with confidence estimation and safety checking.

## 🌟 Features

- **Graph Neural Networks**: HeteroGraphSAGE for multi-relational drug-target-disease graphs
- **Synergy Prediction**: Predict effectiveness of drug combinations
- **Confidence Estimation**: Monte Carlo Dropout for uncertainty quantification
- **Safety Checking**: Integrated DrugBank interaction database
- **Interactive Dashboard**: Streamlit-based UI with molecular visualization
- **REST API**: FastAPI endpoints for programmatic access
- **RDKit Integration**: Molecular fingerprint generation and structure visualization

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- DGL 1.1+
- RDKit 2023.3+
- Streamlit 1.28+
- FastAPI 0.104+

## 🚀 Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/drug-combo-therapy.git
cd drug-combo-therapy
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Install RDKit (if not included)
```bash
conda install -c conda-forge rdkit
# OR
pip install rdkit-pypi
```

## 📁 Project Structure
```
drug-combo-therapy/
├── src/
│   ├── data/                    # Data loading modules
│   ├── preprocessing/           # SMILES processing and feature engineering
│   ├── graph/                   # Graph construction with DGL
│   ├── models/                  # GNN models and training
│   ├── utils/                   # Safety checker and utilities
│   └── inference/               # Prediction pipeline
├── dashboard/                   # Streamlit dashboard
├── api/                         # FastAPI endpoints
├── tests/                       # Unit tests
├── data/                        # Dataset storage
├── models/checkpoints/          # Trained model weights
├── requirements.txt
└── README.md
```

## 🎯 Quick Start

### Training the Model
```bash
python src/models/trainer.py \
    --data_path data/processed/drugcomb.csv \
    --epochs 100 \
    --batch_size 64 \
    --learning_rate 0.001
```

### Running the Dashboard
```bash
streamlit run dashboard/app.py
```

The dashboard will be available at `http://localhost:8501`

### Running the API Server
```bash
python api/main.py
# OR with uvicorn
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

API documentation available at `http://localhost:8000/docs`

### Making Predictions
```python
from src.inference.predictor import SynergyInferencePipeline
from src.models.gnn_model import SynergyPredictor
from src.utils.safety_checker import SafetyChecker

# Load model
model = SynergyPredictor.load('models/checkpoints/best_model.pt')

# Initialize pipeline
pipeline = SynergyInferencePipeline(model, graph, drug_info, SafetyChecker())

# Predict synergy
result = pipeline.predict_synergy(drug1_id=0, drug2_id=1)
print(f"Synergy Score: {result['synergy_score']:.3f}")
print(f"Confidence: {result['confidence']:.1f}%")
print(f"Safety: {result['safety_level']}")
```

## 📊 Dataset

### DrugComb Dataset

Download from: [DrugComb Portal](https://drugcomb.fimm.fi/)

Required format:
```csv
drug1_name,drug2_name,drug1_smiles,drug2_smiles,synergy_score,disease
Doxorubicin,Cyclophosphamide,CC1=C2...,C1CNP...,0.87,Breast Cancer
```

### DrugBank Interactions

Download open data from: [DrugBank](https://go.drugbank.com/)

Required format:
```csv
drug1_name,drug2_name,severity,description
Warfarin,Aspirin,major,Increased bleeding risk
```

## 🏗️ Model Architecture

### Graph Construction

- **Nodes**: Drugs, Targets, Diseases
- **Edges**: 
  - Drug → Target (targets)
  - Target → Disease (associated_with)
  - Drug → Drug (similarity)
  - Drug → Disease (treats)

### GNN Model (HeteroGraphSAGE)

- **Input**: Molecular fingerprints (Morgan, 2048-bit)
- **Hidden Layers**: 256 dimensions
- **Output Embedding**: 128 dimensions
- **Aggregation**: Mean aggregator
- **Layers**: 2-3 graph convolutional layers

### Synergy Predictor

- **Input**: Concatenated drug embeddings (256-dim)
- **Hidden Layers**: 256 → 128 → 1
- **Output**: Synergy score (0-1)
- **Loss**: MSE
- **Optimizer**: Adam with weight decay

### Confidence Estimation

- **Method**: Monte Carlo Dropout
- **Samples**: 20 forward passes
- **Output**: Mean prediction + standard deviation
- **Confidence Score**: 100 * exp(-2 * std)

## 🧪 Testing

Run all tests:
```bash
pytest tests/ -v
```

Run specific test:
```bash
pytest tests/test_model.py -v
```

With coverage:
```bash
pytest tests/ --cov=src --cov-report=html
```

## 📈 Performance Metrics

- **MSE**: Mean Squared Error on synergy scores
- **Pearson Correlation**: Between predicted and actual synergy
- **AUROC**: For binary synergy classification
- **Confidence Calibration**: ECE (Expected Calibration Error)

## 🔬 Example Results

| Disease | Drug Pair | Synergy Score | Confidence | Safety |
|---------|-----------|---------------|------------|--------|
| Breast Cancer | Doxorubicin + Cyclophosphamide | 0.87 | 92% | Safe |
| Lung Cancer | Cisplatin + Pemetrexed | 0.88 | 91% | Caution |
| Leukemia | Cytarabine + Daunorubicin | 0.90 | 94% | Caution |

## 🛠️ Configuration

Edit `config.yaml`:
```yaml
model:
  hidden_dim: 256
  embedding_dim: 128
  num_layers: 2
  dropout: 0.3

training:
  batch_size: 64
  learning_rate: 0.001
  weight_decay: 1e-5
  num_epochs: 100
  early_stopping_patience: 10

data:
  fingerprint_type: "morgan"
  fingerprint_radius: 2
  fingerprint_bits: 2048
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1
```

## 📝 Citation

If you use this code in your research, please cite:
```bibtex
@software{drug_combo_therapy,
  title={AI-Powered Drug Combination Therapy System},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/drug-combo-therapy}
}
```

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **DrugComb** for synergy data
- **DrugBank** for drug interaction data
- **RDKit** for cheminformatics tools
- **DGL** for graph neural network framework
- **PyTorch** for deep learning capabilities

## 📧 Contact

For questions or support, please open an issue or contact: your.email@example.com

## 🔮 Future Enhancements

- [ ] Multi-task learning for side effects
- [ ] Explainable AI for mechanism interpretation
- [ ] Real-time learning from clinical trials
- [ ] 3D molecular structure integration
- [ ] Pharmacokinetic/pharmacodynamic modeling
- [ ] Cross-species drug response prediction

## ⚠️ Disclaimer

This software is for research purposes only and should not be used for clinical decision-making without proper validation and regulatory approval.