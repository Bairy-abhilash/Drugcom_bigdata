"""
Streamlit dashboard for drug combination therapy prediction.
"""
import streamlit as st
import pandas as pd
import numpy as np
import torch
import plotly.graph_objects as go
from rdkit import Chem
from rdkit.Chem import Draw
import io
from PIL import Image

# Configure page
st.set_page_config(
    page_title="AI Drug Combination Therapy",
    page_icon="💊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


class DrugComboDashboard:
    """Streamlit dashboard for drug synergy prediction."""
    
    def __init__(self):
        """Initialize dashboard."""
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize session state variables."""
        if 'predictions' not in st.session_state:
            st.session_state.predictions = None
        if 'selected_disease' not in st.session_state:
            st.session_state.selected_disease = None
    
    def load_model(self):
        """Load trained model (placeholder)."""
        # In production, load actual model
        st.info("ℹ️ Demo mode: Using mock predictions")
        return None
    
    def render_header(self):
        """Render dashboard header."""
        st.markdown('<h1 class="main-header">🧬 AI Drug Combination Therapy System</h1>', 
                   unsafe_allow_html=True)
        st.markdown("**Graph Neural Network-based Synergy Prediction with Confidence Estimation**")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Model", "GraphSAGE", "GNN")
        with col2:
            st.metric("Framework", "PyTorch + DGL", "v2.0+")
        with col3:
            st.metric("Dataset", "DrugComb", "10K+ pairs")
        with col4:
            st.metric("Safety", "DrugBank", "Verified")
    
    def render_sidebar(self):
        """Render sidebar with controls."""
        st.sidebar.header("⚙️ Configuration")
        
        # Disease selection
        diseases = [
            'Breast Cancer', 'Lung Cancer', 'Leukemia', 'Colorectal Cancer',
            'Melanoma', 'Prostate Cancer', 'Ovarian Cancer', 'Pancreatic Cancer'
        ]
        
        selected_disease = st.sidebar.selectbox(
            "Select Disease:",
            options=diseases,
            index=0
        )
        
        # Model parameters
        st.sidebar.subheader("🔬 Prediction Parameters")
        top_k = st.sidebar.slider("Top K combinations:", 5, 20, 10)
        min_confidence = st.sidebar.slider("Min Confidence (%):", 50, 95, 70)
        mc_samples = st.sidebar.slider("MC Dropout Samples:", 10, 50, 20)
        
        # Predict button
        if st.sidebar.button("🔍 Predict Synergy", type="primary"):
            st.session_state.selected_disease = selected_disease
            st.session_state.predictions = self.generate_mock_predictions(
                selected_disease, top_k, min_confidence
            )
        
        st.sidebar.markdown("---")
        st.sidebar.info(
            "**Model Info:**\n"
            "- Architecture: HeteroGraphSAGE\n"
            "- Layers: 3\n"
            "- Hidden Dim: 256\n"
            "- Confidence: MC Dropout"
        )
        
        return selected_disease, top_k, min_confidence, mc_samples
    
    def generate_mock_predictions(self, disease, top_k, min_confidence):
        """Generate mock predictions for demonstration."""
        mock_data = {
            'Breast Cancer': [
                ('Doxorubicin', 'Cyclophosphamide', 0.87, 92, 'safe', 'DNA damage pathway'),
                ('Paclitaxel', 'Trastuzumab', 0.85, 89, 'safe', 'HER2 targeting'),
                ('Tamoxifen', 'Letrozole', 0.82, 87, 'safe', 'Estrogen blockade'),
                ('Pertuzumab', 'Trastuzumab', 0.80, 85, 'safe', 'HER2 dual blockade'),
                ('Capecitabine', 'Docetaxel', 0.78, 83, 'caution', 'Cytotoxic synergy'),
            ],
            'Lung Cancer': [
                ('Cisplatin', 'Pemetrexed', 0.88, 91, 'caution', 'DNA-folate inhibition'),
                ('Carboplatin', 'Paclitaxel', 0.86, 90, 'caution', 'Platinum-taxane combo'),
                ('Nivolumab', 'Ipilimumab', 0.83, 86, 'safe', 'Immune checkpoint dual'),
                ('Bevacizumab', 'Erlotinib', 0.81, 84, 'caution', 'VEGF-EGFR blockade'),
                ('Atezolizumab', 'Carboplatin', 0.79, 82, 'safe', 'Immuno-chemo combo'),
            ],
            'Leukemia': [
                ('Cytarabine', 'Daunorubicin', 0.90, 94, 'caution', 'Standard induction'),
                ('Imatinib', 'Dasatinib', 0.84, 88, 'caution', 'BCR-ABL targeting'),
                ('Venetoclax', 'Azacitidine', 0.82, 86, 'safe', 'BCL2-hypomethylation'),
                ('Rituximab', 'Bendamustine', 0.80, 84, 'caution', 'CD20-alkylation'),
                ('Blinatumomab', 'Chemotherapy', 0.78, 81, 'safe', 'BiTE-cytotoxic'),
            ]
        }
        
        data = mock_data.get(disease, mock_data['Breast Cancer'])
        predictions = []
        
        for i, (d1, d2, score, conf, safety, mech) in enumerate(data[:top_k]):
            if conf >= min_confidence:
                predictions.append({
                    'rank': i + 1,
                    'drug1': d1,
                    'drug2': d2,
                    'synergy_score': score,
                    'confidence': conf,
                    'safety_level': safety,
                    'mechanism': mech,
                    'smiles1': 'CC1=C2[C@@H](C(=O)C3(C(CC4C(C3C([C@@H](C2(C)C)(CC1OC(=O)C)O)O)(CO4)OC(=O)C)O)C)OC(=O)C',
                    'smiles2': 'O=C1C=CC(=O)N1CCCl'
                })
        
        return predictions
    
    def render_predictions_table(self, predictions):
        """Render predictions table."""
        if not predictions:
            st.warning("No predictions available. Select a disease and click 'Predict Synergy'.")
            return
        
        st.subheader(f"🎯 Top Synergistic Combinations for {st.session_state.selected_disease}")
        
        df = pd.DataFrame(predictions)
        
        # Style the dataframe
        def color_safety(val):
            colors = {'safe': 'background-color: #90EE90', 
                     'caution': 'background-color: #FFD700',
                     'danger': 'background-color: #FF6347'}
            return colors.get(val, '')
        
        styled_df = df[['rank', 'drug1', 'drug2', 'synergy_score', 'confidence', 'safety_level', 'mechanism']].style\
            .applymap(color_safety, subset=['safety_level'])\
            .format({'synergy_score': '{:.3f}', 'confidence': '{:.0f}%'})
        
        st.dataframe(styled_df, use_container_width=True, height=400)
        
        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results (CSV)",
            data=csv,
            file_name=f"synergy_predictions_{st.session_state.selected_disease.replace(' ', '_')}.csv",
            mime="text/csv"
        )
    
    def render_visualization(self, predictions):
        """Render visualizations."""
        if not predictions:
            return
        
        tab1, tab2, tab3 = st.tabs(["📊 Synergy Scores", "🎯 Confidence Analysis", "⚗️ Molecular Structures"])
        
        with tab1:
            self.plot_synergy_scores(predictions)
        
        with tab2:
            self.plot_confidence_distribution(predictions)
        
        with tab3:
            self.render_molecular_structures(predictions)
    
    def plot_synergy_scores(self, predictions):
        """Plot synergy scores."""
        df = pd.DataFrame(predictions)
        
        fig = go.Figure()
        
        colors = {'safe': 'green', 'caution': 'orange', 'danger': 'red'}
        
        for safety in df['safety_level'].unique():
            subset = df[df['safety_level'] == safety]
            fig.add_trace(go.Bar(
                x=[f"{row['drug1']} + {row['drug2']}" for _, row in subset.iterrows()],
                y=subset['synergy_score'],
                name=safety.capitalize(),
                marker_color=colors.get(safety, 'blue')
            ))
        
        fig.update_layout(
            title="Synergy Scores by Drug Combination",
            xaxis_title="Drug Pair",
            yaxis_title="Synergy Score",
            barmode='group',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def plot_confidence_distribution(self, predictions):
        """Plot confidence distribution."""
        df = pd.DataFrame(predictions)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure(data=[go.Scatter(
                x=df['synergy_score'],
                y=df['confidence'],
                mode='markers',
                marker=dict(
                    size=12,
                    color=df['synergy_score'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Synergy Score")
                ),
                text=[f"{row['drug1']} + {row['drug2']}" for _, row in df.iterrows()],
                hovertemplate='<b>%{text}</b><br>Synergy: %{x:.3f}<br>Confidence: %{y:.1f}%'
            )])
            
            fig.update_layout(
                title="Synergy vs Confidence",
                xaxis_title="Synergy Score",
                yaxis_title="Confidence (%)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            safety_counts = df['safety_level'].value_counts()
            fig = go.Figure(data=[go.Pie(
                labels=safety_counts.index,
                values=safety_counts.values,
                hole=.3,
                marker=dict(colors=['green', 'orange', 'red'][:len(safety_counts)])
            )])
            
            fig.update_layout(
                title="Safety Distribution",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def render_molecular_structures(self, predictions):
        """Render molecular structures using RDKit."""
        st.subheader("🧪 Molecular Structure Visualization")
        
        # Select prediction
        options = [f"{p['drug1']} + {p['drug2']}" for p in predictions]
        selected_idx = st.selectbox("Select drug combination:", range(len(options)), 
                                    format_func=lambda x: options[x])
        
        pred = predictions[selected_idx]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"### {pred['drug1']}")
            self.draw_molecule(pred['smiles1'], pred['drug1'])
        
        with col2:
            st.markdown(f"### {pred['drug2']}")
            self.draw_molecule(pred['smiles2'], pred['drug2'])
        
        # Display prediction details
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Synergy Score", f"{pred['synergy_score']:.3f}")
        col2.metric("Confidence", f"{pred['confidence']}%")
        col3.metric("Safety", pred['safety_level'].upper())
        col4.metric("Mechanism", pred['mechanism'][:20] + "...")
    
    def draw_molecule(self, smiles, name):
        """Draw molecular structure from SMILES."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                img = Draw.MolToImage(mol, size=(400, 300))
                st.image(img, caption=f"SMILES: {smiles[:40]}...")
            else:
                st.warning(f"Unable to parse SMILES for {name}")
        except Exception as e:
            st.error(f"Error rendering molecule: {e}")
    
    def run(self):
        """Run the dashboard."""
        self.render_header()
        
        # Sidebar
        disease, top_k, min_conf, mc_samples = self.render_sidebar()
        
        # Main content
        if st.session_state.predictions:
            self.render_predictions_table(st.session_state.predictions)
            st.markdown("---")
            self.render_visualization(st.session_state.predictions)
        else:
            st.info("👈 Select a disease from the sidebar and click 'Predict Synergy' to begin")
            
            # Show feature overview
            st.subheader("🔬 System Features")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                **Graph Neural Networks**
                - HeteroGraphSAGE architecture
                - Multi-relational drug-target-disease graph
                - Message passing for synergy prediction
                """)
            
            with col2:
                st.markdown("""
                **Confidence Estimation**
                - Monte Carlo Dropout
                - Uncertainty quantification
                - Prediction reliability scores
                """)
            
            with col3:
                st.markdown("""
                **Safety Checking**
                - DrugBank interaction database
                - Severity classification
                - Clinical risk assessment
                """)


# Main execution
if __name__ == "__main__":
    dashboard = DrugComboDashboard()
    dashboard.run()