"""
Main home page for Drug Synergy Prediction Dashboard.
Streamlit automatically uses this as the first page.
"""

import streamlit as st

st.set_page_config(
    page_title="Drug Synergy Prediction",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">💊 AI-Powered Drug Synergy Prediction</div>', 
            unsafe_allow_html=True)

st.title("Welcome to Drug Synergy Prediction Platform")

st.markdown("""
### 🎯 Overview
This platform uses **Graph Neural Networks (GNNs)** to predict synergistic drug combinations
for cancer treatment.

### 🔬 Key Features
- **Disease-Specific Predictions**: Get ranked drug combinations
- **Safety Analysis**: Automatic checking for harmful interactions
- **Molecular Visualization**: View chemical structures
- **Confidence Scores**: Uncertainty quantification

### 🚀 Quick Start
1. Navigate to **Disease Selector** in the sidebar
2. Choose a disease type
3. Generate predictions
4. View results in **Predictions** page

### 📊 Statistics
""")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Diseases", "5+", "Cancer Types")
with col2:
    st.metric("Drugs", "100+", "Compounds")
with col3:
    st.metric("Model Accuracy", "85%", "Correlation")

st.markdown("---")
st.info("👈 Use the sidebar to navigate between pages")