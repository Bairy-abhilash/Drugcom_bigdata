"""
Disease selection page.
"""

import streamlit as st
import requests

st.set_page_config(page_title="Disease Selector", page_icon="🔍")

API_URL = "http://localhost:8000/api/v1"

st.title("🔍 Select Disease for Prediction")

# Fetch diseases
@st.cache_data
def get_diseases():
    try:
        response = requests.get(f"{API_URL}/diseases")
        if response.status_code == 200:
            return response.json()
        return []
    except:
        return []

diseases = get_diseases()

if not diseases:
    st.warning("No diseases found. Please ensure the API is running and ETL data is loaded.")
    st.stop()

# Disease selector
disease_options = {d['disease_name']: d['disease_id'] for d in diseases}
selected_disease_name = st.selectbox(
    "Select Disease Type",
    options=list(disease_options.keys())
)

selected_disease_id = disease_options[selected_disease_name]
selected_disease = next(d for d in diseases if d['disease_id'] == selected_disease_id)

# Display disease info
st.markdown("### Disease Information")
col1, col2 = st.columns(2)

with col1:
    st.markdown(f"**Name:** {selected_disease['disease_name']}")
    st.markdown(f"**Type:** {selected_disease.get('disease_type', 'N/A')}")

with col2:
    st.markdown(f"**Tissue:** {selected_disease.get('tissue_type', 'N/A')}")
    st.markdown(f"**ID:** {selected_disease['disease_id']}")

if selected_disease.get('description'):
    st.markdown(f"**Description:** {selected_disease['description']}")

# Prediction settings
st.markdown("### Prediction Settings")
top_k = st.slider("Number of top predictions", 5, 50, 10)

# Generate predictions
if st.button("🚀 Generate Predictions", type="primary"):
    with st.spinner("Running GNN model..."):
        try:
            response = requests.get(
                f"{API_URL}/predict",
                params={"disease_id": selected_disease_id, "top_k": top_k}
            )
            
            if response.status_code == 200:
                st.session_state['predictions'] = response.json()
                st.session_state['disease_name'] = selected_disease_name
                st.success("✅ Predictions generated successfully!")
                st.info("Navigate to **Predictions** page to view results →")
            else:
                st.error(f"Prediction failed: {response.text}")
        except Exception as e:
            st.error(f"Error: {str(e)}")