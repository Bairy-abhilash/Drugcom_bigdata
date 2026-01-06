"""
Drug explorer page.
"""

import streamlit as st
import requests
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw

st.set_page_config(page_title="Drug Explorer", page_icon="🔬", layout="wide")

API_URL = "http://localhost:8000/api/v1"

st.title("🔬 Drug Explorer")

# Search
search_term = st.text_input("🔍 Search for a drug", "")

if search_term:
    try:
        response = requests.get(
            f"{API_URL}/drugs",
            params={"search": search_term, "limit": 20}
        )
        
        if response.status_code == 200:
            drugs = response.json()
            
            if drugs:
                drug_names = [d['drug_name'] for d in drugs]
                selected_drug_name = st.selectbox("Select drug", drug_names)
                
                selected_drug = next(d for d in drugs if d['drug_name'] == selected_drug_name)
                
                # Display
                st.markdown(f"## 💊 {selected_drug['drug_name']}")
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown(f"**DrugBank ID:** {selected_drug.get('drugbank_id', 'N/A')}")
                    st.markdown(f"**Mechanism:** {selected_drug.get('mechanism_of_action', 'N/A')}")
                    
                    if selected_drug.get('description'):
                        st.markdown(f"**Description:** {selected_drug['description']}")
                
                with col2:
                    if selected_drug.get('smiles'):
                        st.markdown("**🧬 Molecular Structure:**")
                        try:
                            mol = Chem.MolFromSmiles(selected_drug['smiles'])
                            if mol:
                                img = Draw.MolToImage(mol, size=(400, 400))
                                st.image(img)
                            
                            with st.expander("Show SMILES"):
                                st.code(selected_drug['smiles'])
                        except:
                            st.warning("Could not render structure")
            else:
                st.warning("No drugs found")
    except Exception as e:
        st.error(f"Error: {e}")