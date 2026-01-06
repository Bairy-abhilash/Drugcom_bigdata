"""
Predictions display page.
"""

import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Predictions", page_icon="📊", layout="wide")

st.title("📊 Top Drug Combination Predictions")

# Check if predictions exist
if 'predictions' not in st.session_state or not st.session_state['predictions']:
    st.warning("⚠️ No predictions available.")
    st.info("Please go to **Disease Selector** page and generate predictions first.")
    st.stop()

predictions = st.session_state['predictions']
disease_name = st.session_state.get('disease_name', 'Unknown')

st.markdown(f"### Results for: **{disease_name}**")
st.markdown(f"**Total Predictions:** {predictions['num_predictions']}")

# Convert to DataFrame
pred_list = predictions['predictions']
df = pd.DataFrame([
    {
        'Rank': i + 1,
        'Drug 1': p['drug1_name'],
        'Drug 2': p['drug2_name'],
        'Synergy Score': round(p['synergy_score'], 2),
        'Confidence': round(p['confidence'], 2),
        'Safety': p['safety_flag']
    }
    for i, p in enumerate(pred_list)
])

# Display table
st.markdown("### 📋 Ranked Predictions")

def highlight_safety(val):
    if val == 'Safe':
        return 'background-color: #d4edda; color: #155724'
    else:
        return 'background-color: #f8d7da; color: #721c24'

styled_df = df.style.applymap(highlight_safety, subset=['Safety'])
st.dataframe(styled_df, use_container_width=True)

# Visualization
st.markdown("### 📈 Synergy Score Distribution")

fig = px.bar(
    df,
    x='Rank',
    y='Synergy Score',
    color='Safety',
    color_discrete_map={'Safe': 'green', 'Harmful': 'red'},
    hover_data=['Drug 1', 'Drug 2', 'Confidence']
)
fig.update_layout(height=400)
st.plotly_chart(fig, use_container_width=True)

# Detailed view
st.markdown("### 🔍 Detailed View")

selected_rank = st.selectbox(
    "Select prediction for details",
    options=df['Rank'].tolist(),
    format_func=lambda x: f"Rank {x}: {df[df['Rank']==x]['Drug 1'].values[0]} + {df[df['Rank']==x]['Drug 2'].values[0]}"
)

pred_detail = pred_list[selected_rank - 1]

col1, col2 = st.columns(2)

with col1:
    st.markdown(f"#### 💊 Drug 1: {pred_detail['drug1_name']}")
    st.metric("Synergy Score", f"{pred_detail['synergy_score']:.2f}")

with col2:
    st.markdown(f"#### 💊 Drug 2: {pred_detail['drug2_name']}")
    st.metric("Confidence", f"{pred_detail['confidence']:.2f}")

# Safety
st.markdown("### 🛡️ Safety Analysis")
safety = pred_detail['safety_info']

if safety['is_safe']:
    st.success("✅ No known harmful interactions")
else:
    st.error(f"⚠️ Warning: {safety['severity']} interaction")
    st.markdown(f"**Description:** {safety.get('description', 'N/A')}")