#technical app interface
#python -m streamlit run app2.py- write in terminal to launch streamlit
import streamlit as st
import numpy as np
import joblib
import pandas as pd
import plotly.graph_objects as go

# Page configuration
st.set_page_config(page_title="AI Stress Classifier Dashboard", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load('wesad_stress_model.pkl')


model = load_model()
labels = {1: "Baseline", 2: "High Stress", 3: "Amusement", 4: "Meditation"}

st.markdown("# 📊 AI Stress Classification Dashboard")
st.caption("Advanced Subject-Independent Health State Recognition")

# Header Metric Cards
m1, m2, m3, m4 = st.columns(4)
m1.metric("Heart Rate", "82 BPM", "Steady")
m2.metric("HRV", "45 ms", "-2ms", delta_color="inverse")
m3.metric("EDA (Level)", "4.2 µS", "+0.1")
m4.metric("Temp (Live)", "32.1 °C", "Normal")

st.divider()

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("Manual Feature Input")
    st.info("Slicers represent the 'delta' (change) from the subject's baseline.")

    # Numeric Sliders matching the research methodology
    eda_v = st.slider("Sweat Level (EDA Change)", -3.0, 5.0, 0.0, step=0.1)
    temp_v = st.slider("Temperature Delta (°C)", -2.0, 2.0, 0.0, step=0.1)
    resp_v = st.slider("Breathing Intensity", -3.0, 5.0, 0.0, step=0.1)

    st.write("")
    if st.button("Analyze Stress Level", use_container_width=True):
        # Broadcast input to 30 features
        test_vector = np.zeros(30)
        test_vector[0:10], test_vector[10:20], test_vector[20:30] = eda_v, temp_v, resp_v

        prediction = model.predict([test_vector])[0]

        # Result Gauge
        fig_res = go.Figure(go.Indicator(
            mode="gauge",
            value=90 if prediction == 2 else (40 if prediction == 3 else 15),
            title={'text': f"Result: {labels[prediction]}"},
            gauge={
                'axis': {'range': [0, 100]},
                'steps': [
                    {'range': [0, 40], 'color': "#ADD8E6"},
                    {'range': [40, 70], 'color': "#FFFACD"},
                    {'range': [70, 100], 'color': "#FFC0CB"}
                ],
                'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': 85}
            }
        ))
        fig_res.update_layout(height=250, margin=dict(t=50, b=0))
        st.plotly_chart(fig_res, use_container_width=True)

        if prediction == 2:
            st.error("ALERT: High Stress detected. High physiological arousal.")
        else:
            st.success(f"System Normal: {labels[prediction]} detected.")

with col_right:
    st.subheader("Model Architecture")

    # Feature Importance Data
    importances = model.feature_importances_
    sums = [np.sum(importances[0:10]), np.sum(importances[10:20]), np.sum(importances[20:30])]
    chart_df = pd.DataFrame({
        'Signal Source': ['EDA (Electrodermal)', 'RESP (Respiration)', 'TEMP (Thermal)'],
        'Influence Score': sums
    })

    st.bar_chart(chart_df, x='Signal Source', y='Influence Score', color="#4682B4")

    st.subheader("Technical Methodology")
    st.markdown(f"""
    - **Normalization:** Quantile-Normal Transformation
    - **Input Type:** Feature Broadcasting (Filling all 30 statistical indices)
    - **Accuracy:** 71.52% (Subject-Independent)
    - **Algorithm:** ExtraTrees Ensemble (1500 Trees)
    """)

st.sidebar.markdown("### Model Metadata")
st.sidebar.write("**Dataset:** WESAD (Chest Sensor)")
st.sidebar.write("**Validation:** Leave-One-Subject-Out")
st.sidebar.progress(71)
st.sidebar.caption("Current Accuracy: 71.52%")
