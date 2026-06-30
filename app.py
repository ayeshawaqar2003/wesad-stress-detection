#user friendly app interface
#python -m streamlit run app.py- write in terminal to launch streamlit
import streamlit as st
import numpy as np
import joblib
import plotly.graph_objects as go

# Page configuration
st.set_page_config(page_title="Personal Wellness Monitor", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load('wesad_stress_model.pkl')


model = load_model()

# Mapping for user results
labels = {
    1: "Relaxed & Calm",
    2: "Stressed / Anxious",
    3: "Amused / Happy",
    4: "Deeply Meditating"
}

# UI Layout
st.markdown("# 🌿 Personal Wellness Monitor")
st.write("This AI assistant predicts your emotional state based on your body's vital signs.")

with st.expander("📖 How to use this dashboard"):
    st.write(
        "Adjust the sliders below to reflect how you feel physically. The AI will calculate your current emotional state.")

st.divider()

col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("How does your body feel?")

    # User-friendly Sliders
    eda_txt = st.select_slider("**Skin & Sweat** (EDA)",
                               options=["Dry", "Normal", "Slightly Clammy", "Sweaty"],
                               value="Normal")

    temp_txt = st.select_slider("**Body Temperature** (TEMP)",
                                options=["Chilly", "Normal", "Warm", "Hot"],
                                value="Normal")

    resp_txt = st.select_slider("**Breathing** (RESP)",
                                options=["Slow/Deep", "Normal", "Fast/Shallow", "Short of Breath"],
                                value="Normal")

    # Mapping words to Quantile-Transformed values (-3.0 to 3.0 range)
    eda_map = {"Dry": -1.2, "Normal": 0.0, "Slightly Clammy": 1.5, "Sweaty": 3.2}
    temp_map = {"Chilly": -1.5, "Normal": 0.0, "Warm": 1.2, "Hot": 2.8}
    resp_map = {"Slow/Deep": -1.8, "Normal": 0.0, "Fast/Shallow": 1.5, "Short of Breath": 3.5}

    st.write("")
    analyze = st.button("Check My State", use_container_width=True, type="primary")

with col2:
    st.subheader("AI Result")
    if analyze:
        # Broadcast 3 values to 30 features
        input_data = np.zeros(30)
        input_data[0:10] = eda_map[eda_txt]
        input_data[10:20] = temp_map[temp_txt]
        input_data[20:30] = resp_map[resp_txt]

        pred = model.predict([input_data])[0]
        probs = model.predict_proba([input_data])[0]
        confidence = np.max(probs) * 100

        # Gauge Visualization
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence,
            number={'suffix': "% Confidence", 'font': {'size': 50}},
            title={'text': f"Predicted: {labels[pred]}",
                   'font': {'size': 24, 'color': "red" if pred == 2 else "green"}},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#e74c3c" if pred == 2 else "#2ecc71"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "#d3d3d3",
            }
        ))
        fig.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)

        # Contextual Advice Box
        if pred == 2:
            st.error(
                "⚠️ **Stress Detected.** It might be a good time to take a 5-minute break or try some box breathing.")
        elif pred == 1:
            st.success("✅ **State: Balanced.** Your vitals suggest you are currently relaxed and in a baseline state.")
        elif pred == 3:
            st.info("😊 **State: Positive Arousal.** You seem amused or in high spirits!")
        elif pred == 4:
            st.info("🧘 **State: Deep Calm.** Your vitals indicate a meditative or deeply relaxed state.")
