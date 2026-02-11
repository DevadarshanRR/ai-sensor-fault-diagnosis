import streamlit as st
import pandas as pd
import joblib
import numpy as np

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="AI Sensor Fault Diagnosis", layout="centered")

# Load trained model
model = joblib.load("sensor_model.pkl")

# -------------------------
# HEADER
# -------------------------
st.markdown(
    """
    <h1 style='text-align: center; color: #2E86C1;'>
    AI Smart Sensor Fault Diagnosis System
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center;'>Machine Learning-powered sensor health monitoring dashboard</p>",
    unsafe_allow_html=True
)

st.divider()

# -------------------------
# QUICK TEST SECTION
# -------------------------
st.subheader("📱 Quick Test (Mobile Friendly)")

st.markdown("Use the buttons below to simulate different sensor conditions.")

data = None

col1, col2 = st.columns(2)

with col1:
    if st.button("Normal Condition"):
        data = pd.DataFrame({"value":[25.1,25.2,24.9,25.0,25.3,25.1,25.2]})

    if st.button("Drift Fault"):
        data = pd.DataFrame({"value":[20,22,24,26,28,30,32,34,36,38,40]})

with col2:
    if st.button("Noise Fault"):
        data = pd.DataFrame({"value":[25,40,10,35,15,30,20,45,12,38,18]})

    if st.button("Stuck Fault"):
        data = pd.DataFrame({"value":[30,30,30,30,30,30,30,30,30]})

st.divider()

# -------------------------
# FILE UPLOAD SECTION
# -------------------------
st.subheader("💻 Upload Your Own Sensor Data")

uploaded_file = st.file_uploader(
    "Upload CSV file with a column named 'value'",
    type=["csv"]
)

if uploaded_file is not None:
    uploaded_data = pd.read_csv(uploaded_file)
    if "value" not in uploaded_data.columns:
        st.error("CSV must contain a column named 'value'")
    else:
        data = uploaded_data

# -------------------------
# PROCESSING SECTION
# -------------------------
if data is not None:

    st.subheader("📈 Sensor Signal Visualization")
    st.line_chart(data["value"])

    mean = data["value"].mean()
    std = data["value"].std()

    features = [[mean, std]]

    prediction = model.predict(features)
    probabilities = model.predict_proba(features)

    labels = {
        0: "Normal",
        1: "Noise Fault",
        2: "Drift Fault",
        3: "Stuck Fault"
    }

    fault_type = labels[prediction[0]]
    confidence = round(np.max(probabilities) * 100, 2)

    st.divider()
    st.subheader("🔍 Diagnosis Result")

    if fault_type == "Normal":
        st.success(f"✅ Status: {fault_type}")
        health_score = 100
    else:
        st.error(f"⚠️ Status: {fault_type}")
        health_score = 100 - confidence

    colA, colB = st.columns(2)

    with colA:
        st.metric("Confidence Level (%)", confidence)

    with colB:
        st.metric("Health Score (%)", round(health_score, 2))

    explanations = {
        "Normal": "Sensor is operating within expected range.",
        "Noise Fault": "High variability detected in sensor readings.",
        "Drift Fault": "Gradual trend change observed over time.",
        "Stuck Fault": "Sensor output appears constant (possible failure)."
    }

    st.info(explanations[fault_type])

st.divider()

st.caption("Developed using Python, Machine Learning, and Streamlit.")