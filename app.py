import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load trained model
model = joblib.load("sensor_model.pkl")

st.title("AI Smart Sensor Fault Diagnosis System")
st.markdown("Intelligent fault detection using machine learning.")

# -----------------------------
# QUICK TEST SECTION
# -----------------------------
st.subheader("Quick Test (Mobile Friendly)")

data = None

col1, col2 = st.columns(2)

with col1:
    if st.button("Test Normal"):
        data = pd.DataFrame({"value":[25.1,25.2,24.9,25.0,25.3,25.1,25.2]})

    if st.button("Test Drift"):
        data = pd.DataFrame({"value":[20,22,24,26,28,30,32,34,36,38,40]})

with col2:
    if st.button("Test Noise"):
        data = pd.DataFrame({"value":[25,40,10,35,15,30,20,45,12,38,18]})

    if st.button("Test Stuck"):
        data = pd.DataFrame({"value":[30,30,30,30,30,30,30,30,30]})

# -----------------------------
# FILE UPLOAD
# -----------------------------
st.subheader("Upload Sensor CSV (Desktop Recommended)")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    uploaded_data = pd.read_csv(uploaded_file)
    if "value" not in uploaded_data.columns:
        st.error("CSV must contain a column named 'value'")
    else:
        data = uploaded_data

# -----------------------------
# PROCESSING SECTION
# -----------------------------
if data is not None:

    st.subheader("Sensor Signal")
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

    st.subheader("Diagnosis Result")

    if fault_type == "Normal":
        st.success(f"Status: {fault_type}")
        health_score = 100
    else:
        st.error(f"Status: {fault_type}")
        health_score = 100 - confidence

    st.metric("Confidence Level (%)", confidence)
    st.metric("Health Score (%)", round(health_score, 2))

    st.subheader("Fault Explanation")

    explanations = {
        "Normal": "Sensor is operating within expected range.",
        "Noise Fault": "High variability detected in sensor readings.",
        "Drift Fault": "Gradual change detected over time.",
        "Stuck Fault": "Sensor appears stuck at constant value."
    }

    st.info(explanations[fault_type])