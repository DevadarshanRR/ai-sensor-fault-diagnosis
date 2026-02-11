import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("sensor_model.pkl")

# App Title
st.title("AI Smart Sensor Fault Diagnosis System")

# ----------------------------
# SAMPLE FILE DOWNLOAD SECTION
# ----------------------------
st.subheader("Download Sample Files")

# Normal Sample
normal_sample = """value
25.1
25.2
24.9
25.0
25.3
25.1
25.2
25.0
"""

st.download_button(
    label="Download Normal Sample",
    data=normal_sample,
    file_name="normal_sample.csv",
    mime="text/csv"
)

# Noise Sample
noise_sample = """value
25
40
10
35
15
30
20
45
12
38
18
33
"""

st.download_button(
    label="Download Noise Sample",
    data=noise_sample,
    file_name="noise_sample.csv",
    mime="text/csv"
)

# ----------------------------
# FILE UPLOAD SECTION
# ----------------------------
st.subheader("Upload Sensor CSV")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # Safety check
    if "value" not in data.columns:
        st.error("CSV must contain a column named 'value'")
    else:
        st.subheader("Sensor Signal")
        st.line_chart(data["value"])

        # Feature extraction
        mean = data["value"].mean()
        std = data["value"].std()

        # Prediction
        prediction = model.predict([[mean, std]])

        labels = {
            0: "Normal",
            1: "Noise Fault",
            2: "Drift Fault",
            3: "Stuck Fault"
        }

        st.subheader("Diagnosis Result")
        st.success(f"Fault Type: {labels[prediction[0]]}")