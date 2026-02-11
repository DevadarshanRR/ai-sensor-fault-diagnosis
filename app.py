import streamlit as st
import pandas as pd
import joblib

model = joblib.load("sensor_model.pkl")

st.title("AI Smart Sensor Fault Diagnosis System")

# Sample CSV download
sample_data = """value
25.1
25.2
24.9
25.0
25.3
"""

st.download_button(
    label="Download Sample CSV",
    data=sample_data,
    file_name="sample_sensor_data.csv",
    mime="text/csv"
)

uploaded_file = st.file_uploader("Upload Sensor CSV")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    
    st.subheader("Sensor Signal")
    st.line_chart(data["value"])
    
    mean = data["value"].mean()
    std = data["value"].std()
    
    prediction = model.predict([[mean, std]])
    
    labels = {
        0: "Normal",
        1: "Noise Fault",
        2: "Drift Fault",
        3: "Stuck Fault"
    }
    
    st.subheader("Diagnosis Result")
    st.success(f"Fault Type: {labels[prediction[0]]}")