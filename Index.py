import streamlit as st
import pandas as pd
import numpy as np
import os
from model import load_artifacts, predict_single

# Page setup
st.set_page_config(page_title="REMI - Thyroid Disease Detector", layout="wide")

# Inject CSS
with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load components
def load_html(file):
    with open(f"Components/{file}", "r", encoding="utf-8") as f:
        return f.read()

# Tabs
tabs = st.tabs(["🏠 Home", "👤 Single Prediction", "📂 Batch Prediction", "ℹ️ About", "📬 Contact"])

# HOME
with tabs[0]:
    st.markdown(load_html("Home.html"), unsafe_allow_html=True)

# SINGLE PREDICTION
with tabs[1]:
    st.markdown(load_html("single_prediction.html"), unsafe_allow_html=True)
    model, scaler, label_map = load_artifacts()
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 0, 120, 35)
        TSH = st.number_input("TSH", 0.0, 500.0, 2.5)
        T3 = st.number_input("T3", 0.0, 10.0, 3.0)
        TT4 = st.number_input("TT4", 0.0, 500.0, 110.0)
        T4U = st.number_input("T4U", 0.0, 10.0, 1.0)
        FTI = st.number_input("FTI", 0.0, 500.0, 120.0)
    with col2:
        sex = st.selectbox("Sex", ["Female", "Male"])
        sex = 1 if sex == "Male" else 0

    input_data = {
        "age": age, "sex": sex, "TSH": TSH, "T3": T3, "TT4": TT4, "T4U": T4U, "FTI": FTI
    }

    if st.button("🔍 Predict Result"):
        label, conf, probs = predict_single(input_data)
        st.success(f"### 🧾 Prediction: {label}")
        st.info(f"Confidence: {conf}%")

# BATCH
with tabs[2]:
    st.markdown(load_html("batch_prediction.html"), unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        data = pd.read_csv(uploaded)
        model, scaler, label_map = load_artifacts()
        data_scaled = scaler.transform(data)
        preds = model.predict(data_scaled)
        data["Prediction"] = [label_map[p] for p in preds]
        st.dataframe(data)
        st.success("✅ Batch prediction completed!")

# ABOUT
with tabs[3]:
    st.markdown(load_html("About.html"), unsafe_allow_html=True)

# CONTACT
with tabs[4]:
    st.markdown(load_html("Contact.html"), unsafe_allow_html=True)
    with st.form("contact_form"):
        name = st.text_input("Your Name")
        email = st.text_input("Your Email")
        message = st.text_area("Your Message")
        submitted = st.form_submit_button("📨 Send")
        if submitted:
            st.success("✅ Message received. Thank you!")
