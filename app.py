# app.py
import streamlit as st
import numpy as np
import joblib

# Load models and scaler
scaler = joblib.load("scaler.pkl")
dtree1 = joblib.load("dtree_classifier.pkl")

# Tier label mapping
tier_labels = {
    0: "Low Performance",
    1: "Medium Performance",
    2: "High Performance"
}

# Streamlit app UI
st.set_page_config(page_title="Apple Sales Classifier", layout="centered")
st.title("\U0001F34E Apple Sales Performance Classifier")
st.markdown("Enter Apple product sales data below to performance tiers.")

# User inputs
iphone = st.number_input("iPhone Sales (in million units)", min_value=0.0, step=0.1)
ipad = st.number_input("iPad Sales (in million units)", min_value=0.0, step=0.1)
mac = st.number_input("Mac Sales (in million units)", min_value=0.0, step=0.1)
wearables = st.number_input("Wearables (in million units)", min_value=0.0, step=0.1)

# Predict button
if st.button("Predict Performance"):
    try:
        # Prepare input and scale it
        input_data = np.array([[iphone, ipad, mac, wearables]])
        input_scaled = scaler.transform(input_data)

        # Predictions
        tier1 = dtree1.predict(input_scaled)[0]

        # Display results
        st.info(f"Predicted Performance Tier 1: {tier_labels.get(tier1, 'Unknown')}")

    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
