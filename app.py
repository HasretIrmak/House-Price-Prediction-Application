import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the pre-trained model
model = joblib.load("house_price_model.pkl")

# Title and instructions
st.title("House Price Prediction Application")
st.write("""
Enter the details of the house below, and click 'Predict Price' to get the estimated price.
""")

# Input fields for user
st.header("House Features:")
lot_area = st.number_input("Lot Area (e.g., 8500)", min_value=0, value=8500)
year_built = st.number_input("Year Built (e.g., 2000)", min_value=1800, max_value=2025, value=2000)
total_bsmt_sf = st.number_input("Total Basement Area (e.g., 800)", min_value=0, value=800)
gr_liv_area = st.number_input("Above Ground Living Area (e.g., 1500)", min_value=0, value=1500)

# Predict button
if st.button("Predict Price"):
    features = np.array([[lot_area, year_built, total_bsmt_sf, gr_liv_area]])
    prediction = model.predict(features)
    st.success(f"The estimated house price is: ${prediction[0]:,.2f}")
