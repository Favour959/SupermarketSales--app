import streamlit as st
import pandas as pd
import joblib
from datetime import date, time

# Load trained model
model = joblib.load("sales_model.pkl")

st.title("Supermarket Sales Prediction App")
st.write("Predict total sales based on purchase details")

# Input features
# User inputs
unit_price = st.number_input("Unit Price", min_value=0.0)
quantity = st.number_input("Quantity", min_value=1)
purchase_date = st.date_input("Date of Purchase")
purchase_time = st.time_input("Time of Purchase")
# Extract features from date & time
month = purchase_date.month
hour = purchase_time.hour
# Create input DataFrame
input_df = pd.DataFrame({
    "Unit price": [unit_price],
    "Quantity": [quantity],
    "Month": [month],
    "Hour": [hour]
})


# Make prediction when button clicked
if st.button("Predict Sales"):
    prediction = model.predict(input_df)
    st.success(f"Predicted Total Sales: {prediction[0]:.2f}")
