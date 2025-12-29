import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("sales_model.pkl")

st.title("Sales Prediction App")

# Input features
unit_price = st.number_input("Unit Price")
quantity = st.number_input("Quantity")
tax = st.number_input("Tax 5%")
cogs = st.number_input("COGS")
gross_margin = st.number_input("Gross Margin %")
gross_income = st.number_input("Gross Income")

# Make prediction when button clicked
if st.button("Predict Sales"):
    input_df = pd.DataFrame([[unit_price, quantity, tax, cogs, gross_margin, gross_income]],
                            columns=["Unit price","Quantity","Tax 5%","cogs","gross margin percentage","gross income"])
    prediction = model.predict(input_df)
    st.success(f"Predicted Sales: {prediction[0]:.2f}")
