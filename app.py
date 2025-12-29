import streamlit as st
import pandas as pd
import joblib
from datetime import date, time

# Load trained model
model = joblib.load("supermarket.pkl")

st.title("Supermarket Sales Prediction App")
st.write("Predict total sales based on purchase details")

# Input features
# User inputs
unit_price = st.number_input("Unit Price", min_value=0.0)
quantity = st.number_input("Quantity", min_value=1)
purchase_date = st.date_input("Date of Purchase")
purchase_time = st.time_input("Time of Purchase")
gender = st.selectbox("Gender", ["Male", "Female"])
product_line = st.selectbox(
    "Product Line",
    ["Fashion accessories", "Food and beverages",
     "Health and beauty", "Home and lifestyle", "Sports and travel", "Electronic accessories", "Fashion accessories"]
)

# Extract features from date & time
month = purchase_date.month
hour = purchase_time.hour
# Create input DataFrame
input_df = pd.DataFrame({
    "Unit price": [unit_price],
    "Quantity": [quantity],
    "Month": [month],
    "Hour": [hour],
    "Product line": [product_line]
})

# Fill categorical features
if gender == "Male":
    input_df["Gender_Male"] = 1
input_df[f"Product line_{product_line}"] = 1

# Make a prediction when the button is clicked
if st.button("Predict Sales"):
    prediction = model.predict(input_df)
    st.success(f"Predicted Total Sales: {prediction[0]:.2f}")


