import streamlit as st
import pandas as pd
import joblib
from datetime import date, time

# Load trained model
model = joblib.load("supermarket.pkl")
model_columns = joblib.load("model_columns.pkl")


st.title("Supermarket Sales Prediction App")
st.write("Predict total sales based on purchase details")

# Input features
# User inputs
unit_price = st.number_input("Unit Price", min_value=0.0)
quantity = st.number_input("Quantity", min_value=1, step=1)
purchase_date = st.date_input("Date of Purchase",  value=date.today())
purchase_time = st.time_input("Time of Purchase")
gender = st.selectbox("Gender", ["Male", "Female"])
product_line = st.selectbox(
    "Product Line",
    ["Fashion accessories", "Food and beverages",
     "Health and beauty", "Home and lifestyle", "Sports and travel", "Electronic accessories"]
)

# Extract features from date & time
month = purchase_date.month
hour = purchase_time.hour
# Create input DataFrame
input_df = pd.DataFrame(0, index=[0], columns=model_columns)

# Fill numeric features
# Mapping model column names to the variables
col_var_map = {
    "Unit price": unit_price,
    "Quantity": quantity,
    "Month": month,
    "Hour": hour
}

for col, val in col_var_map.items():
    input_df[col] = val
    
# Fill categorical features
if "Gender_Male" in model_columns and gender == "Male":
    input_df["Gender_Male"] = 1
    
product_col_name = f"Product line_{product_line}"
if product_col_name in model_columns:
    input_df[product_col_name] = 1

# Make a prediction when the button is clicked
if st.button("Predict Sales"):
    prediction = model.predict(input_df)
    st.success(f"Predicted Total Sales: {prediction[0]:.2f}")






