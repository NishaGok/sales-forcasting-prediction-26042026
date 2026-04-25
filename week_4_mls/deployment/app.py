
import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="NishaGok/sales-forcasting-model-26042026",
    filename="best_sales_forecast_model_v1.joblib"
)

model = joblib.load(model_path)

st.title("Sales Forecast Prediction App")
st.write("Predict Product Store Sales Total using product and store details.")

product_weight = st.number_input("Product Weight", value=12.0)
product_sugar = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
allocated_area = st.number_input("Product Allocated Area", value=0.10)
product_type = st.text_input("Product Type", value="Dairy")
product_mrp = st.number_input("Product MRP", value=150.0)
store_year = st.number_input("Store Establishment Year", value=2005)
store_size = st.selectbox("Store Size", ["Small", "Medium", "High"])
store_city = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"])
store_type = st.text_input("Store Type", value="Supermarket Type1")
input_data = pd.DataFrame([{
    "Product_Weight": product_weight,
    "Product_Sugar_Content": product_sugar,
    "Product_Allocated_Area": allocated_area,
    "Product_Type": product_type,
    "Product_MRP": product_mrp,
    "Store_Establishment_Year": store_year,
    "Store_Size": store_size,
    "Store_Location_City_Type": store_city,
    "Store_Type": store_type
}])

if st.button("Predict Sales"):
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Product Store Sales Total: {round(prediction, 2)}")
