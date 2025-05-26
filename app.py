import streamlit as st
import joblib
import numpy as np
from glob import glob
import pandas as pd


df = pd.read_csv('D:\python_mastery\machine_learning\WQU\housing buenos aires\combined_dataset_buenos_aires.csv')
neighborhoods = sorted(df['neighborhood'].dropna().unique())


# Load model
model = joblib.load(r'D:\python_mastery\machine_learning\WQU\housing buenos aires\model.pkl')

st.title("Buenos Aires Housing Price Predictor")

# Example input
lat = st.slider("Select Latitude", 
    min_value=float(df['lat'].min()), 
    max_value=float(df['lat'].max()), 
    value=float(df['lat'].mean()))

lon = st.slider("Select Longitude", 
    min_value=float(df['lon'].min()), 
    max_value=float(df['lon'].max()), 
    value=float(df['lon'].mean()))

area = st.number_input("Enter Area (in square meters)", min_value=1)
neighborhood = st.selectbox("Select Neighborhood", neighborhoods)
#st.write(df['neighborhood'].nunique())


if st.button("Predict"):
    input_data = pd.DataFrame([{
    'surface_covered_in_m2': area,
    'lat': lat,
    'lon': lon,
    'neighborhood': neighborhood
}])
    prediction = float(model.predict(input_data)[0])

    st.markdown(f"""
    <div style='
        background: rgba(33, 53, 41, 0.75);
        backdrop-filter: blur(6px);
        color: #fff;
        padding: 1rem;
        border-radius: 12px;
        font-size: 1.2rem;
        font-weight: 500;
        margin-top: 1rem;
        text-align: center;
    '>
        The predicted price is: ${prediction:,.2f}
    </div>
    """, unsafe_allow_html=True)



