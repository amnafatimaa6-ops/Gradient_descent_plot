# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from model import load_data, prepare_data, train_model, predict_price

st.title("Interactive House Price Playground")

# Load and prepare data
df = load_data()
X, y, enc, feature_names = prepare_data(df)
model = train_model(X, y)

# Sidebar inputs
property_type = st.sidebar.selectbox("Property Type", df['type'].unique())
location = st.sidebar.selectbox("Location", df['location'].unique())
furnishing = st.sidebar.selectbox("Furnishing Status", df['furnishing_status'].unique())
bedrooms = st.sidebar.slider("Bedrooms", int(df['bedrooms'].min()), int(df['bedrooms'].max()), 3)
bathrooms = st.sidebar.slider("Bathrooms", int(df['bathrooms'].min()), int(df['bathrooms'].max()), 3)
area = st.sidebar.slider("Area (sqft)", int(df['area sqft'].min()), int(df['area sqft'].max()), 1500)

custom_input = {
    'type': property_type,
    'location': location,
    'furnishing_status': furnishing,
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'area sqft': area
}

# Predict price
predicted_price = predict_price(model, enc, feature_names, custom_input)

st.subheader(f"Predicted Price: {predicted_price:,.0f} PKR")

# Gradient-Style Visual: show impact of changing one feature
feature_to_test = st.selectbox("Test impact of feature", ['bedrooms','bathrooms','area sqft'])
values = np.linspace(df[feature_to_test].min(), df[feature_to_test].max(), 50)
predictions = []

for v in values:
    temp = custom_input.copy()
    temp[feature_to_test] = v
    price = predict_price(model, enc, feature_names, temp)
    predictions.append(price)

# 3D-ish plot: feature vs predicted price
fig = go.Figure()
fig.add_trace(go.Scatter(x=values, y=predictions, mode='lines+markers', name='Predicted Price'))
fig.update_layout(
    title=f"Impact of changing {feature_to_test} on predicted price",
    xaxis_title=feature_to_test,
    yaxis_title="Price (PKR)",
)
st.plotly_chart(fig)
