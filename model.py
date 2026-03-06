# model.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

# Load data
def load_data():
    df = pd.read_csv("house_prices.csv")  
    return df

# Prepare features
def prepare_data(df):
    df = df.copy()
    X_cat = df[['type','location','furnishing_status']]
    X_num = df[['bedrooms','bathrooms','area sqft']]
    
    # One-hot encode categorical
    enc = OneHotEncoder(sparse=False)
    X_cat_enc = enc.fit_transform(X_cat)
    feature_names = enc.get_feature_names_out(['type','location','furnishing_status'])
    
    # Combine numerical + categorical
    X = np.hstack([X_num.values, X_cat_enc])
    y = df['price'].values.reshape(-1,1)
    return X, y, enc, feature_names

# Train a simple linear regression
def train_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

# Predict price for a custom input
def predict_price(model, enc, feature_names, custom_features):
    # custom_features: dict with keys 'type','location','furnishing_status','bedrooms','bathrooms','area sqft'
    X_num = np.array([[custom_features['bedrooms'], 
                       custom_features['bathrooms'], 
                       custom_features['area sqft']]])
    
    X_cat = pd.DataFrame([{
        'type': custom_features['type'],
        'location': custom_features['location'],
        'furnishing_status': custom_features['furnishing_status']
    }])
    X_cat_enc = enc.transform(X_cat)
    X_input = np.hstack([X_num, X_cat_enc])
    
    price = model.predict(X_input)
    return price[0,0]
