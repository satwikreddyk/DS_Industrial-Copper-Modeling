import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import streamlit as st

data = pd.read_csv('your_data.csv')
y_regression = data['Selling_Price']
y_classification = data['Status']  # WON or LOST

X_train, X_test, y_train_regression, y_test_regression, y_train_classification, y_test_classification = train_test_split(
    X, y_regression, y_classification, test_size=0.2, random_state=42)

regression_model = RandomForestRegressor()
regression_model.fit(X_train, y_train_regression)

classification_model = RandomForestClassifier()
classification_model.fit(X_train, y_train_classification)

st.title("Copper Industry Prediction App")

input_feature1 = st.number_input("Feature 1", min_value=0.0, max_value=1000.0)
input_feature2 = st.number_input("Feature 2", min_value=0.0, max_value=1000.0)

if st.button("Predict"):
    regression_prediction = regression_model.predict([[input_feature1, input_feature2]])[0]
    classification_prediction = classification_model.predict([[input_feature1, input_feature2]])[0]

    st.write(f"Selling Price Prediction: ${regression_prediction:.2f}")
    st.write(f"Lead Status: {'WON' if classification_prediction == 1 else 'LOST'}")

