import streamlit as st
import joblib
import numpy as np

# Load the trained model
rf_model = joblib.load("C:/Users/hp/Downloads/random_forest_model.pkl")

# Streamlit UI
st.title("Fitness Tracker - Calorie Prediction")

# Input fields for user to enter values
st.write("Enter values for the following features:")

feature_names = ['TotalSteps', 'TotalDistance', 'VeryActiveMinutes', 'FairlyActiveMinutes', 'LightlyActiveMinutes', 'SedentaryMinutes']
user_input = []

for feature in feature_names:
    value = st.number_input(f"{feature}", min_value=0.0, step=0.1)
    user_input.append(value)

# Predict button
if st.button("Predict"):
    user_data = np.array(user_input).reshape(1, -1)
    prediction = rf_model.predict(user_data)
    st.success(f"Predicted Calories Burned: {prediction[0]:.2f}")