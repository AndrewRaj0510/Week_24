import streamlit as st
import joblib
import numpy as np

# Load your model
model = joblib.load('model_svc.joblib')

st.title("ML Model Prediction App")

# Get user input
feature1 = st.number_input("Enter feature 1:")
feature2 = st.number_input("Enter feature 2:")
# Add as many features as needed...

if st.button("Predict"):
    input_data = np.array([[feature1, feature2]])  # match model input shape
    prediction = model.predict(input_data)
    st.success(f"Prediction: {prediction[0]}")
