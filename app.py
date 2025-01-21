import streamlit as st
import pickle
import numpy as np

# Load the model
model_path = 'Trained_Random_Forest_Model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Feature names (if available)
try:
    feature_labels = model.feature_names_in_
except AttributeError:
    feature_labels = ["Feature " + str(i) for i in range(model.n_features_in_)]

st.title("Heart Attack Prediction")

# User input
st.header("Input Features")
user_input = []
for feature in feature_labels:
    value = st.number_input(f"Enter value for {feature}:", value=0.0)
    user_input.append(value)

# Prediction
if st.button("Predict"):
    input_data = np.array(user_input).reshape(1, -1)
    prediction = model.predict(input_data)[0]
    result = "Heart Attack Risk" if prediction == 1 else "No Heart Attack Risk"
    st.success(result)
