
# streamlit application
import streamlit as st
import pandas as pd
import pickle
import numpy as np

# model and scaler load
model = pickle.load(open("notebook/model.sav", "rb"))
scaler = pickle.load(open("notebook/scaler.sav", "rb"))

# predictive system
def prediction_system(input_data):
    input_data = (4,110,92,0,0,37.6,0.191,30)

    # as array
    input_array = np.asarray(input_data)
    input_reshaped = input_array.reshape(1, -1)
    input_scaled = scaler.transform(input_reshaped)

    #prediction
    prediction = model.predict(input_scaled)
    if (prediction[0] == 1):
        return "Diabetic"
    else:
        return "Not diabetics"

# main function
def main():

    # title
    st.title("Diabetes Prediction Model")

    # getting input data from the user
    Pregnancies = st.text_input("Number of Pregnancies")
    Glucose = st.text_input("Glucose Level")
    BloodPressure = st.text_input("Blood Pressure")
    SkinThickness = st.text_input("Skin Thickness")
    Insulin = st.text_input("Insulin Level")
    BMI = st.text_input("BMI")
    DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function")
    Age = st.text_input("Age")


if __name__ == "__main__":
    main()