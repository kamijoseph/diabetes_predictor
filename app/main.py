
# streamlit application
import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os

# model and scaler load
model_path = os.path.join(os.path.dirname(__file__), "../notebook/model.sav")
scaler_path = os.path.join(os.path.dirname(__file__), "../notebook/scaler.sav")

# loading the model
with open(model_path, "rb") as model_file:
    model = pickle.load(model_file)

# loading the scaler
with open(scaler_path, "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# model = pickle.load(open("notebook/model.sav", "rb"))
# scaler = pickle.load(open("notebook/scaler.sav", "rb"))

# predictive system
def prediction_system(input_data):

    # converting to array
    input_array = np.asarray(input_data)

    # reshaping the input array
    input_reshaped = input_array.reshape(1, -1)

    # scaling the reshaped input
    input_scaled = scaler.transform(input_reshaped)

    # model prediction
    prediction = model.predict(input_scaled)[0]

    # class probabilities
    probabilities = model.predict_proba(input_scaled)[0]

    # get results and class probabilities and return them
    result = "DIABETIC!, Seek medical help." if prediction == 1 else "NOT DIABETIC, but still seek medical diagnosis."
    prob_dict = {
        "Not Diabetic": round(probabilities[0] * 100, 2),
        "Diabetic": round(probabilities[1] * 100, 2)
    }
    return result, prob_dict

# main execute function
def main():

    # title
    st.title("Diabetes Prediction Model")
    st.write("This project is to be used by medical professional in making a diagnosis but is never a substitute for professional and proper medical diagnosis")
    st.write("The prediction model works by taking different inputs through the slider on the left or by manually filling in the values in the prompt. Press get results when you are done.")

    # Sidebar sliders
    st.sidebar.header("Adjust parameters with sliders")

    preg_slider = st.sidebar.slider("Pregnancies", 0, 20, value=3, step=1)
    glu_slider = st.sidebar.slider("Glucose", 0, 200, value=128, step=1)
    bp_slider = st.sidebar.slider("Blood Pressure", 0, 130, value=62, step=1)
    skin_slider = st.sidebar.slider("Skin Thickness", 0, 100, value=24, step=1)
    insulin_slider = st.sidebar.slider("Insulin", 0, 300, value=52, step=1)
    bmi_slider = st.sidebar.slider("BMI", 0.0, 70.0, value=30.9, step=0.1)
    dpf_slider = st.sidebar.slider("Diabetes Pedigree Function", 0.0, 3.0, value=0.82, step=0.01)
    age_slider = st.sidebar.slider("Age", 10, 100, value=33, step=1)

    # getting input data from the user
    Pregnancies = st.text_input("Number of Pregnancies", value=str(preg_slider))
    Glucose = st.text_input("Glucose Level", value=str(glu_slider))
    BloodPressure = st.text_input("Blood Pressure", value=str(bp_slider))
    SkinThickness = st.text_input("Skin Thickness", value=str(skin_slider))
    Insulin = st.text_input("Insulin Level", value=str(insulin_slider))
    BMI = st.text_input("BMI", value=str(bmi_slider))
    DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function", value=str(dpf_slider))
    Age = st.text_input("Age", value=str(age_slider))

    # Final values: if user inputs text, use it; else fallback to slider
    def get_final_value(text_value, slider_value, is_float=False):
        try:
            return float(text_value) if is_float else int(text_value)
        except:
            return slider_value

    final_input = [
        get_final_value(Pregnancies, preg_slider),
        get_final_value(Glucose, glu_slider),
        get_final_value(BloodPressure, bp_slider),
        get_final_value(SkinThickness, skin_slider),
        get_final_value(Insulin, insulin_slider),
        get_final_value(BMI, bmi_slider, is_float=True),
        get_final_value(DiabetesPedigreeFunction, dpf_slider, is_float=True),
        get_final_value(Age, age_slider),
    ]

    # rendering prediction
    diagnosis = ""

    # get results and output probabilities
    if st.button("Get Results"):
        result, prob_dict = prediction_system(final_input)
        st.success(result)
        for cls, prob in prob_dict.items():
            st.write(f"- **{cls}**: {prob}%")
    
    st.success(diagnosis)


if __name__ == "__main__":
    main()