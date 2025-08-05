
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
    pass


if __name__ == "__main__":
    main()