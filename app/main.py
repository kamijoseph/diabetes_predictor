
# streamlit application
import streamlit as st
import pandas as pd
import pickle
import numpy as np

model = pickle.load(open("notebook/model.sav", "rb"))
scaler = pickle.load(open("notebook/scaler.sav", "rb"))