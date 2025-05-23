import streamlit as st
import joblib
import pandas as pd

# Set up the app
st.set_page_config(page_title="Simple House Price Predictor", layout="centered")
st.title("Simple House Price Predictor")


# Load your trained model
@st.cache_data
def load_model():
    return joblib.load('models/model_with_pipeline.pkl')

model = load_model()
