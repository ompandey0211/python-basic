
import streamlit as st
import joblib

model = joblib.load("car_price_prediction.pkl")

st.title("Car Price Prediction")
