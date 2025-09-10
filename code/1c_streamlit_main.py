import streamlit as st
import pandas as pd
import requests

fastapi_url = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Obesity Level Predictor", layout="centered")

st.title(" BMI-Based Obesity Level Predictor")
st.markdown("Enter the personal and lifestyle details below to predict the obesity level.")

with st.form("obesity_form"):
    st.header("Demographics & Family History")
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
    with col2:
        age_input = st.number_input("Age (years)", min_value=1.0, max_value=120.0, value=20.0, step=1.0)
    col3, col4 = st.columns(2)
    with col3:
        height = st.number_input("Height (meters)", min_value=0.5, max_value=2.5, value=1.70, step=0.01)
    with col4:
        weight = st.number_input("Weight (kilograms)", min_value=10.0, max_value=200.0, value=70.0, step=0.1)

    family_history_with_overweight = st.radio("Family History with Overweight", ["yes", "no"])

    st.header("Eating Habits")
    col5, col6 = st.columns(2)
    with col5:
        favc = st.radio("Frequent Consumption of High Caloric Food", ["yes", "no"])
    with col6:
        fcvc = st.slider("Frequency of Consumption of Vegetables (1=Never, 3=Always)", 1.0, 3.0, 2.0, step=1.0)

    col7, col8 = st.columns(2)
    with col7:
        ncp = st.slider("Number of Main Meals (1=1 meal, 4=4+ meals)", 1.0, 4.0, 3.0, step=1.0)
    with col8:
        caec = st.selectbox("Consumption of Food Between Meals", ["no", "Sometimes", "Frequently", "Always"])
    
    col9, col10 = st.columns(2)
    with col9:
        ch2o = st.slider("Consumption of Water Daily (1=Low, 3=High)", 1.0, 3.0, 2.0, step=1.0)
    with col10:
        calc = st.selectbox("Consumption of Alcohol", ["no", "Sometimes", "Frequently"])
    
    st.header("Activity & Technology")
    col11, col12 = st.columns(2)
    with col11:
        smoke = st.radio("Smoker", ["yes", "no"])
    with col12:
        scc = st.radio("Calories Consumption Monitoring", ["yes", "no"])
    
    col13, col14 = st.columns(2)
    with col13:
        faf = st.slider("Physical Activity Frequency (0=None, 3=High)", 0.0, 3.0, 1.0, step=1.0)
    with col14:
        tue = st.slider("Time Using Technology Devices (0=None, 2=High)", 0.0, 3.0, 1.0, step=1.0)
    
    mtrans = st.selectbox("Transportation Mode", ["Automobile", "Public_Transportation", "Walking", "Bike", "Motorbike"])

    submitted = st.form_submit_button("Predict Obesity Level")

    if submitted:
        input_data = {
            "Gender": gender,
            "Age": age_input,
            "Height": height,
            "Weight": weight,
            "family_history_with_overweight": family_history_with_overweight,
            "FAVC": favc,
            "FCVC": fcvc,
            "NCP": ncp,
            "CAEC": caec,
            "SMOKE": smoke,
            "CH2O": ch2o,
            "SCC": scc,
            "FAF": faf,
            "TUE": tue,
            "CALC": calc,
            "MTRANS": mtrans
        }

        try:
            # Make the POST request to the FastAPI /predict endpoint
            response = requests.post(fastapi_url, json=input_data)
            response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)

            prediction_result = response.json()
            predicted_label = prediction_result.get("predicted_obesity_level", "Unknown")

            st.success(f"Predicted Obesity Level: **{predicted_label}**")

        except requests.exceptions.ConnectionError:
            st.error("Connection Error: Could not connect to the FastAPI service. Please ensure the FastAPI server is running and accessible.")
        except requests.exceptions.HTTPError as http_err:
            st.error(f"HTTP error occurred: {http_err} - {response.json().get('detail', 'No details provided')}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

