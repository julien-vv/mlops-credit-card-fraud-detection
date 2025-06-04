import streamlit as st
import requests
import json

## HELP ##
# streamlit run src/frontend.py

st.title("Bank Fraud Prediction")

uploaded_file = st.file_uploader("Upload a JSON file containing data to predict", type=["json"])

if uploaded_file is not None:
    # Load JSON file
    input_json = json.load(uploaded_file)

    st.write("Loaded data:")
    st.json(input_json)

    if st.button("Send request to API"):
        # POST request to the Flask API
        url = "http://localhost:5000/predict"
        response = requests.post(url, json=input_json)

        if response.status_code == 200:
            prediction = response.json()["prediction"]
            st.success("Prediction received:")
            st.write(prediction)
        else:
            st.error(f"API error: {response.status_code} - {response.text}")
