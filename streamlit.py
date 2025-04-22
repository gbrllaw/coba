import streamlit as st
import pandas as pd
from predict import LoanXGBoostModelInference  # Assuming you have the model inference class in loan_model.py

# Inisialisasi model inference
model_inference = LoanXGBoostModelInference(
    model_path='xgb_model.pkl',
    scaler_path='scaler.pkl',
    columns_path='columns.pkl',
    encoders_path='encoders.pkl'
)

# Judul aplikasi
st.title("Loan Status Prediction")

# Deskripsi aplikasi
st.write("This application predicts whether a loan will be approved or rejected based on user data.")

# Input data pengguna
st.subheader("Input Data")

# Form input untuk data pengguna
with st.form(key='input_form'):
    person_age = st.number_input("Age", min_value=18, max_value=100)
    person_gender = st.selectbox("Gender", options=['Male', 'Female'])
    person_education = st.selectbox("Education Level", options=['High School', 'Undergraduate', 'Graduate', 'Postgraduate'])
    person_income = st.number_input("Annual Income (in USD)", min_value=1000)
    person_emp_exp = st.number_input("Years of Experience", min_value=0)
    person_home_ownership = st.selectbox("Home Ownership", options=['Own', 'Rent'])
    loan_amnt = st.number_input("Loan Amount Requested (in USD)", min_value=100)
    loan_intent = st.selectbox("Loan Intent", options=['Business', 'Personal', 'Education'])
    loan_int_rate = st.number_input("Loan Interest Rate", min_value=1.0, max_value=20.0)
    loan_percent_income = st.number_input("Loan Amount as % of Income", min_value=1.0, max_value=100.0)
    cb_person_cred_hist_length = st.number_input("Credit History Length (in years)", min_value=0)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850)
    previous_loan_defaults_on_file = st.selectbox("Previous Loan Defaults on File", options=['Yes', 'No'])

    submit_button = st.form_submit_button(label='Predict Loan Status')

# If the submit button is pressed
if submit_button:
    # Prepare the input data
    user_data = {
        'person_age': person_age,
        'person_gender': person_gender,
        'person_education': person_education,
        'person_income': person_income,
        'person_emp_exp': person_emp_exp,
        'person_home_ownership': person_home_ownership,
        'loan_amnt': loan_amnt,
        'loan_intent': loan_intent,
        'loan_int_rate': loan_int_rate,
        'loan_percent_income': loan_percent_income,
        'cb_person_cred_hist_length': cb_person_cred_hist_length,
        'credit_score': credit_score,
        'previous_loan_defaults_on_file': previous_loan_defaults_on_file
    }

    # Convert input data to DataFrame
    input_df = pd.DataFrame([user_data])

    # Make prediction
    prediction = model_inference.predict(input_df)

    # Show the prediction result
    if prediction[0] == 1:
        st.success("The loan is approved!")
    else:
        st.error("The loan is rejected.")
