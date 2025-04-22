import joblib
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler

# Inisialisasi class untuk prediksi
class LoanXGBoostModelInference:
    def __init__(self, model_path, scaler_path, columns_path, encoders_path):
        # Memuat model, scaler, dan struktur kolom
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.columns = joblib.load(columns_path)
        self.encoders = joblib.load(encoders_path)  # Memuat encoder yang telah dilatih
        
    def preprocess_new_data(self, new_data):
        # Preprocessing data baru
        features = new_data
        
        # Pastikan 'person_gender' dan 'previous_loan_defaults_on_file' dalam format yang sesuai
        features['person_gender'] = features['person_gender'].str.lower()
        features['person_gender'] = features['person_gender'].replace('fe male', 'female')

        # Imputasi missing value: Mengisi nilai kosong pada 'person_income' dengan median
        median_income = features['person_income'].median()  # Hitung median dari kolom person_income
        features['person_income'].fillna(median_income, inplace=True)  # Isi nilai kosong dengan median

        # Scaling numerik
        numeric_cols = features.select_dtypes(include=['int64', 'float64']).columns
        features[numeric_cols] = self.scaler.transform(features[numeric_cols])

        # Encoding binary kategorikal
        label_cols = ['person_gender', 'previous_loan_defaults_on_file']
        for col in label_cols:
            encoder = self.encoders[col]  # Ambil encoder yang sudah dilatih untuk kolom tersebut
            features[col] = encoder.transform(features[col])

        # One-hot encoding untuk kolom multikategori
        one_hot_cols = ['person_education', 'loan_intent', 'person_home_ownership']
        features = pd.get_dummies(features, columns=one_hot_cols, drop_first=True)

        # Samakan struktur kolom
        features, _ = features.align(pd.DataFrame(columns=self.columns), join='left', axis=1, fill_value=0)
        
        return features
    
    def predict(self, new_data):
        # Prediksi status pinjaman
        processed_data = self.preprocess_new_data(new_data)
        prediction = self.model.predict(processed_data)
        return prediction


