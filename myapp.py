import streamlit as st
import nbconvert
from nbconvert import NotebookExporter
import nbformat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
import joblib
from sklearn.ensemble import RandomForestClassifier


st.sidebar.title('Overshoot Prediction Parameters')
st.sidebar.markdown('This application is used to predict overshoot of given parameters')
loaded_rf = joblib.load("Overshooting_Predictor.joblib")


def user_input_features():
    area_owner_ali = 0
    area_owner_chawki = 0
    area_owner_emna = 0
    area_owner_nabila = 0
    area_owner_naoufel = 0
    area_owner_zied = 0
    covered_area_type_low_densely_population = 0
    covered_area_type_medium_density_urban_areas = 0
    covered_area_type_overcrowded_area = 0
    covered_area_type_rural_area = 0

    isd_km = st.sidebar.slider('ISD(KM)', 0.0, 100.0, 0.0)
    km_15_4 = st.sidebar.slider('15.4 KM <> 16 KM', 0.0, 100.0, 0.0)
    km_16_0 = st.sidebar.slider('16 KM <> 16.5 KM', 0.0, 100.0, 0.0)
    km_16_5 = st.sidebar.slider('16.5 KM <> 17.6 KM', 0.0, 100.0, 0.0)
    km_17_6 = st.sidebar.slider('17.6 KM <> 18.7 KM', 0.0, 100.0, 0.0)
   
    percentage_overshooting = st.sidebar.slider('Percentage of Overshooting', 0.0, 100.0, 0.0)
    actual_tilt_value = st.sidebar.slider('Actual Tilt value', 0.0, 100.0, 0.0)
    target_tilt_value = st.sidebar.slider('Target Tilt value', 0.0, 100.0, 0.0)
    area_owner = st.sidebar.radio('Area Owner', ['Ali', 'Chawki', 'Emna', 'Nabila', 'Naoufel', 'Zied'])
    covered_area_type = st.sidebar.radio('Covered Area Type / Morphologhy', ['Low Densely Population', 'Medium-Density Urban Areas', 'Overcrowded Area', 'Rural Area'])
    if area_owner == 'Ali':
        area_owner_ali = 1
    elif area_owner == 'Chawki':
        area_owner_chawki = 1
    elif area_owner == 'Emna':
        area_owner_emna = 1
    elif area_owner == 'Nabila':
        area_owner_nabila = 1
    elif area_owner == 'Naoufel':
        area_owner_naoufel = 1
    elif area_owner == 'Zied':
        area_owner_zied = 1
    
    if covered_area_type == 'Low Densely Population':
        covered_area_type_low_densely_population = 1
    elif covered_area_type == 'Medium-Density Urban Areas':
        covered_area_type_medium_density_urban_areas = 1
    elif covered_area_type == 'Overcrowded Area':
        covered_area_type_overcrowded_area = 1
    elif covered_area_type == 'Rural Area':
        covered_area_type_rural_area = 1

    data = {'ISD(KM)': isd_km,
            '15.4 KM <> 16 KM': km_15_4,
            '16 KM <> 16.5 KM': km_16_0,
            '16.5 KM <> 17.6 KM': km_16_5,
            '17.6 KM <> 18.7 KM': km_17_6,
            
            'Percentage of Overshooting': percentage_overshooting,
            'Actual Tilt value': actual_tilt_value,
            'Target Tilt value': target_tilt_value,
            'Area Owner_Ali': area_owner_ali,
            'Area Owner_Chawki': area_owner_chawki,
            'Area Owner_Emna': area_owner_emna,
            'Area Owner_Nabila': area_owner_nabila,
            'Area Owner_Naoufel': area_owner_naoufel,
            'Area Owner_Zied': area_owner_zied,
            'Covered Area Type / Morphologhy_Low Densely Population': covered_area_type_low_densely_population,
            'Covered Area Type / Morphologhy_Medium-Density Urban Areas': covered_area_type_medium_density_urban_areas,
            'Covered Area Type / Morphologhy_Overcrowded Area': covered_area_type_overcrowded_area,
            'Covered Area Type / Morphologhy_Rural Area': covered_area_type_rural_area}
    features = pd.DataFrame(data, index=[0])
    return features

test_case_data = user_input_features()

st.subheader('User Input parameters')
st.write(test_case_data)

prediction = loaded_rf.predict(test_case_data)


st.title("Overshooting Prediction")
if prediction[0] == 1:
    st.error("Overshooting Detected!")
else:
    st.success("Not Overshooting")