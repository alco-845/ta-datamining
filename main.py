import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

st.write("""
# Project UAS DATA MINING
Oleh : Nicola Yanni Alivant | 200411100146
""")

Data, preprocessing, modeling, implementation = st.tabs(["Data", "Prepocessing", "Modeling", "Implementation"])

with Data:
    st.write("# Data")
    uploaded_files = st.file_uploader("Upload file CSV", accept_multiple_files=True)
    for uploaded_file in uploaded_files:
        data = pd.read_csv(uploaded_file)
        st.write("Nama File : ", uploaded_file.name)
        st.dataframe(data)

with preprocessing:
    st.write("# Preprocessing")
    data.head()
    X = data.drop(columns=["campaign", "pdays", "previous", "poutcome", "subscribed"])

    X.head()

    split_overdue_X = pd.get_dummies(X["duration"], prefix="Duration")
    X = X.join(split_overdue_X)

    X = X.drop(columns = "duration")
    
    loan = pd.get_dummies(X["loan"], prefix="Loan")
    X = X.join(loan)

    X = X.drop(columns = "loan")

    st.write("menampilkan dataframe yang duration dan loan sudah di drop")
    st.dataframe(X)

    st.write("# Normalisasi")
    st.write("Normalize feature 'age', 'balance', 'housing'")
    old_normalize_feature_labels = ['age', 'balance', 'housing']
    new_normalized_feature_labels = ['norm_age', 'norm_balance', 'norm_housing']
    normalize_feature = data[old_normalize_feature_labels]

    st.dataframe(normalize_feature)

    scaler = MinMaxScaler()

    scaler.fit(normalize_feature)

    normalized_feature = scaler.transform(normalize_feature)

    normalized_feature_df = pd.DataFrame(normalized_feature, columns = new_normalized_feature_labels)

    st.write("data setelah dinormalisasi")
    st.dataframe(normalized_feature_df)