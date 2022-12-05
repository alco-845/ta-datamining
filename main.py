import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.validation import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score

# Note :
# 1. Menambahkan checkbox di dalam preprocessing yang berisi normalisasi ataupun kategorial, kategorial to numerik
# 2. Menambahkan checkbox di dalam modelling yang berisi model yang pernah digunakan
# 3. Save model di dalam modelling dan digunakan lagi di implementasi

st.write("""
# Project UAS DATA MINING
Oleh : Nicola Yanni Alivant | 200411100146
""")

importData, preprocessing, modeling, implementation = st.tabs(["Import Data", "Prepocessing", "Modeling", "Implementation"])

def loadData():
    data = pd.read_csv("https://raw.githubusercontent.com/alco-845/ta-datamining/master/Banking%20Analyze.csv")
    return data

data = loadData()

with importData:
    st.write("""
    # Deskripsi
    Data yang digunakan adalah data banking yang bisa didapatkan di link berikut :
    https://www.kaggle.com/code/berkinkaplanolu/banking-analyze/data?select=train.csv

    Data ini digunakan untuk melakukan prediksi apakah nasabah akan melakukan deposito secara berkala 
    atau tidak dikarenakan sebuah bank sedang melakukan sebuah kampanye atau event dan terdapat sales yang 
    mempromosikannya dengan cara mengontak nasabah / calon nasabah.

    Kolom numerik yang akan digunakan yaitu :
    - age : umur nasabah.
    - balance : jumlah saldo nasabah.
    - duration : durasi telpon terakhir nasabah ketika di telpon oleh sales bank (menggunakan satuan detik).
    - campaign : jumlah kontak sales yang menghubungi nasabah selama kampanye ini
    - pdays : jumlah hari terakhir nasabah di kontak oleh sales (Jika belum di kontak, maka nilainya adalah -1).
    - previous : jumlah kontak sales yang menghubungi nasabah sebelum kampanye ini dimulai.

    Kolom default memiliki arti apakah nasabah memiliki kartu kredit atau tidak.
    """)

    st.markdown("---")
    st.write("# Import Data")    
    st.write(data)

with preprocessing:
    st.write("# Preprocessing")
    normalisasi = st.checkbox("Normalisasi data dengan MinMaxScallar")
    encoding = st.checkbox("Encoding (Kategorial ke Numerik)")

    st.markdown("---")

    if normalisasi:
        st.write("## Normalisasi")
        st.write("Melakukan Normalisasi pada semua fitur dan mengambil fitur yang memiliki tipe data numerik")
        data_baru = data.drop(columns=["ID", "job", "contact", "day", "month", "poutcome", "subscribed"])

        sebelum_dinormalisasi = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']
        setelah_dinormalisasi = ['Norm_age', 'Norm_balance', 'Norm_duration', 'Norm_campaign', 'Norm_pdays', 'Norm_previous']

        normalisasi_fitur = data[sebelum_dinormalisasi]
        st.dataframe(normalisasi_fitur)

        scaler = MinMaxScaler()
        scaler.fit(normalisasi_fitur)
        fitur_ternormalisasi = scaler.transform(normalisasi_fitur)
        
        # save normalisasi
        joblib.dump(scaler, 'normal')

        fitur_ternormalisasi_df = pd.DataFrame(fitur_ternormalisasi, columns = setelah_dinormalisasi)

        st.markdown("---")
        st.write("## Data yang telah dinormalisasi")
        st.write("Fitur numerikal sudah dinormalisasi")
        st.dataframe(fitur_ternormalisasi)        
        st.markdown("---")

    if encoding:
        st.write("## Encoding (Kategorial ke Numerik)")
        st.write("Mengubah fitur dengan tipe data kategorial menjadi numerik")

        data.loc[(data['marital'] == "divorced") ,"marital"] = 2
        data.loc[(data['marital'] == "married") ,"marital"] = 1
        data.loc[(data['marital'] == "single"),"marital"] = 0

        data.loc[(data['education'] == "tertiary"),"education"] = 2
        data.loc[(data['education'] == "unknown") ,"education"] = 1
        data.loc[(data['education'] == "secondary") ,"education"] = 1
        data.loc[(data['education'] == "primary"),"education"] = 0

        data['default'] = data['default'].replace({'yes': 1, 'no': 0})
        data['housing'] = data['housing'].replace({'yes': 1, 'no': 0}) 
        data['loan'] = data['loan'].replace({'yes': 1, 'no': 0})

        data_baru = data.drop(columns=["ID", "age", "job", "balance", "contact", "day", "month", "duration", "campaign", "pdays", "previous", "subscribed"])

        split_overdue = pd.get_dummies(data_baru["poutcome"], prefix="poutcome")
        data_baru = data_baru.join(split_overdue)
        data_baru = data_baru.drop(columns = "poutcome")
        
        st.dataframe(data_baru)
        st.markdown("---")
    
    if normalisasi and encoding:
        st.write("## Data yang sudah diNormaliasi dan Encoding")
        data_sudah_normal = data_baru
        
        data_sudah_normal = data_sudah_normal.join(fitur_ternormalisasi_df)

        st.write("Hasil data yang sudah dinormalisasi dan diencoding disatukan dalam satu frame")
        st.dataframe(data_sudah_normal)

with modeling:
    st.write("# Modeling")

    st.write("Sistem ini menggunakan 3 modeling yaitu KNN, Naive-Bayes, dan Decission Tree")
    knn_cekbox = st.checkbox("KNN")
    bayes_gaussian_cekbox = st.checkbox("Naive-Bayes Gaussian")
    decission3_cekbox = st.checkbox("Decission Tree")

    #=========================== Spliting data ======================================
    X = data_sudah_normal.iloc[:,8:14]
    Y = data.iloc[:,-1]

    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.25, random_state=0)    

    #============================ Model =================================
    #===================== KNN =======================
    knn = KNeighborsClassifier()
    knn.fit(X_train, Y_train)
    y_predknn = knn.predict(X_test)
    knn_accuracy = round(100 * accuracy_score(Y_test, y_predknn), 2)

    #===================== Bayes Gaussian =============
    gaussian = GaussianNB()
    gaussian.fit(X_train,Y_train)
    y_pred_gaussian   =  gaussian.predict(X_test)
    gauss_accuracy  = round(100*accuracy_score(Y_test, y_pred_gaussian),2)
    gaussian_eval = classification_report(Y_test, y_pred_gaussian,output_dict = True)
    gaussian_eval_df = pd.DataFrame(gaussian_eval).transpose()

    #===================== Decission tree =============
    decission3  = DecisionTreeClassifier(criterion="gini")
    decission3.fit(X_train,Y_train)
    y_pred_decission3 = decission3.predict(X_test)
    decission3_accuracy = round(100*accuracy_score(Y_test, y_pred_decission3),2)
    decission3_eval = classification_report(Y_test, y_pred_decission3,output_dict = True)
    decission3_eval_df = pd.DataFrame(decission3_eval).transpose()

    st.markdown("---")

    #===================== Cek Box ====================
    if knn_cekbox:
        st.write("##### KNN")
        st.warning("Dengan menggunakan metode KNN didapatkan akurasi sebesar:")
        # st.warning(knn_accuracy)
        st.warning(f"Akurasi  =  {knn_accuracy}%")
        st.markdown("---")

    if bayes_gaussian_cekbox:
        st.write("##### Naive Bayes Gausssian")
        st.info("Dengan menggunakan metode Bayes Gaussian didapatkan hasil akurasi sebesar:")
        st.info(f"Akurasi = {gauss_accuracy}%")
        st.markdown("---")

    if decission3_cekbox:
        st.write("##### Decission Tree")
        st.success("Dengan menggunakan metode Decission tree didapatkan hasil akurasi sebesar:")
        st.success(f"Akurasi = {decission3_accuracy}%")

with implementation:
    st.write("# Implementation")
    st.write("##### Input fitur")
    name = st.text_input("Masukkan nama anda")
    age = st.number_input("Masukkan umur anda saat ini")
    balance = st.number_input("Masukkan saldo anda saat ini")
    duration = st.number_input("Masukkan jumlah durasi telpon terakhir anda ketika di telpon oleh sales (menggunakan satuan detik)")
    campaign = st.number_input("Masukkan jumlah kontak sales yang menghubungi anda selama kampanye ini")    
    pdays = st.number_input("Masukkan jumlah hari terakhir anda di kontak oleh sales (Jika belum di kontak, silahkan masukan -1)", min_value=-1.00)
    previous = st.number_input("Masukkan jumlah kontak sales yang menghubungi anda sebelum kampanye ini dimulai")    

    cek_hasil = st.button("Cek Prediksi")

    knn = joblib.load("import/knn.joblib")
    decission3 = joblib.load("import/decission3.joblib")
    gaussian = joblib.load("import/gaussian.joblib")
    # scaler = joblib.load("import/scaler.joblib") 

    #============================ Mengambil akurasi tertinggi ===========================
    if knn_accuracy > gauss_accuracy and knn_accuracy > decission3_accuracy:
        use_model = knn
        metode = "KNN"
    elif gauss_accuracy > knn_accuracy and gauss_accuracy > decission3_accuracy:
        use_model = gaussian
        metode = "Naive-Bayes Gaussian"
    else:
        use_model = decission3
        metode = "Decission Tree"

    #============================ Normalisasi inputan =============================
    inputan = [[age, balance, duration, campaign, pdays, previous]]
    inputan_norm = scaler.transform(inputan)
    # inputan
    # inputan_norm
    if cek_hasil:
        hasil_prediksi = use_model.predict(inputan_norm)[0]
        if hasil_prediksi == "yes":
            st.success(f"{name} berpotensi untuk melakukan deposito secara berkala, berdasarkan metode {metode}")
        else:
            st.error(f"{name} tidak berpotensi untuk melakukan deposito secara berkala, berdasarkan metode {metode}")