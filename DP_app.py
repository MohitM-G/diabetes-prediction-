import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.set_page_config(page_title="Diabetes Prediction App", layout="wide")

st.title("""Diabetes Prediction using SVM""")

st.sidebar.title("📘 About")

with st.sidebar.expander("🔍 How It Works"):
    st.sidebar.write("""
    This app uses a trained **Machine Learning model** that analyzes input features such as:
    - Glucose Level
    - Blood Pressure
    - Age
    - Insulin Levels
    - DiabetesPedigreeFunction
    
    The model was trained on the **Pima Indian Diabetes Dataset**, a well-known dataset in the medical ML community.
    """)

with st.sidebar.expander("🎯 Objective"):
    st.write("""
    The main goals of this project are:
    - To provide an early, non-invasive prediction of diabetes risk.
    - To showcase how ML can be applied in healthcare for real-time decision support.
    - To demonstrate the integration of predictive modeling with interactive web apps using Streamlit.
    """)

with st.sidebar.expander("🛠️ Technologies Used"):
    st.write("""
    - **Python**
    - **Scikit-learn**
    - **Pandas & NumPy**
    - **Matplotlib & Seaborn**
    - **Streamlit**
    """)

with st.sidebar.expander("📊 Dataset Information"):
    st.write("""
    The app uses the **Pima Indian Diabetes Dataset**, which contains 768 samples of female patients aged 21 or older.
    Each record includes:
    - Glucose
    - Blood Pressure
    - Insulin
    - Diabetes Pedigree Function
    - Age
    - Outcome (0 or 1)
    """)



# Load local CSV file directly
csv_file = "diabetes.csv"
if not os.path.exists(csv_file):
    st.error(f"'{csv_file}' not found in the app directory.")
    st.stop()

## Load dataset
data = pd.read_csv(csv_file).drop(['Pregnancies', 'SkinThickness', 'BMI'], axis=1)
st.subheader("Dataset Preview")
st.dataframe(data.head())


with st.expander("Show Correlation Heatmap"):
    fig, ax = plt.subplots()
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# Data Preprocessing
X = data.drop(columns='Outcome', axis=1)
Y = data['Outcome']
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Model Training
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

# Evaluation
Y_pred = classifier.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)

st.subheader("Model Evaluation")
st.write(f"Accuracy Score: {accuracy:.2f}")
st.write("Confusion Matrix:")
st.write(confusion_matrix(Y_test, Y_pred))
st.write("Classification Report:")
st.text(classification_report(Y_test, Y_pred))

# Prediction Interface
st.subheader("Make a Prediction")
user_input = []
feature_names = data.columns[:-1]

with st.form("prediction_form"):
    for feature in feature_names:
        val = st.number_input(f"{feature}", min_value=0.0)
        user_input.append(val)
    predict_btn = st.form_submit_button("Predict")

if predict_btn:
    input_array = scaler.transform([user_input])
    prediction = classifier.predict(input_array)
    result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"
    st.success(f"The model predicts: {result}")
