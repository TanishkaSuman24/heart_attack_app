import streamlit as st
import pandas as pd 
import numpy as np 
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#to track website traffic in Google Analytics
st.markdown("""
    <script async src="https://www.googletagmanager.com/gtag/js?id=G-MZMG2MK4JZ"></script>
    <script>
    window.dataLayer = window.dataLayer || [];
    function gtag(){dataLayer.push(arguments);}
    gtag('js', new Date());

    gtag('config', 'G-MZMG2MK4JZ');
    </script> 
""", unsafe_allow_html=True)


df = pd.read_csv("heart.csv")
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

#Predictions:
y_prediction = model.predict(X_test)

print("R^2 Value: ", r2_score(y_test, y_prediction))
  
#since we didn't yet launch our app, we will not be in session state
#once we launch, it will go to our home page which is 'form'
if "page" not in st.session_state:  
    st.session_state.page = "form"

if st.session_state.page == "form":
    st.image("heart_image.jpg",  use_container_width=True)
    st.title("Heart Attack Predictor")

    st.header("This app predicts the likelihood of a heart attack")
    st.write("Enter the patient information below")


    sex_input = st.selectbox('Sex', ["Male", "Female"])
    if sex_input == "Female":
        sex = 0
    else:
        sex = 1

    cp_type_input = st.selectbox('Select the type of chest pain:', 
    ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
    if cp_type_input == "Typical Angina":
        cp_type = 0
    elif cp_type_input == "Atypical Angina":
        cp_type = 1
    elif cp_type_input == "Non-anginal Pain":
        cp_type = 2
    else:
        cp_type = 3


    fbs_type = st.selectbox('Fasting Blood Sugar', ["≤120 mg/dL", "≥120 mg/dL"])
    if fbs_type == "≤120 mg/dL":
        fbs = 0
    else:
        fbs = 1

    recr_type = st.selectbox('Resting Electrocardiographic Results', ["Normal", "Abnormal", "Hypertrophy"])
    if recr_type == "Normal":
        recr = 0
    elif recr_type == "Abnormal":
        recr = 1
    else:
        recr = 2

    eia_tof = st.selectbox('History of Exercise-Induced Angina?', ["Yes", "No"])
    if eia_tof == "Yes":
        eia = 1
    else:
        eia = 0

    slope_type = st.selectbox('ST/HR Slope', ["Upsloping", "Flat", "Downsloping"])
    if slope_type == "Upsloping":
        slope = 0
    elif slope_type == "Flat":
        slope = 1
    else: slope = 2

    numvc = st.selectbox('Number of Major Vessels Colored by Fluroscopy', [0,1,2,3])

    thal_input = st.selectbox('Thalassemia', ["Normal", "Fixed Defect", "Reversible Defect", "Missing Data"])
    if thal_input == "Normal":
        thal = 1
    elif thal_input == "Fixed Defect":
        thal = 2
    elif thal_input == "Reversible Defect":
        thal = 3
    else:
        thal = 0

    age = st.slider('Age', 0, 100)
    restbp = st.slider('Resting Blood Pressure', 0, 300)
    chol = st.slider('Cholesterol Level (in mg/dL)', 0, 300)
    mhr = st.slider('Maximum Heart Rate Achieved', 0, 300)
    oldpeak = st.slider('ST Depression Induced by Exercise', 0.0, 6.0)

    input_data = np.array([sex, cp_type, fbs, recr, eia, age, restbp, chol, mhr, oldpeak, slope, numvc, thal])

    if st.button("Predict Likelihood of Heart Attack"):
        # Store input and switch page
        st.session_state.input_data = input_data
        st.session_state.page = "result"
        st.rerun()

elif st.session_state.page == "result":
    st.header("Prediction Result")

    input_data = st.session_state.input_data
    risk_score = model.predict(input_data.reshape(1,-1))[0] 
    st.write(f'The Predicted Heart Attack Risk Score is: {risk_score:.2f}')

    st.write("For more information on how to mitigate the risk of a heart attack, check out https://www.heart.org/en/health-topics/heart-attack/life-after-a-heart-attack/lifestyle-changes-for-heart-attack-prevention")

    if st.button("Go Back"):
        st.session_state.page = "form"
        st.rerun()



