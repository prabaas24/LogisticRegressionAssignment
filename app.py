import streamlit as st
import numpy as np
import pickle
import pathlib

# Load model
MODEL_PATH = pathlib.Path(__file__).parent / "model.pkl"
logr = pickle.load(open(MODEL_PATH, "rb"))

st.title("Titanic Survival Prediction")
st.write("Enter passenger details:")

# Inputs matching training columns
pclass = st.selectbox("Pclass", [1, 2, 3])
age = st.number_input("Age", min_value=0, max_value=100, value=30)
sibsp = st.number_input("Siblings/Spouses Aboard", 0, 10, 0)
parch = st.number_input("Parents/Children Aboard", 0, 10, 0)
fare = st.number_input("Fare", 0.0, 600.0, 30.0)

# Sex one-hot
sex = st.selectbox("Sex", ["male", "female"])
sex_female = 1 if sex == "female" else 0
sex_male = 1 if sex == "male" else 0

# Embarked one-hot
embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])
embarked_C = 1 if embarked == "C" else 0
embarked_Q = 1 if embarked == "Q" else 0
embarked_S = 1 if embarked == "S" else 0

# Arrange input in EXACT FEATURE ORDER
input_data = np.array([[ 
    pclass,
    sex_female, sex_male,
    age,
    sibsp,
    parch,
    fare,
    embarked_C, embarked_Q, embarked_S
]])

if st.button("Predict"):
    pred = logr.predict(input_data)[0]
    if pred == 1:
        st.success("Survived ✔")
    else:
        st.error("Did NOT survive ✘")
