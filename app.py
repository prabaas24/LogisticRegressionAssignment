import streamlit as st
import pickle
import numpy as np
import pathlib

# Safe model path for Streamlit Cloud + local
MODEL_PATH = pathlib.Path(__file__).parent / "logistic_model.pkl"
logr = pickle.load(open(MODEL_PATH, "rb"))

st.title("Titanic Survival Prediction")
st.write("Enter passenger details to predict survival.")

pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
age = st.number_input("Age", min_value=0, max_value=100, value=30)
fare = st.number_input("Ticket Fare", min_value=0.0, max_value=600.0, value=30.0)
sex = st.selectbox("Sex (0 = female, 1 = male)", [0, 1])
sibsp = st.number_input("Siblings/Spouses Aboard (SibSp)", 0, 10, 0)
parch = st.number_input("Parents/Children Aboard (Parch)", 0, 10, 0)

input_data = np.array([[pclass, sex, age, sibsp, parch, fare]])

if st.button("Predict"):
    prediction = logr.predict(input_data)[0]

    if prediction == 1:
        st.success("Prediction: The passenger would have SURVIVED.")
    else:
        st.error("Prediction: The passenger would NOT have survived.")
