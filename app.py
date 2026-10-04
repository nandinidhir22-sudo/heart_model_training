from pathlib import Path

import pickle
import numpy as np
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "heart_disease_models.pkl"


@st.cache_resource
def load_models():
    return pickle.load(open(MODEL_PATH, "rb"))


st.title("Heart Disease Prediction")
st.write("Enter the feature values to get model predictions.")
st.caption(
    "This is an educational model demonstration, not a medical diagnosis."
)

if not MODEL_PATH.exists():
    st.error("Model file not found. Run heart.py first to create it.")
    st.stop()

bundle = load_models()
logistic_model = bundle["logistic_model"]
linear_model = bundle["linear_model"]
FEATURE_COLUMNS = bundle["feature_columns"]


with st.form("heart_prediction_form"):
    age = st.number_input("Age", min_value=18, max_value=100, value=55)
    sex = st.selectbox("Sex", ["M", "F"])
    chest_pain = st.selectbox(
        "Chest pain type", ["ASY", "ATA", "NAP", "TA"]
    )
    resting_bp = st.number_input(
        "Resting blood pressure", min_value=0, max_value=250, value=130
    )
    cholesterol = st.number_input(
        "Cholesterol", min_value=0, max_value=700, value=200
    )
    fasting_bs = st.selectbox("Fasting blood sugar", [0, 1])
    resting_ecg = st.selectbox(
        "Resting ECG", ["Normal", "ST", "LVH"]
    )
    max_hr = st.number_input(
        "Maximum heart rate", min_value=40, max_value=250, value=140
    )
    exercise_angina = st.selectbox("Exercise angina", ["N", "Y"])
    oldpeak = st.number_input(
        "Oldpeak", min_value=-5.0, max_value=10.0, value=0.0, step=0.1
    )
    st_slope = st.selectbox("ST slope", ["Up", "Flat", "Down"])

    submitted = st.form_submit_button("Predict")


if submitted:
    input_values = {
        "Age": age,
        "Sex": sex,
        "ChestPainType": chest_pain,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "RestingECG": resting_ecg,
        "MaxHR": max_hr,
        "ExerciseAngina": exercise_angina,
        "Oldpeak": oldpeak,
        "ST_Slope": st_slope,
    }

    # Preserve the names and order used when training.
    input_df = pd.DataFrame([
        {column: input_values[column] for column in FEATURE_COLUMNS}
    ])

    # Apply the same zero-value rule used during training.
    for column in ["RestingBP", "Cholesterol"]:
        if input_df.loc[0, column] == 0:
            input_df[column] = np.nan

    # Logistic Regression is the proper classifier for this 0/1 target.
    logistic_prediction = int(logistic_model.predict(input_df)[0])

    st.subheader("Logistic Regression")
    if logistic_prediction == 1:
        st.error("Prediction: Heart disease")
    else:
        st.success("Prediction: No heart disease")

    # Linear Regression is shown only as a thresholded comparison.
    linear_score = float(linear_model.predict(input_df)[0])
    linear_prediction = int(linear_score >= 0.5)

    st.subheader("Linear Regression comparison")
    st.write(f"Raw model score: {linear_score:.3f}")
    if linear_prediction == 1:
        st.write("Prediction at 0.5 cutoff: Heart disease")
    else:
        st.write("Prediction at 0.5 cutoff: No heart disease")