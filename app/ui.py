import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load model and scaler
model = joblib.load('model/diabetes_model.pkl')
scaler = joblib.load('model/scaler.pkl')

st.set_page_config(page_title="Diabetes Prediction App", layout="centered")
st.title("🩺 Diabetes Prediction App")
st.markdown("Enter the patient's health data to check diabetes risk.")

# Input validation
def get_input(label, min_val, max_val, step=1.0, format="%.1f"):
    return st.number_input(label, min_value=float(min_val), max_value=float(max_val), step=float(step), format=format)

preg = get_input("Pregnancies", 0, 20, step=1)
glucose = get_input("Glucose Level", 50, 200)
bp = get_input("Blood Pressure", 40, 140)
skin = get_input("Skin Thickness", 10, 100)
insulin = get_input("Insulin", 10, 900)
bmi = get_input("BMI", 10.0, 60.0, step=0.1)
dpf = get_input("Diabetes Pedigree Function", 0.1, 2.5, step=0.01)
age = get_input("Age", 10, 100, step=1)

if st.button("Predict"):
    input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    result = "✅ Likely Diabetic" if prediction == 1 else "🟢 Not Diabetic"
    st.subheader(f"Result: {result}")
    st.write(f"Probability of diabetes: **{probability * 100:.2f}%**")

    # Visualization
    labels = ['Not Diabetic', 'Diabetic']
    values = model.predict_proba(input_scaled)[0]
    fig, ax = plt.subplots()
    ax.bar(labels, values, color=['green', 'red'])
    ax.set_ylabel('Probability')
    st.pyplot(fig)
