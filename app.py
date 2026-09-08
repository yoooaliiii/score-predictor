"""
app.py
Streamlit app: Real-Time Student Exam Score Predictor
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Load the trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="ScorePredictor", page_icon="🎓")

st.title("🎓 ScorePredictor")
st.write("Real-Time Exam Score Prediction based on study habits.")

st.subheader("Enter your details:")

hours_studied = st.slider("Hours studied per day", 0.0, 10.0, 5.0, 0.5)
attendance = st.slider("Attendance percentage", 50.0, 100.0, 75.0, 1.0)
assignments = st.slider("Assignments completed (out of 10)", 0, 10, 5, 1)

if st.button("Predict My Score"):
    input_data = pd.DataFrame({
        "hours_studied": [hours_studied],
        "attendance_percent": [attendance],
        "assignments_completed": [assignments]
    })

    prediction = model.predict(input_data)[0]
    prediction = max(0, min(100, prediction))  # keep within 0-100

    st.success(f"Predicted Final Exam Score: **{prediction:.1f} / 100**")

    # Simple bar chart to visualize
    fig, ax = plt.subplots()
    ax.bar(["Predicted Score"], [prediction], color="skyblue")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Score")
    st.pyplot(fig)

st.markdown("---")
st.caption("Built with scikit-learn + Streamlit | Multiple Linear Regression")


