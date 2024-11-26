import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the model and scaler
svm_model = joblib.load('model/svm_model.pkl')
scaler = joblib.load('model/scaler.pkl')

# Initialize a list to store the history of predictions
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# Custom CSS
st.markdown("""
    <style>
        body {
            background-color: #f4f7f6;
            font-family: 'Arial', sans-serif;
        }
        .title {
            color: #2d3e50;
            font-size: 36px;
            text-align: center;
            margin-bottom: 20px;
        }
        .subtitle {
            color: #3a5a5f;
            font-size: 24px;
            text-align: center;
            margin-bottom: 20px;
        }
        .input-section {
            margin: 20px;
        }
        .st.subheader{
            text-align: center;
            font-size: 24px;
        }
        .prediction-output {
            display: flex;
            justify-content: center;
            align-items: center;
            flex-direction: column;
            font-size: 22px;
            font-weight: bold;
            margin-top: 20px;
            color: #2d3e50;
        }
        .stButton button {
            background-color: blue;
            color: white;
            font-size: 16px;
            padding: 10px 20px;
            border: none;
            cursor: pointer;
            border-radius: 5px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: background-color 0.3s;
        }
        .stButton button:hover {
            background-color: #ffffff;
        }
        .feedback {
            font-size: 18px;
            font-weight: bold;
            margin-top: 20px;
            text-align: center;
        }
        .footer {
            text-align: center;
            margin-top: 40px;
            font-size: 14px;
            color: #888;
        }
        .footer a {
            color: #4CAF50;
            text-decoration: none;
        }
    </style>
""", unsafe_allow_html=True)

# Title of the web app
st.markdown('<div class="title">Student Performance System</div>', unsafe_allow_html=True)

# Subtitle
st.markdown('<div class="subtitle">Predict your exam score based on study factors</div>', unsafe_allow_html=True)

# User inputs
hours_studied = st.number_input('Hours Studied', min_value=0, max_value=24, value=10, step=1)
attendance = st.number_input('Attendance (%)', min_value=0, max_value=100, value=85, step=1)
previous_scores = st.number_input('Previous Scores', min_value=0, max_value=100, value=75, step=1)
sleep_hours = st.number_input('Sleep Hours', min_value=0, max_value=24, value=7, step=1)
motivation_level = st.selectbox('Motivation Level', options=['Low', 'Medium', 'High'])

# Map the motivation level
motivation_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
motivation_value = motivation_mapping[motivation_level]

# Predict Button
if st.button('Predict'):
    # Prepare input features for prediction
    input_features = np.array([[hours_studied, attendance, previous_scores, sleep_hours, motivation_value]])

    # Scale the input features
    input_scaled = scaler.transform(input_features)

    # Predict the exam score
    predicted_score = svm_model.predict(input_scaled)

    # Store the current prediction in the history
    prediction_data = {
        'Hours Studied': hours_studied,
        'Attendance (%)': attendance,
        'Previous Scores': previous_scores,
        'Sleep Hours': sleep_hours,
        'Motivation Level': motivation_level,
        'Predicted Score': predicted_score[0]
    }
    st.session_state.prediction_history.append(prediction_data)

    # Display the prediction centered on the page
    st.markdown(f'<div class="prediction-output">Predicted Exam Score: {predicted_score[0]:.2f}</div>', unsafe_allow_html=True)

    # Performance feedback based on the predicted score
    if predicted_score >= 90:
        feedback = 'Excellent performance! Keep up the hard work and consistency.'
    elif predicted_score >= 75:
        feedback = 'Good job! You are doing well, but there’s room for improvement.'
    elif predicted_score >= 50:
        feedback = 'Fair performance. Focus on improving your weak areas.'
    else:
        feedback = 'Poor performance. You need to put in more effort and stay motivated.'
    
    st.markdown(f'<div class="feedback">{feedback}</div>', unsafe_allow_html=True)

# Display prediction history
st.markdown('---')
st.markdown('<div class="subtitle">Prediction History</div>', unsafe_allow_html=True)

if st.session_state.prediction_history:
    # Convert the history to a DataFrame
    history_df = pd.DataFrame(st.session_state.prediction_history)
    st.dataframe(history_df)

# Footer section
st.markdown('---')
st.markdown('<div class="footer">© 2024 Student Performance System. All rights reserved.<br>Created by Vaibhav Ayush and Amol from MIT AOE Alandi</div>', unsafe_allow_html=True)
