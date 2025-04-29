import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import time

# Load model and encoders
model = joblib.load("random_forest_optimized_model.joblib")
encoders = joblib.load("label_encoders.joblib")

# Load base64-encoded video
with open("video_base64.txt", "r") as f:
    video_base64 = f.read()

# Streamlit page config
st.set_page_config(page_title="CVD Risk Predictor", layout="centered")

# ===== Custom CSS =====
st.markdown("""
    <style>
        body {
            background-color: #fafafa;
        }
        .section-card {
            background-color: #f9f9fc;
            padding: 20px;
            border-radius: 15px;
            border: 1px solid #e6e6e6;
            box-shadow: 2px 2px 5px rgba(200,200,200,0.2);
            margin-bottom: 25px;
        }
        .risk-box {
            background-color: #fffaf0;
            padding: 20px;
            border-left: 8px solid #FF8C00;
            border-radius: 10px;
            margin-top: 20px;
        }
        .sticky-info {
            position: fixed;
            top: 100px;
            right: 20px;
            width: 250px;
            background-color: #ffffffcc;
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #ccc;
            box-shadow: 1px 1px 8px rgba(0,0,0,0.1);
            z-index: 999;
            font-size: 13px;
        }
        .sticky-info h4 {
            margin-top: 0;
            font-size: 16px;
        }
    </style>
    <div class='sticky-info'>
        <h4>ℹ️ Tips</h4>
        <ul style="padding-left: 18px;">
            <li>Answer all questions accurately</li>
            <li>Hover on sliders to fine-tune inputs</li>
            <li>Your data is not stored or shared</li>
            <li>Prediction = lifestyle + health data</li>
        </ul>
    </div>
""", unsafe_allow_html=True)

# ===== App Header =====
st.markdown("<h1 style='text-align: center;'>💓 Cardiovascular Disease Risk Predictor</h1>", unsafe_allow_html=True)

# ===== Embedded Video via Base64 =====
st.markdown(f"""
    <div style="text-align: center;">
        <video autoplay loop muted playsinline width="70%">
            <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
            Your browser does not support the video tag.
        </video>
    </div>
""", unsafe_allow_html=True)

# ===== App Intro Text =====
st.markdown("""
<div style='text-align: center; font-size: 16px;'>
Welcome to the <b>Cardiovascular Disease (CVD) Risk Predictor</b>.<br>
Fill in the details below to estimate your risk level based on your lifestyle and health indicators.
</div><br>
""", unsafe_allow_html=True)

# === Section 1: Personal & Lifestyle Information ===
with st.container():
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("### 👤 Personal & Lifestyle Information")

    col1, col2 = st.columns(2)
    with col1:
        sex = st.selectbox("🧍 Sex", ["Female", "Male"])
        age_category = st.selectbox("🎂 Age Category", [
            "18-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54",
            "55-59", "60-64", "65-69", "70-74", "75-79", "80+"
        ])
        height = st.number_input("📏 Height (cm)", min_value=90, max_value=250, value=170)
        weight = st.number_input("⚖️ Weight (kg)", min_value=20, max_value=300, value=70)

    with col2:
        smoking = st.selectbox("🚬 Smoking History", ["No", "Yes"])
        depression = st.selectbox("🧠 Do you have depression?", ["No", "Yes"])
        exercise = st.selectbox("🏃‍♂️ Do you exercise regularly?", ["No", "Yes"])
        alcohol = st.slider("🍷 Alcohol Consumption (days/month)", 0, 30, 0)

    st.markdown("</div>", unsafe_allow_html=True)

# === Section 2: Diet & Health Perception ===
with st.container():
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("### 🥗 Diet & Health Perception")

    col3, col4 = st.columns(2)
    with col3:
        general_health = st.selectbox("🩺 General Health", [
            "Excellent", "Very Good", "Good", "Fair", "Poor"
        ])
        checkup = st.selectbox("📋 Last Medical Checkup", [
            "Within the past year",
            "Within the past 2 years",
            "Within the past 5 years",
            "5 or more years ago",
            "Never"
        ])
        diabetes = st.selectbox("🧬 Do you have diabetes?", [
            "No", 
            "Yes", 
            "No, pre-diabetes or borderline diabetes", 
            "Yes, but female told only during pregnancy"
        ])
        arthritis = st.selectbox("🦴 Do you have arthritis?", ["No", "Yes"])

    with col4:
        skin_cancer = st.selectbox("🌞 History of Skin Cancer?", ["No", "Yes"])
        other_cancer = st.selectbox("🏥 History of Other Cancer?", ["No", "Yes"])
        fruit = st.slider("🍎 Fruit Consumption (servings/month)", 0, 130, 30)
        veggies = st.slider("🥦 Green Veg Consumption (servings/month)", 0, 130, 30)
        fries = st.slider("🍟 Fried Potato Consumption (servings/month)", 0, 130, 15)

    st.markdown("</div>", unsafe_allow_html=True)

# === Prediction Button ===
if st.button("🔍 Predict Risk"):

    input_data = pd.DataFrame({
        'Sex': [sex],
        'Age_Category': [age_category],
        'Height_(cm)': [height],
        'Weight_(kg)': [weight],
        'Smoking_History': [smoking],
        'Exercise': [exercise],
        'Depression': [depression],
        'Alcohol_Consumption': [alcohol],
        'General_Health': [general_health],
        'Checkup': [checkup],
        'Fruit_Consumption': [fruit],
        'Green_Vegetables_Consumption': [veggies],
        'FriedPotato_Consumption': [fries],
        'Diabetes': [diabetes],
        'Arthritis': [arthritis],
        'Skin_Cancer': [skin_cancer],
        'Other_Cancer': [other_cancer],
    })

    try:
        # Progress Bar Animation
        progress_text = "🔄 Calculating your risk... Please wait."
        progress_bar = st.progress(0, text=progress_text)
        for percent_complete in range(100):
            time.sleep(0.015)
            progress_bar.progress(percent_complete + 1, text=progress_text)

        # Encode input
        for col in input_data.columns:
            if col in encoders:
                input_data[col] = encoders[col].transform(input_data[col])

        # Prediction
        input_data = input_data[model.feature_names_in_]
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0][1]

        # Determine risk level
        if prediction_proba < 0.4:
            risk_level = "Low"
            risk_color = "#6FCF97"
            tip = "You're doing great! Keep maintaining your healthy habits. 💚"
        elif prediction_proba < 0.7:
            risk_level = "Moderate"
            risk_color = "#F2C94C"
            tip = "Consider improving your diet and staying active. Small changes matter! 💪"
        else:
            risk_level = "High"
            risk_color = "#EB5757"
            tip = "Your risk is high. Please consult your doctor and review your lifestyle. 🩺"

        # Risk Meter Gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(prediction_proba * 100, 1),
            title={'text': "CVD Risk %"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': risk_color},
                'steps': [
                    {'range': [0, 40], 'color': '#D4EFDF'},
                    {'range': [40, 70], 'color': '#FCF3CF'},
                    {'range': [70, 100], 'color': '#F5B7B1'}
                ],
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

        # Display Risk Info
        st.markdown(f"""
            <div class='risk-box'>
                <h3>🩺 Predicted CVD Risk Level: <span style='color:{risk_color}'>{risk_level}</span></h3>
                <p><b>Probability of Heart Disease:</b> {prediction_proba:.2%}</p>
                <p>💡 <i>{tip}</i></p>
            </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error("⚠️ Error during prediction. Please make sure input values are valid and match model requirements.")
        st.exception(e)

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; font-size: 13px;'>Made with ❤️ </div>", unsafe_allow_html=True)
