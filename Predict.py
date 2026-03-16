import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import catboost as cb  # 确保使用 CatBoost 官方类

# Load the trained model
model = joblib.load('Catboost.pkl')  # 如果报错，推荐使用 CatBoost 的 model.save_model/.load_model

# Streamlit UI
st.markdown("<h3 style='font-size:28px'>Early Non-curative Recurrence Predictor</h3>", unsafe_allow_html=True)# 早期非_CURATIVE复发预测器

st.markdown("<div style='margin-top:50px'></div>", unsafe_allow_html=True)

# Sidebar for input options
st.sidebar.header("Input Sample Data")

AFP = st.sidebar.number_input("Alpha-fetoprotein (AFP, ng/mL):", min_value=0.0, max_value=100000.0, value=20.0)  
Tumor_diameter = st.sidebar.number_input("Tumor Diameter (cm):", min_value=0.1, max_value=25.0, value=3.0)  
MVI = st.sidebar.selectbox("Microvascular Invasion (MVI):", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)') 
ALBI = st.sidebar.number_input("ALBI Score:", min_value=-4.0, max_value=0.0, value=-2.6)  
Liver_cirrhosis = st.sidebar.selectbox("Liver Cirrhosis:", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)') 
PLT = st.sidebar.number_input("Platelet Count (PLT, ×10^9/mL):", min_value=30, max_value=600, value=150) 

# Create a DataFrame for SHAP and prediction
input_dict = {
    "AFP": [AFP],
    "Tumor_diameter": [Tumor_diameter],
    "MVI": [MVI],
    "ALBI": [ALBI],
    "Liver_cirrhosis": [Liver_cirrhosis],
    "PLT": [PLT]
}
input_df = pd.DataFrame(input_dict)

if st.button("Make Prediction"):
    # Predict probabilities
    predicted_proba = model.predict_proba(input_df)[0]

    # Assume class 1 = early non-curative recurrence
    probability = predicted_proba[1] * 100

    # Display single line
    st.markdown("<div style='margin-top:50px'></div>", unsafe_allow_html=True)
    
    st.markdown(
        f"<p style='font-family:Source Serif; font-size:18px; font-weight:bold;'>Predicted possibility of early non-curative recurrence is {probability:.1f}%</p>",
        unsafe_allow_html=True
    )

    if probability >= 50:
        st.markdown(
            "<p style='font-family:Source Serif; font-size:18px;'>According to our model, your risk of early non-curative recurrence is high. Please consult a specialist for further evaluation.</p>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<p style='font-family:Source Serif; font-size:18px;'>According to our model, your risk of early non-curative recurrence is low. Maintain regular check-ups and a healthy lifestyle.</p>",
            unsafe_allow_html=True
        )
        
    st.markdown("<div style='margin-top:50px'></div>", unsafe_allow_html=True)

    # SHAP 解释
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)

    # 绘制 force_plot
    plt.figure(figsize=(14, 30), dpi=150)  # 增大竖向高度
    plt.rcParams.update({'font.size': 20})  # 字体大一些
    shap.force_plot(
    explainer.expected_value,
    shap_values[0],
    input_df.iloc[0],
    matplotlib=True,
    show=False
    )

    # 在 Streamlit 显示
    st.pyplot(plt.gcf())
    plt.close()
