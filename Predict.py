import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the trained model
model = joblib.load('Catboost.pkl')  # 加载训练好的Catboost模型

# Streamlit UI
st.title("Early Non-curative Recurrence Predictor")  # 早期非_CURATIVE复发预测器

# Sidebar for input options
st.sidebar.header("Input Sample Data")  # 侧边栏输入样本数据

AFP = st.sidebar.number_input("Alpha-fetoprotein (AFP, ng/mL):", min_value=0.0, max_value=100000.0, value=20.0)  
Tumor_diameter = st.sidebar.number_input("Tumor Diameter (cm):", min_value=0.1, max_value=25.0, value=3.0)  
MVI = st.sidebar.selectbox("Microvascular Invasion (MVI):", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)') 
ALBI = st.sidebar.number_input("ALBI Score:", min_value=-4, max_value=0, value=-2.6)  
Liver_cirrhosis = st.sidebar.selectbox("Liver Cirrhosis:", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)') 
PLT = st.sidebar.number_input("Platelet Count (PLT, ×10<sup>9</sup>/mL):", min_value=30, max_value=600, value=150) 

# Process the input and make a prediction
feature_values = [AFP, Tumor_diameter, MVI, ALBI, Liver_cirrhosis, PLT]  # 收集所有输入的特征
features = np.array([feature_values])  # 转换为NumPy数组

if st.button("Make Prediction"):  # 如果点击了预测按钮
    # Predict the class and probabilities
    predicted_class = model.predict(features)[0]  # 预测心脏病类别
    predicted_proba = model.predict_proba(features)[0]  # 预测各类别的概率

    # Display the prediction results
    st.write(f"**Predicted Class:** {predicted_class}")  # 显示预测的类别
    st.write(f"**Prediction Probabilities:** {predicted_proba}")  # 显示各类别的预测概率

    # Generate advice based on the prediction result
    probability = predicted_proba[predicted_class] * 100  # 根据预测类别获取对应的概率，并转化为百分比

    if predicted_class == 1:  # 如果预测为心脏病
        advice = (
            f"According to our model, your risk of heart disease is high. "
            f"The probability of you having heart disease is {probability:.1f}%. "
            "Although this is just a probability estimate, it suggests that you might have a higher risk of heart disease. "
            "I recommend that you contact a cardiologist for further examination and assessment, "
            "to ensure you receive an accurate diagnosis and necessary treatment."
        )  # 如果预测为心脏病，给出相关建议
    else:  # 如果预测为无心脏病
        advice = (
            f"According to our model, your risk of heart disease is low. "
            f"The probability of you not having heart disease is {probability:.1f}%. "
            "Nevertheless, maintaining a healthy lifestyle is still very important. "
            "I suggest that you have regular health check-ups to monitor your heart health, "
            "and seek medical attention if you experience any discomfort."
        )  # 如果预测为无心脏病，给出相关建议

    st.write(advice)  # 显示建议

    # SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)

    plt.rcParams["font.family"] = "Source Serif"
    plt.rcParams["axes.unicode_minus"] = False

    plt.figure(figsize=(8, 4))
    shap.force_plot(
        explainer.expected_value,
        shap_values[0],
        input_df.iloc[0],
        matplotlib=True
    )
    st.pyplot(plt.gcf())
