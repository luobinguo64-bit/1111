import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import catboost as cb 


loaded = joblib.load('CatBoost.pkl')
model = loaded["model"] 

st.markdown("<h3 style='font-size:28px'>Early Non-curative Recurrence Predictor after Liver Resection</h3>", unsafe_allow_html=True)

st.markdown("<div style='margin-top:30px'></div>", unsafe_allow_html=True)


AFP = st.sidebar.number_input("Alpha-fetoprotein (AFP, ng/mL):", min_value=0.0, max_value=100000.0, value=20.0)  
Tumor_diameter = st.sidebar.number_input("Tumor Size (cm):", min_value=0.1, max_value=25.0, value=3.0)  
MVI = st.sidebar.selectbox("Microvascular Invasion (MVI):", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)') 
ALBI = st.sidebar.number_input("ALBI Score:", min_value=-4.0, max_value=0.0, value=-2.6)  
Liver_cirrhosis = st.sidebar.selectbox("Liver Cirrhosis:", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)') 
PLT = st.sidebar.number_input("Platelet Count (PLT, ×10^9/mL):", min_value=30, max_value=600, value=150) 

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
    st.markdown("<div style='margin-top:30px'></div>", unsafe_allow_html=True)
    
    st.markdown(
        f"<p style='font-family:Source Serif; font-size:18px; font-weight:bold;'>Predicted possibility of early non-curative recurrence is {probability:.1f}%</p>",
        unsafe_allow_html=True
    )

    if probability >= 20:
        st.markdown(
            "<p style='font-family:Source Serif; font-size:18px;'>According to our model, you are at high risk of early non-curative recurrence. This type of recurrence is typically associated with limited eligibility for curative treatments and poorer post-recurrence outcomes. Early multidisciplinary evaluation and consideration of intensified surveillance or adjuvant strategies may be warranted.</p>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<p style='font-family:Source Serif; font-size:18px;'>According to our model, you are at low risk of early non-curative recurrence. Patients in this group are more likely to remain eligible for curative treatment options if recurrence occurs. Standard postoperative surveillance is recommended.</p>",
            unsafe_allow_html=True
        )
        
    st.markdown("<div style='margin-top:40px'></div>", unsafe_allow_html=True)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)

    shap_display_df = input_df.rename(columns={
        "Liver_cirrhosis": "Liver cirrhosis",
        "Tumor_diameter": "Tumor size"
    })

    plt.figure(figsize=(14, 30), dpi=150)  
    plt.rcParams.update({'font.size': 20})  
    shap.force_plot(
        explainer.expected_value,
        shap_values[0],
        shap_display_df.iloc[0],  
        matplotlib=True,
        show=False
    )

    st.pyplot(plt.gcf())
    plt.close()
