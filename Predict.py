import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

# 自定义 HTML 标签
def custom_label(text, size=18, bold=False):
    weight = "bold" if bold else "normal"
    return f'<span style="font-family:serif; font-size:{size}px; font-weight:{weight}">{text}</span>'

# 模型加载
model = joblib.load("Catboost.pkl")

feature_order = ["AFP", "Tumor_diameter", "MVI", "ALBI", "Liver_cirrhosis", "PLT"]

# 显示名称（可以加括号、单位等）
feature_display_names = {
    "AFP": "Alpha-fetoprotein (AFP, ng/mL)",
    "Tumor_diameter": "Tumor Diameter (cm)",
    "MVI": "Microvascular Invasion (0=No, 1=Yes)",
    "ALBI": "ALBI Score",
    "Liver_cirrhosis": "Liver Cirrhosis (0=No, 1=Yes)",
    "PLT": "Platelet Count (10^9/L)"
}

feature_ranges = {
    "AFP": {"type": "numerical", "default": 20},
    "Tumor_diameter": {"type": "numerical", "default": 5},
    "MVI": {"type": "categorical", "options": [0, 1]},
    "ALBI": {"type": "numerical", "default": -2.6},
    "Liver_cirrhosis": {"type": "categorical", "options": [0, 1]},
    "PLT": {"type": "numerical", "default": 150},
}

# 页面标题
st.markdown(custom_label("Prediction Model with SHAP Visualization", size=30, bold=True), unsafe_allow_html=True)

feature_values = {}
for feature in feature_order:
    props = feature_ranges[feature]
    display_name = feature_display_names[feature]

    # 列布局，让标签和输入框靠得近
    col1, col2 = st.columns([1, 3])
    col1.markdown(custom_label(display_name, size=18, bold=True), unsafe_allow_html=True)

    if props["type"] == "numerical":
        value = col2.number_input(
            label="",  # label 为空，不显示默认文本
            value=float(props.get("default", 0)),
            key=feature
        )
    else:
        value = col2.selectbox(
            label="",  # label 为空
            options=props["options"],
            key=feature
        )
    feature_values[feature] = value

# 预测按钮
if st.button("Predict"):
    input_df = pd.DataFrame([feature_values])[feature_order]
    predicted_proba = model.predict_proba(input_df)[0][1]
    st.markdown(custom_label(f"Predicted possibility of Non-curative recurrence: {predicted_proba*100:.2f}%", size=20, bold=True), unsafe_allow_html=True)

    # SHAP 解释
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)

    # Matplotlib 图表字体
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["axes.unicode_minus"] = False

    plt.figure(figsize=(8, 4))
    shap.force_plot(
        explainer.expected_value,
        shap_values[0],
        input_df.iloc[0],
        matplotlib=True
    )

    st.pyplot(plt.gcf())
