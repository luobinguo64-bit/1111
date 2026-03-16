import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

# =====================
# 自定义文字显示函数
# =====================
def custom_label(text, size=18, bold=False, font_family="Source Serif Pro Semibold"):
    weight = "bold" if bold else "normal"
    return f'<span style="font-family:\'{font_family}\'; font-size:{size}px; font-weight:{weight}">{text}</span>'

# =====================
# 模型加载
# =====================
model = joblib.load("Catboost.pkl")

# 内部特征名（不变，用于模型）
feature_order = ["AFP", "Tumor_diameter", "MVI", "ALBI", "Liver_cirrhosis", "PLT"]

# 每个特征对应的显示名称（可以随意修改）
feature_display_names = {
    "AFP": "Alpha-fetoprotein",
    "Tumor_diameter": "Tumor Diameter (cm)",
    "MVI": "Microvascular Invasion",
    "ALBI": "ALBI Score",
    "Liver_cirrhosis": "Liver Cirrhosis",
    "PLT": "Platelet Count"
}

feature_ranges = {
    "AFP": {"type": "numerical", "min": 0.0, "max": 1000000, "default": 20},
    "Tumor_diameter": {"type": "numerical", "min": 0, "max": 25, "default": 5},
    "MVI": {"type": "categorical", "options": [0, 1]},
    "ALBI": {"type": "numerical", "min": -4.0, "max": 0, "default": -2.6},
    "Liver_cirrhosis": {"type": "categorical", "options": [0, 1]},
    "PLT": {"type": "numerical", "min": 30, "max": 600, "default": 150},
}

# =====================
# 页面标题
# =====================
st.markdown(custom_label("Prediction Model with SHAP Visualization", size=30, bold=True), unsafe_allow_html=True)

# =====================
# 输入特征
# =====================
feature_values = {}
for feature in feature_order:
    props = feature_ranges[feature]
    display_name = feature_display_names[feature]

    # 自定义标签显示
    st.markdown(custom_label(display_name, size=20, bold=True), unsafe_allow_html=True)

    # 数值输入框或选项框
    if props["type"] == "numerical":
        value = st.number_input(
            label="",  # label 为空，用自定义 HTML
            min_value=float(props["min"]),
            max_value=float(props["max"]),
            value=float(props["default"]),
            key=feature
        )
    else:
        value = st.selectbox(
            label="",  # label 为空
            options=props["options"],
            key=feature
        )
    feature_values[feature] = value

# =====================
# 预测 & SHAP
# =====================
if st.button("Predict"):
    input_df = pd.DataFrame([feature_values])[feature_order]
    predicted_proba = model.predict_proba(input_df)[0][1]

    st.markdown(
        custom_label(f"Predicted possibility of Non-curative recurrence: {predicted_proba*100:.2f}%", size=20, bold=True),
        unsafe_allow_html=True
    )

    # SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)

    plt.rcParams["font.family"] = "Source Serif Pro Semibold"
    plt.rcParams["axes.unicode_minus"] = False

    plt.figure(figsize=(8, 4))
    shap.force_plot(
        explainer.expected_value,
        shap_values[0],
        input_df.iloc[0],
        matplotlib=True
    )
    st.pyplot(plt.gcf())
