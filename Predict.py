import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

# =====================
# 页面自定义文字样式
# =====================
def custom_text(text, size=18, bold=False, font_family="serif"):
    """
    显示自定义字体文字
    - text: 要显示的文字
    - size: 字号
    - bold: 是否加粗
    - font_family: 字体，例如 'Source Serif Pro Semibold'
    """
    weight = "bold" if bold else "normal"
    st.markdown(
        f'<span style="font-family:\'{font_family}\'; font-size:{size}px; font-weight:{weight}">{text}</span>',
        unsafe_allow_html=True
    )

# =====================
# 模型加载
# =====================
model = joblib.load("Catboost.pkl")

feature_order = ["AFP", "Tumor_diameter", "MVI", "ALBI", "Liver_cirrhosis", "PLT"]
feature_ranges = {
    "AFP": {"type": "numerical", "min": 0.0, "max": 1000000, "default": 20},
    "Tumor_diameter": {"type": "numerical", "min": 0, "max": 25, "default": 5},
    "MVI": {"type": "categorical", "options": [0, 1]},
    "ALBI": {"type": "numerical", "min": -4.0, "max": 0, "default": -2.6},
    "Liver_cirrhosis": {"type": "categorical", "options": [0, 1]},
    "PLT": {"type": "numerical", "min": 30, "max": 600, "default": 150},
}

# =====================
# 页面 UI
# =====================
# 大标题使用 Source Serif Pro Semibold
custom_text("Prediction Model with SHAP Visualization", size=36, bold=True, font_family="Source Serif Pro Semibold")

feature_values = {}
for feature in feature_order:
    props = feature_ranges[feature]
    if props["type"] == "numerical":
        value = st.number_input(
            label=f"{feature} ({props['min']} - {props['max']})",
            min_value=float(props["min"]),
            max_value=float(props["max"]),
            value=float(props["default"]),
        )
    else:
        value = st.selectbox(label=f"{feature}", options=props["options"])
    feature_values[feature] = value

# =====================
# 预测 & SHAP
# =====================
if st.button("Predict"):
    input_df = pd.DataFrame([feature_values])[feature_order]
    predicted_proba = model.predict_proba(input_df)[0][1]

    # 使用自定义字体显示预测结果
    custom_text(
        f"Predicted possibility of Non-curative recurrence: {predicted_proba*100:.2f}%",
        size=20,
        bold=True,
        font_family="Source Serif Pro Semibold"
    )

    # SHAP 解释
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)

    # Matplotlib 图表字体
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
