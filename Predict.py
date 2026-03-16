import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

# =====================
# 自定义标签样式函数
# =====================
def custom_label(text, size=18, bold=False, font_family="Source Serif"):
    weight = "bold" if bold else "normal"
    # 加 class="custom-label" 用于 CSS 调整距离
    return f'<div class="custom-label" style="font-family:\'{font_family}\'; font-size:{size}px; font-weight:{weight}">{text}</div>'

# =====================
# 模型加载
# =====================
model = joblib.load("Catboost.pkl")

feature_order = ["AFP", "Tumor_diameter", "MVI", "ALBI", "Liver_cirrhosis", "PLT"]

feature_display_names = {
    "AFP": "Alpha-fetoprotein (AFP, ng/mL)",
    "Tumor_diameter": "Tumor Diameter (cm)",
    "MVI": "Microvascular Invasion",
    "ALBI": "ALBI Score",
    "Liver_cirrhosis": "Liver Cirrhosis",
    "PLT": "Platelet Count"
}

feature_ranges = {
    "AFP": {"type": "numerical", "default": 20},
    "Tumor_diameter": {"type": "numerical", "default": 5},
    "MVI": {"type": "categorical", "options": [0, 1]},
    "ALBI": {"type": "numerical", "default": -2.6},
    "Liver_cirrhosis": {"type": "categorical", "options": [0, 1]},
    "PLT": {"type": "numerical", "default": 150},
}

# =====================
# 页面全局 CSS（调整标签和输入框间距）
# =====================
st.markdown(
    """
    <style>
    div.custom-label {
        margin-bottom: 2px;  /* 标签和输入框紧贴 */
    }
    div.stNumberInput, div.stSelectbox {
        margin-top: 2px;     /* 控件上方间距 */
        margin-bottom: 6px;  /* 控件下方间距 */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =====================
# 页面标题
# =====================
st.markdown(
    '<div style="font-family:\'Source Serif Pro Semibold\'; font-size:32px; font-weight:bold; margin-bottom:10px">'
    'Prediction Model with SHAP Visualization</div>',
    unsafe_allow_html=True
)

# =====================
# 输入特征
# =====================
feature_values = {}
for feature in feature_order:
    props = feature_ranges[feature]
    display_name = feature_display_names[feature]

    # 标签在输入框上方
    st.markdown(custom_label(display_name), unsafe_allow_html=True)

    # 输入框（不显示默认 label，也不显示范围）
    if props["type"] == "numerical":
        value = st.number_input(
            label="",  # 空 label
            value=float(props.get("default", 0)),
            key=feature,
            format="%.2f",
        )
    else:
        value = st.selectbox(
            label="",  # 空 label
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
        custom_label(
            f"Predicted possibility of Non-curative recurrence: {predicted_proba*100:.2f}%",
            size=20,
            bold=True
        ),
        unsafe_allow_html=True
    )

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
