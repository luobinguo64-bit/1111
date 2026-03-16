import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

st.markdown(
    """
    <style>
    html, body, [class*="css"]  {
        font-family: 'serif';
    }
    </style>
    """,
    unsafe_allow_html=True
)

model = joblib.load("Catboost.pkl")

feature_order = [
    "AFP",
    "Tumor_diameter",
    "MVI",
    "ALBI",
    "Liver_cirrhosis",
    "PLT"
]

# 特征范围
feature_ranges = {
    "AFP": {"type": "numerical", "min": 0.0, "max": 1000000, "default": 20},
    "Tumor_diameter": {"type": "numerical", "min": 0, "max": 25, "default": 5},
    "MVI": {"type": "categorical", "options": [0, 1]},
    "ALBI": {"type": "numerical", "min": -4.0, "max": 0, "default": -2.6},
    "Liver_cirrhosis": {"type": "categorical", "options": [0, 1]},
    "PLT": {"type": "numerical", "min": 30, "max": 600, "default": 150},
}

# =====================
# Streamlit 页面
# =====================
st.title("Prediction Model with SHAP Visualization")
st.header("Enter feature values")

feature_values = {}

for feature in feature_order:
    properties = feature_ranges[feature]

    if properties["type"] == "numerical":
        value = st.number_input(
            label=f"{feature} ({properties['min']} - {properties['max']})",
            min_value=float(properties["min"]),
            max_value=float(properties["max"]),
            value=float(properties["default"]),
        )
    else:
        value = st.selectbox(
            label=f"{feature}",
            options=properties["options"],
        )
    feature_values[feature] = value

# =====================
# 预测与 SHAP
# =====================
if st.button("Predict"):
    input_df = pd.DataFrame([feature_values])[feature_order]

    # 预测概率
    predicted_proba = model.predict_proba(input_df)[0][1]
    probability = predicted_proba * 100
    st.subheader(f"Predicted possibility of Non-curative recurrence: {probability:.2f}%")

    # SHAP 解释
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)

    # 设置 matplotlib 使用 Source Serif
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
