import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

font_path = "fonts/times.ttf"
font_prop = fm.FontProperties(fname=font_path)
plt.rcParams["font.family"] = font_prop.get_name()

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

# 页面标题
st.title("Prediction Model with SHAP Visualization")

st.header("Enter feature values")

feature_values = {}

# 输入组件
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


# 预测按钮
if st.button("Predict"):

    # 转换为 DataFrame（保证特征顺序正确）
    input_df = pd.DataFrame([feature_values])[feature_order]

    # 预测概率
    predicted_proba = model.predict_proba(input_df)[0][1]

    probability = predicted_proba * 100

    st.subheader(
        f"Predicted possibility of Non-curative recurrence: {probability:.2f}%"
    )

    # =====================
    # SHAP 解释
    # =====================
  # 计算 SHAP 值
input_df = pd.DataFrame([feature_values], columns=feature_ranges.keys())

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(input_df)

# 生成 SHAP 力图
plt.figure()
plt.rcParams["font.family"] = font_prop.get_name()

shap.force_plot(
    explainer.expected_value,
    shap_values,
    input_df,
    matplotlib=True
)

# 保存并显示
plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
st.image("shap_force_plot.png")
