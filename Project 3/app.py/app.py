
# =====================================================
# 🏦 CUSTOMER RISK PREDICTION SYSTEM
# Built with Streamlit and Machine Learning
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page Configuration
st.set_page_config(
    page_title="Customer Risk Prediction",
    page_icon="🏦",
    layout="wide"
)

# Professional Theme
st.title("🏦 Customer Risk Prediction System")

st.markdown(
    "### 📊 AI-Based Customer Risk Classification"
)

st.markdown("---")

# Load Saved Model
@st.cache_resource
def load_model():

    package = joblib.load(
        "Final_Output/final_risk_model.pkl"
    )

    return package

package = load_model()

model = package["model"]
model_name = package["model_name"]
feature_columns = package["feature_columns"]
training_data = package["training_data"]

# Model Information
col1, col2 = st.columns(2)

with col1:
    st.metric("Selected Model", model_name)

with col2:
    st.metric("Total Input Features", len(feature_columns))

st.markdown("---")

# Customer Input Section
st.subheader("👤 Enter Customer Information")

customer_data = {}

numeric_columns = training_data.select_dtypes(
    include=np.number
).columns.tolist()

categorical_columns = [
    col for col in feature_columns
    if col not in numeric_columns
]

with st.form("customer_form"):

    st.markdown("#### 📋 Customer Details")

    col1, col2 = st.columns(2)

    for i, column in enumerate(feature_columns):

        current_column = col1 if i % 2 == 0 else col2

        with current_column:

            if column in numeric_columns:

                default_value = training_data[column].median()

                if pd.api.types.is_integer_dtype(
                    training_data[column].dtype
                ):
                    customer_data[column] = st.number_input(
                        column.replace("_", " ").title(),
                        value=int(default_value)
                    )

                else:
                    customer_data[column] = st.number_input(
                        column.replace("_", " ").title(),
                        value=float(default_value),
                        format="%.4f"
                    )

            else:

                options = sorted(
                    training_data[column]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )

                if options:

                    customer_data[column] = st.selectbox(
                        column.replace("_", " ").title(),
                        options
                    )

                else:

                    customer_data[column] = st.text_input(
                        column.replace("_", " ").title()
                    )

    st.markdown("---")

    threshold = st.slider(
        "High Risk Prediction Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05
    )

    predict_button = st.form_submit_button(
        "🔍 Predict Customer Risk",
        use_container_width=True
    )

# Prediction Section
if predict_button:

    input_df = pd.DataFrame(
        [customer_data],
        columns=feature_columns
    )

    probabilities = model.predict_proba(input_df)[0]

    high_risk_probability = probabilities[1]

    if high_risk_probability >= threshold:

        st.error("⚠️ HIGH RISK CUSTOMER")

        st.write(
            "The customer is classified as High Risk."
        )

    else:

        st.success("✅ LOW RISK CUSTOMER")

        st.write(
            "The customer is classified as Low Risk."
        )

    st.markdown("---")

    st.subheader("📊 Prediction Results")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Low Risk Probability",
            f"{probabilities[0] * 100:.2f}%"
        )

    with col2:

        st.metric(
            "High Risk Probability",
            f"{probabilities[1] * 100:.2f}%"
        )

    st.progress(float(high_risk_probability))

    st.caption(
        "This prediction is a decision-support estimate, "
        "not a guarantee of actual customer risk."
    )

st.markdown("---")

st.caption(
    "Customer Risk Prediction System | "
    "Machine Learning Project 3"
)

