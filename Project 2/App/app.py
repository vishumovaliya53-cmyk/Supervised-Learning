
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from datetime import date

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. CUSTOM PROFESSIONAL THEME
# ==========================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #0b1020, #17132e, #111827);
    color: #f8fafc;
}

[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #19152f, #0b1020);
    border-right: 1px solid #39305c;
}

[data-testid="stSidebar"] * {
    color: #e9d5ff !important;
}

h1 {
    color: #c4b5fd !important;
    font-size: 38px !important;
    font-weight: 800 !important;
}

h2, h3 {
    color: #e9d5ff !important;
}

p, label {
    color: #d1d5db;
}

.metric-card {
    background: linear-gradient(135deg, #211b40, #302052);
    border: 1px solid #59458c;
    padding: 22px;
    border-radius: 16px;
    margin-bottom: 15px;
}

.metric-title {
    color: #c4b5fd;
    font-size: 14px;
}

.metric-value {
    color: white;
    font-size: 25px;
    font-weight: bold;
    margin-top: 8px;
}

[data-testid="stMetric"] {
    background: linear-gradient(135deg, #211b40, #302052);
    border: 1px solid #59458c;
    padding: 20px;
    border-radius: 15px;
}

[data-testid="stMetricValue"] {
    color: #c4b5fd !important;
}

[data-testid="stForm"] {
    background: #17132e;
    padding: 24px;
    border: 1px solid #49396f;
    border-radius: 18px;
}

.stNumberInput input {
    background-color: #211b40 !important;
    color: white !important;
    border: 1px solid #6854a0 !important;
}

.stSelectbox div[data-baseweb="select"] > div {
    background-color: #211b40;
    color: white;
    border-color: #6854a0;
}

.stDateInput input {
    background-color: #211b40 !important;
    color: white !important;
}

div.stButton > button,
div.stFormSubmitButton > button {
    background: linear-gradient(90deg, #7c3aed, #a855f7);
    color: white;
    border: none;
    border-radius: 12px;
    height: 48px;
    font-weight: bold;
    width: 100%;
}

div.stButton > button:hover,
div.stFormSubmitButton > button:hover {
    background: linear-gradient(90deg, #9333ea, #c084fc);
    color: white;
}

[data-testid="stAlert"] {
    background-color: #211b40;
    border-radius: 12px;
}

hr {
    border-color: #39305c;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. LOAD TRAINED MODEL
# ==========================================

MODEL_PATH = (
    Path(__file__).parent.parent
    / "Model"
    / "best_house_price_model.pkl"
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    bundle = load_model()

    model = bundle["model"]
    model_name = bundle["model_name"]
    feature_columns = bundle["feature_columns"]

except Exception as e:
    st.error("Model loading failed!")
    st.code(str(e))
    st.stop()

# ==========================================
# 4. SIDEBAR NAVIGATION
# ==========================================

st.sidebar.markdown("# 🏡 Property AI")
st.sidebar.caption("Intelligent House Price Estimator")
st.sidebar.divider()

page = st.sidebar.radio(
    "MAIN MENU",
    ["Dashboard", "Predict Price", "About Model"]
)

st.sidebar.divider()
st.sidebar.success("Random Forest Model Loaded")

# ==========================================
# 5. MAIN HEADER
# ==========================================

st.markdown("# 🏡 House Price Prediction")
st.markdown(
    "### Smart Property Valuation Dashboard"
)

st.caption(
    "Machine Learning Based House Price Estimation"
)

st.divider()

# ==========================================
# 6. DASHBOARD PAGE
# ==========================================

if page == "Dashboard":

    st.subheader("Welcome to Your Dashboard")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Selected Model",
            model_name
        )

    with col2:
        st.metric(
            "Total Features",
            len(feature_columns)
        )

    with col3:
        st.metric(
            "Model Category",
            "Regression"
        )

    st.divider()

    st.subheader("Project Overview")

    st.markdown("""
    This application predicts house prices using
    a trained Machine Learning regression model.

    **Main Features:**

    - Property details input
    - House price prediction
    - Random Forest model integration
    - Professional dashboard
    - Prediction history during the current session
    """)

    st.info(
        "Select Predict Price from the sidebar "
        "to estimate a property's price."
    )

    if "history" not in st.session_state:
        st.session_state.history = []

    st.subheader("Recent Predictions")

    if st.session_state.history:
        history_df = pd.DataFrame(
            st.session_state.history
        )
        st.dataframe(
            history_df,
            use_container_width=True
        )
    else:
        st.write("No predictions made yet.")

# ==========================================
# 7. PREDICT PRICE PAGE
# ==========================================

elif page == "Predict Price":

    st.subheader("Enter Property Information")

    st.write(
        "Fill in the property details below "
        "to calculate the estimated house price."
    )

    with st.form("house_prediction_form"):

        col1, col2 = st.columns(2)

        with col1:

            st.markdown("#### 🏠 Property Details")

            property_id = st.number_input(
                "Property ID",
                min_value=1,
                value=1001
            )

            area_sqft = st.number_input(
                "Area (Square Feet)",
                min_value=100,
                value=1200
            )

            bedrooms = st.number_input(
                "Bedrooms",
                min_value=1,
                value=2
            )

            bathrooms = st.number_input(
                "Bathrooms",
                min_value=1,
                value=2
            )

            property_age = st.number_input(
                "Property Age (Years)",
                min_value=0,
                value=5
            )

        with col2:

            st.markdown("#### 📍 Location Details")

            location_score = st.number_input(
                "Location Score",
                min_value=0.0,
                max_value=10.0,
                value=7.0
            )

            distance_city_km = st.number_input(
                "Distance from City (KM)",
                min_value=0.0,
                value=5.0
            )

            crime_rate_index = st.number_input(
                "Crime Rate Index",
                min_value=0.0,
                value=20.0
            )

            near_school = st.selectbox(
                "Near School?",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )

            near_metro = st.selectbox(
                "Near Metro?",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )

        sale_date = st.date_input(
            "Sale Date",
            value=date.today()
        )

        predict_button = st.form_submit_button(
            "✨ Predict House Price"
        )

    # ======================================
    # 8. MAKE PREDICTION
    # ======================================

    if predict_button:

        input_data = {
            "property_id": property_id,
            "sale_date": sale_date.toordinal(),
            "area_sqft": area_sqft,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "location_score": location_score,
            "property_age": property_age,
            "distance_city_km": distance_city_km,
            "near_school": near_school,
            "near_metro": near_metro,
            "crime_rate_index": crime_rate_index
        }

        try:

            input_df = pd.DataFrame([input_data])

            input_df = input_df.reindex(
                columns=feature_columns
            )

            if input_df.isnull().any().any():
                raise ValueError(
                    "Some model features are missing. "
                    "Please check the notebook feature columns."
                )

            prediction = model.predict(input_df)[0]

            if bundle.get("scaled_target", False):
                target_scaler = bundle["target_scaler"]

                prediction = target_scaler.inverse_transform(
                    [[prediction]]
                )[0][0]

            st.divider()

            st.success("Prediction completed successfully!")

            st.markdown(
                "### 🏡 Estimated Property Price"
            )

            st.markdown(
                f"<h1 style='color:#c4b5fd;'>₹ {prediction:,.2f}</h1>",
                unsafe_allow_html=True
            )

            st.caption(
                "This is a machine learning estimate, "
                "not a guaranteed market valuation."
            )

            st.session_state.history.append({
                "Area (sq ft)": area_sqft,
                "Bedrooms": bedrooms,
                "Bathrooms": bathrooms,
                "Estimated Price (₹)": round(float(prediction), 2)
            })

        except Exception as e:

            st.error("Prediction could not be completed.")
            st.code(str(e))

# ==========================================
# 9. ABOUT MODEL PAGE
# ==========================================

elif page == "About Model":

    st.subheader("Machine Learning Model Details")

    st.markdown(
        "### 🌲 Random Forest Regression"
    )

    st.write(
        "Selected Model:", model_name
    )

    st.write(
        "Total Input Features:",
        len(feature_columns)
    )

    st.divider()

    st.subheader("Features Used by the Model")

    for feature in feature_columns:
        st.write("•", feature)

    st.divider()

    st.info(
        "This application uses the trained model "
        "saved from the original House Price Prediction project."
    )

# ==========================================
# 10. FOOTER
# ==========================================

st.divider()

st.markdown(
    "<center style='color:#a78bfa;'>"
    "House Price Prediction | Machine Learning Project"
    "</center>",
    unsafe_allow_html=True
)
