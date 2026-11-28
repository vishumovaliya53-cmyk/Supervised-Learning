
import streamlit as st
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, r2_score
from xgboost import XGBClassifier, XGBRegressor


# --------------------------------
# PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="Smart Outcome Predictor",
    page_icon="🎓",
    layout="wide"
)


# --------------------------------
# CUSTOM CSS
# --------------------------------

st.markdown("""
<style>

.stApp {
    background-color: #F5EFE6;
}

[data-testid="stHeader"] {
    background-color: #F5EFE6;
}

h1, h2, h3 {
    color: #6F5846;
}

p, label {
    color: #3E342D;
}

.stButton > button {
    background-color: #6F5846;
    color: white;
    border-radius: 10px;
    border: none;
    height: 45px;
    font-weight: bold;
}

.stButton > button:hover {
    background-color: #A68A75;
    color: white;
}

[data-testid="stMetric"] {
    background-color: #E8DCCB;
    padding: 18px;
    border-radius: 12px;
}

div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background-color: #FFFFFF;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------
# HEADER
# --------------------------------

st.markdown(
    "<h1 style='text-align:center;'>🎓 Smart Outcome Predictor</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center;'>"
    "Student Performance Analysis & Prediction System"
    "</p>",
    unsafe_allow_html=True
)

st.divider()


# --------------------------------
# LOAD DATASET
# --------------------------------

@st.cache_data
def load_data():

    possible_paths = [
        "Dataset/Smart_Outcome_Predictor_Dataset_5200.xlsx",
        "../Dataset/Smart_Outcome_Predictor_Dataset_5200.xlsx",
        "Smart_Outcome_Predictor_Dataset_5200.xlsx"
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return pd.read_excel(path)

    raise FileNotFoundError(
        "Dataset not found. Please check the Dataset folder."
    )


# --------------------------------
# TRAIN MODELS
# --------------------------------

@st.cache_resource
def train_models():

    df = load_data()

    feature_columns = [
        'age',
        'country_region',
        'device_type',
        'education_background',
        'course_level',
        'course_category',
        'week_of_year',
        'sessions',
        'time_spent_hours',
        'videos_watched',
        'quiz_attempts',
        'assignments_submitted',
        'forum_posts',
        'avg_quiz_score',
        'attendance_rate'
    ]

    X = df[feature_columns]

    y_cls = df['completion_status']
    y_reg = df['final_score']

    numeric_features = X.select_dtypes(
        include=['int64', 'float64']
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=['object', 'category', 'bool']
    ).columns.tolist()

    def create_preprocessor():

        numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])

        return ColumnTransformer([
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
        X, y_cls,
        test_size=0.2,
        random_state=42,
        stratify=y_cls
    )

    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X, y_reg,
        test_size=0.2,
        random_state=42
    )

    classification_model = Pipeline([
        ('preprocessor', create_preprocessor()),
        ('model', XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1
        ))
    ])

    regression_model = Pipeline([
        ('preprocessor', create_preprocessor()),
        ('model', XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1
        ))
    ])

    classification_model.fit(X_train_cls, y_train_cls)
    regression_model.fit(X_train_reg, y_train_reg)

    cls_accuracy = accuracy_score(
        y_test_cls,
        classification_model.predict(X_test_cls)
    )

    reg_r2 = r2_score(
        y_test_reg,
        regression_model.predict(X_test_reg)
    )

    return (
        classification_model,
        regression_model,
        feature_columns,
        cls_accuracy,
        reg_r2
    )


# --------------------------------
# MODEL LOADING
# --------------------------------

try:

    with st.spinner("Training prediction models... Please wait."):

        (
            classification_model,
            regression_model,
            feature_columns,
            cls_accuracy,
            reg_r2
        ) = train_models()

    st.success("Prediction models are ready!")

except Exception as e:

    st.error("Unable to load or train models.")
    st.exception(e)
    st.stop()


# --------------------------------
# MODEL PERFORMANCE
# --------------------------------

st.subheader("📊 Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Classification Accuracy",
        f"{cls_accuracy * 100:.2f}%"
    )

with col2:
    st.metric(
        "Regression R² Score",
        f"{reg_r2:.3f}"
    )

st.divider()


# --------------------------------
# STUDENT INPUT FORM
# --------------------------------

st.subheader("📝 Enter Student Details")

df = load_data()

with st.form("student_prediction_form"):

    col1, col2, col3 = st.columns(3)

    with col1:

        age = st.number_input(
            "Student Age",
            min_value=1,
            max_value=100,
            value=20
        )

        country_region = st.selectbox(
            "Country / Region",
            sorted(df['country_region'].dropna().unique())
        )

        device_type = st.selectbox(
            "Device Type",
            sorted(df['device_type'].dropna().unique())
        )

        education_background = st.selectbox(
            "Education Background",
            sorted(df['education_background'].dropna().unique())
        )

        course_level = st.selectbox(
            "Course Level",
            sorted(df['course_level'].dropna().unique())
        )

    with col2:

        course_category = st.selectbox(
            "Course Category",
            sorted(df['course_category'].dropna().unique())
        )

        week_of_year = st.slider(
            "Week of Year",
            1, 53, 20
        )

        sessions = st.number_input(
            "Learning Sessions",
            min_value=0,
            value=10
        )

        time_spent_hours = st.number_input(
            "Time Spent (Hours)",
            min_value=0.0,
            value=10.0
        )

        videos_watched = st.number_input(
            "Videos Watched",
            min_value=0,
            value=10
        )

    with col3:

        quiz_attempts = st.number_input(
            "Quiz Attempts",
            min_value=0,
            value=5
        )

        assignments_submitted = st.number_input(
            "Assignments Submitted",
            min_value=0,
            value=5
        )

        forum_posts = st.number_input(
            "Forum Posts",
            min_value=0,
            value=2
        )

        avg_quiz_score = st.slider(
            "Average Quiz Score",
            0.0, 100.0, 70.0
        )

        attendance_rate = st.slider(
            "Attendance Rate",
            0.0, 100.0, 80.0
        )

    submitted = st.form_submit_button(
        "🚀 Predict Student Outcome",
        use_container_width=True
    )


# --------------------------------
# PREDICTION
# --------------------------------

if submitted:

    student_data = pd.DataFrame([{
        'age': age,
        'country_region': country_region,
        'device_type': device_type,
        'education_background': education_background,
        'course_level': course_level,
        'course_category': course_category,
        'week_of_year': week_of_year,
        'sessions': sessions,
        'time_spent_hours': time_spent_hours,
        'videos_watched': videos_watched,
        'quiz_attempts': quiz_attempts,
        'assignments_submitted': assignments_submitted,
        'forum_posts': forum_posts,
        'avg_quiz_score': avg_quiz_score,
        'attendance_rate': attendance_rate
    }])

    student_data = student_data[feature_columns]

    completion_prediction = classification_model.predict(
        student_data
    )[0]

    final_score_prediction = regression_model.predict(
        student_data
    )[0]

    st.divider()

    st.subheader("🎯 Student Prediction Results")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### Course Completion")

        st.success(
            f"Prediction: {completion_prediction}"
        )

        if completion_prediction == 1:
            st.write("The model predicts that the student will complete the course.")
        else:
            st.warning("The model predicts that the student may not complete the course.")

    with col2:

        st.markdown("### Final Score")

        st.metric(
            "Predicted Final Score",
            f"{final_score_prediction:.2f}"
        )

    st.divider()

    st.subheader("📋 Student Input Summary")

    st.dataframe(
        student_data,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "⬇️ Download Prediction Report",
        data=student_data.assign(
            predicted_completion=completion_prediction,
            predicted_final_score=round(float(final_score_prediction), 2)
        ).to_csv(index=False),
        file_name="Student_Prediction_Report.csv",
        mime="text/csv"
    )


# --------------------------------
# FOOTER
# --------------------------------

st.divider()

st.markdown(
    "<p style='text-align:center; color:#6F5846;'>"
    "🤎 Smart Outcome Predictor | Powered by Machine Learning"
    "</p>",
    unsafe_allow_html=True
)
