
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
from pathlib import Path
from datetime import datetime

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Message Intelligence System",
    page_icon="💬",
    layout="wide"
)

# ---------------- NUDE THEME ----------------

st.markdown("""
<style>

/* Main Background */
.stApp {
    background-color: #FAF7F2;
    color: #594638;
}

/* Headings */
h1, h2, h3 {
    color: #6F5748 !important;
}

/* Normal Text */
p, label, span {
    color: #594638;
}

/* Message Input and Text Area */
.stTextArea textarea {
    background-color: #F0E5D8 !important;
    color: #4B382D !important;
    border: 2px solid #BFA58A !important;
    border-radius: 12px !important;
    font-size: 16px !important;
    caret-color: #6F5748 !important;
}

/* Placeholder Color */
.stTextArea textarea::placeholder {
    color: #8C715D !important;
    opacity: 1 !important;
}

/* Input Focus */
.stTextArea textarea:focus {
    background-color: #F0E5D8 !important;
    border: 2px solid #8C715D !important;
    box-shadow: 0 0 5px #D6C3AD !important;
}

/* Number Input */
.stNumberInput input {
    background-color: #F0E5D8 !important;
    color: #4B382D !important;
    border: 1px solid #BFA58A !important;
    border-radius: 8px !important;
}

/* Select Box */
.stSelectbox div[data-baseweb="select"] {
    background-color: #F0E5D8;
    border-radius: 8px;
}

/* Buttons */
.stButton > button {
    background-color: #8C715D;
    color: white !important;
    border-radius: 10px;
    border: none;
    padding: 10px 25px;
    font-weight: 600;
}

.stButton > button:hover {
    background-color: #6F5748;
    color: white !important;
    border: none;
}

/* Metrics */
div[data-testid="stMetric"] {
    background-color: #E8DCCB;
    padding: 15px;
    border-radius: 12px;
}

/* Divider */
hr {
    border-color: #D6C3AD;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD SAVED MODELS ----------------

BASE_DIR = Path(__file__).resolve().parent

possible_paths = [
    BASE_DIR / "saved_models",
    BASE_DIR / "analysis" / "saved_models"
]

MODEL_DIR = next(
    (path for path in possible_paths if path.exists()),
    None
)

if MODEL_DIR is None:
    st.error("Saved models folder not found. Please check your project folder.")
    st.stop()

try:
    knn_model = joblib.load(MODEL_DIR / "knn_model.pkl")
    svm_linear = joblib.load(MODEL_DIR / "svm_linear.pkl")
    svm_rbf = joblib.load(MODEL_DIR / "svm_rbf.pkl")
    nb_model = joblib.load(MODEL_DIR / "naive_bayes_model.pkl")

    imputer = joblib.load(MODEL_DIR / "imputer.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    feature_columns = joblib.load(MODEL_DIR / "feature_columns.pkl")

except Exception as e:
    st.error("Error loading saved models: " + str(e))
    st.stop()

# ---------------- KEYWORDS ----------------

spam_keywords = [
    "free", "win", "winner", "prize", "cash",
    "urgent", "claim", "offer", "bonus", "click",
    "reward", "limited", "loan", "lottery", "discount"
]

legit_keywords = [
    "hello", "thanks", "meeting", "project", "please",
    "tomorrow", "schedule", "team", "update", "regards",
    "good", "morning", "work", "office"
]

# ---------------- FEATURE EXTRACTION ----------------

def extract_features(message, activity, account_age, messages_sent):

    current_time = datetime.now()
    words = message.split()

    spam_score = sum(
        word.lower().strip(".,!?") in spam_keywords
        for word in words
    )

    legit_score = sum(
        word.lower().strip(".,!?") in legit_keywords
        for word in words
    )

    feature_values = {
        "message_length": len(message),
        "word_count": len(words),
        "num_urls": len(re.findall(r'https?://|www\.', message)),
        "num_digits": sum(char.isdigit() for char in message),
        "num_special_chars": sum(
            not char.isalnum() and not char.isspace()
            for char in message
        ),
        "spam_keyword_score": spam_score,
        "legit_keyword_score": legit_score,
        "sender_activity_score": activity,
        "sender_account_age_days": account_age,
        "messages_sent_last_24h": messages_sent,
        "hour_of_day": current_time.hour,
        "day_of_week": current_time.weekday()
    }

    return pd.DataFrame(
        [[feature_values[col] for col in feature_columns]],
        columns=feature_columns
    )

# ---------------- HEADER ----------------

st.title("💬 Message Intelligence System")

st.markdown("""
### Intelligent Spam & Legitimate Message Detection

Analyze messages using Machine Learning classifiers.
""")

st.divider()

# ---------------- SIDEBAR ----------------

st.sidebar.title("⚙️ Model Settings")

model_choice = st.sidebar.selectbox(
    "Select Classification Model",
    ["KNN", "Linear SVM", "RBF SVM", "Naive Bayes"]
)

model_dict = {
    "KNN": knn_model,
    "Linear SVM": svm_linear,
    "RBF SVM": svm_rbf,
    "Naive Bayes": nb_model
}

selected_model = model_dict[model_choice]

st.sidebar.markdown("---")
st.sidebar.caption("Message Intelligence System | ML Project")

# ---------------- MESSAGE INPUT ----------------

st.subheader("📝 Enter Your Message")

message = st.text_area(
    "Message Content",
    placeholder="Type or paste your message here...",
    height=170,
    key="message_input"
)

# ---------------- SENDER INFORMATION ----------------

st.subheader("👤 Sender Information")

col1, col2, col3 = st.columns(3)

with col1:
    activity = st.number_input(
        "Sender Activity Score",
        min_value=0.0,
        max_value=1.0,
        value=0.5
    )

with col2:
    account_age = st.number_input(
        "Account Age (Days)",
        min_value=0,
        max_value=10000,
        value=365
    )

with col3:
    messages_sent = st.number_input(
        "Messages Sent in Last 24 Hours",
        min_value=0,
        max_value=10000,
        value=5
    )

st.caption(
    "Enter sender details if available. Default values are example inputs."
)

# ---------------- PREDICTION ----------------

if st.button("🔍 Analyze Message", use_container_width=True):

    if not message.strip():
        st.warning("Please enter a message first.")

    else:
        input_data = extract_features(
            message,
            activity,
            account_age,
            messages_sent
        )

        imputed_data = imputer.transform(input_data)
        scaled_data = scaler.transform(imputed_data)

        prediction = selected_model.predict(scaled_data)[0]

        st.divider()
        st.subheader("📊 Analysis Result")

        if prediction == 1:
            st.error("🚨 Prediction: SPAM MESSAGE")
        else:
            st.success("✅ Prediction: LEGITIMATE MESSAGE")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Selected Model", model_choice)

        with col2:
            st.metric("Message Length", len(message))

        # Probability Display
        if hasattr(selected_model, "predict_proba"):

            probabilities = selected_model.predict_proba(scaled_data)[0]

            legitimate_probability = probabilities[0] * 100
            spam_probability = probabilities[1] * 100

            st.markdown("### 📈 Prediction Probabilities")

            p1, p2 = st.columns(2)

            with p1:
                st.metric(
                    "Legitimate",
                    f"{legitimate_probability:.2f}%"
                )
                st.progress(float(probabilities[0]))

            with p2:
                st.metric(
                    "Spam",
                    f"{spam_probability:.2f}%"
                )
                st.progress(float(probabilities[1]))

        else:
            decision_score = selected_model.decision_function(
                scaled_data
            )[0]

            st.info(
                f"SVM Decision Score: {decision_score:.4f}. "
                "This is not a calibrated probability."
            )

        with st.expander("View Extracted Message Features"):
            st.dataframe(input_data, use_container_width=True)

# ---------------- FOOTER ----------------

st.divider()

st.markdown("""
<div style="
    text-align:center;
    color:#8C715D;
    padding:15px;
">
    <h4 style="color:#6F5748;">Message Intelligence System</h4>
    Machine Learning Project | Spam Detection
</div>
""", unsafe_allow_html=True)
