import streamlit as st
import pandas as pd
import pickle

st.markdown("""
<style>
.main {
    background-color: #0E1117;
}
.block-container {
    padding-top: 2rem;
}
h1, h2, h3 {
    color: #00ADB5;
}
.stButton>button {
    background-color: #00ADB5;
    color: white;
    border-radius: 10px;
    height: 3em;
}
</style>
""", unsafe_allow_html=True)

# Load model
model = pickle.load(open("pipeline.pkl", "rb"))

st.set_page_config(page_title="Churn Predictor", layout="centered")

st.title("📊 Customer Churn Prediction")
st.markdown("Fill details to predict churn")

st.divider()

# -------------------------------
# BASIC INFO
# -------------------------------
st.subheader("👤 Customer Info")

col1, col2 = st.columns(2)

with col1:
    Gender = st.selectbox("Gender", ["Male", "Female"])
    SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
    Partner = st.selectbox("Partner", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["Yes", "No"])

with col2:
    Tenure = st.slider("Tenure (months)", 0, 72, 12)
    MonthlyCharges = st.slider("Monthly Charges", 0.0, 150.0, 50.0)
    TotalCharges = st.number_input("Total Charges", 0.0, 10000.0, 2000.0)

# -------------------------------
# SERVICES
# -------------------------------
st.subheader("📡 Services")

PhoneService = st.selectbox("Phone Service", ["Yes", "No"])

MultipleLines = st.selectbox(
    "Multiple Lines",
    ["No", "Yes", "No phone service"]
)

InternetService = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

OnlineSecurity = st.selectbox(
    "Online Security",
    ["No", "Yes", "No internet service"]
)

OnlineBackup = st.selectbox(
    "Online Backup",
    ["No", "Yes", "No internet service"]
)

DeviceProtection = st.selectbox(
    "Device Protection",
    ["No", "Yes", "No internet service"]
)

TechSupport = st.selectbox(
    "Tech Support",
    ["No", "Yes", "No internet service"]
)

StreamingTV = st.selectbox(
    "Streaming TV",
    ["No", "Yes", "No internet service"]
)

StreamingMovies = st.selectbox(
    "Streaming Movies",
    ["No", "Yes", "No internet service"]
)

# -------------------------------
# BILLING
# -------------------------------
st.subheader("💳 Billing")

Contract = st.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])

PaymentMethod = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

# -------------------------------
# VALIDATION
# -------------------------------
def validate():
    if TotalCharges < MonthlyCharges * Tenure * 0.5:
        st.warning("⚠️ TotalCharges seems too low")
        return False
    return True

st.divider()

# -------------------------------
# PREDICT
# -------------------------------
if st.button("🔍 Predict", use_container_width=True):

    if validate():

        input_df = pd.DataFrame({
            "Gender": [Gender],
            "SeniorCitizen": [SeniorCitizen],
            "Partner": [Partner],
            "Dependents": [Dependents],
            "Tenure": [Tenure],
            "PhoneService": [PhoneService],
            "MultipleLines": [MultipleLines],
            "InternetService": [InternetService],
            "OnlineSecurity": [OnlineSecurity],
            "OnlineBackup": [OnlineBackup],
            "DeviceProtection": [DeviceProtection],
            "TechSupport": [TechSupport],
            "StreamingTV": [StreamingTV],
            "StreamingMovies": [StreamingMovies],
            "Contract": [Contract],
            "PaperlessBilling": [PaperlessBilling],
            "PaymentMethod": [PaymentMethod],
            "MonthlyCharges": [MonthlyCharges],
            "TotalCharges": [TotalCharges]
        })

        prediction = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

        st.subheader("📈 Result")

        if prediction == 1:
            st.error(f"⚠️ Customer likely to churn (Prob: {prob:.2f})")
        else:
            st.success(f"✅ Customer will stay (Prob: {prob:.2f})")

        st.progress(float(prob))

        

import pandas as pd
import matplotlib.pyplot as plt

if st.checkbox("Show Feature Importance"):

    model_rf = model.named_steps["model"]
    preprocessor = model.named_steps["preprocessor"]

    # Get feature names
    ohe = preprocessor.named_transformers_["cat"]
    cat_features = ohe.get_feature_names_out()

    num_features = preprocessor.transformers_[0][2]

    all_features = list(num_features) + list(cat_features)

    importances = model_rf.feature_importances_

    # Create dataframe
    feat_df = pd.DataFrame({
        "Feature": all_features,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)
    feat_df = feat_df.head(10)

    st.bar_chart(feat_df.set_index("Feature"))

    