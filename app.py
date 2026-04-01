import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Churn Dashboard", layout="wide")

# -------------------------------
# CUSTOM CSS (SaaS STYLE)
# -------------------------------
st.markdown("""
<style>
body {
    background-color: #0E1117;
}
.block-container {
    padding-top: 1rem;
}
h1, h2, h3 {
    color: #00ADB5;
}
.card {
    background-color: #1c1f26;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 0px 10px rgba(0,0,0,0.5);
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# LOAD MODEL
# -------------------------------
model = pickle.load(open("pipeline.pkl", "rb"))

# -------------------------------
# SIDEBAR NAVIGATION
# -------------------------------
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Dashboard", "🔮 Prediction", "📈 Insights"])

# -------------------------------
# SIDEBAR INPUTS
# -------------------------------
st.sidebar.markdown("---")
st.sidebar.title("⚙️ Inputs")

Gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.sidebar.selectbox("Senior Citizen", [0, 1])
Partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
Dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])

Tenure = st.sidebar.slider("Tenure", 0, 72, 12)
MonthlyCharges = st.sidebar.slider("Monthly Charges", 0.0, 150.0, 50.0)
TotalCharges = st.sidebar.number_input("Total Charges", 0.0, 10000.0, 2000.0)

PhoneService = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
MultipleLines = st.sidebar.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])

InternetService = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

OnlineSecurity = st.sidebar.selectbox("Online Security", ["No", "Yes", "No internet service"])
OnlineBackup = st.sidebar.selectbox("Online Backup", ["No", "Yes", "No internet service"])
DeviceProtection = st.sidebar.selectbox("Device Protection", ["No", "Yes", "No internet service"])
TechSupport = st.sidebar.selectbox("Tech Support", ["No", "Yes", "No internet service"])

StreamingTV = st.sidebar.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
StreamingMovies = st.sidebar.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

Contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])

PaymentMethod = st.sidebar.selectbox("Payment Method", [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)"
])

# -------------------------------
# CREATE INPUT DF
# -------------------------------
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

# -------------------------------
# DASHBOARD PAGE
# -------------------------------
if page == "🏠 Dashboard":
    st.title("📊 Telecom Churn Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.markdown(f"<div class='card'>👥 Gender<br><b>{Gender}</b></div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='card'>📅 Tenure<br><b>{Tenure}</b></div>", unsafe_allow_html=True)
    col3.markdown(f"<div class='card'>💰 Monthly Charges<br><b>${MonthlyCharges}</b></div>", unsafe_allow_html=True)

    st.divider()

    st.subheader("📌 Customer Summary")
    st.write(input_df)

# -------------------------------
# PREDICTION PAGE
# -------------------------------
elif page == "🔮 Prediction":

    st.title("🔮 Churn Prediction")

    if st.button("Predict", use_container_width=True):

        prediction = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

        if prediction == 1:
            st.markdown(f"<div class='card'>⚠️ High Churn Risk<br>Probability: {prob:.2f}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='card'>✅ Customer Will Stay<br>Probability: {prob:.2f}</div>", unsafe_allow_html=True)

        st.progress(float(prob))

# -------------------------------
# INSIGHTS PAGE
# -------------------------------
elif page == "📈 Insights":

    st.title("📈 Model Insights")

    model_rf = model.named_steps["model"]
    preprocessor = model.named_steps["preprocessor"]

    ohe = preprocessor.named_transformers_["cat"]
    cat_features = ohe.get_feature_names_out()
    num_features = preprocessor.transformers_[0][2]

    all_features = list(num_features) + list(cat_features)

    importances = model_rf.feature_importances_

    feat_df = pd.DataFrame({
        "Feature": all_features,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    st.subheader("Top Features")
    st.bar_chart(feat_df.set_index("Feature").head(10))

    fig, ax = plt.subplots()
    ax.pie(feat_df.head(5)["Importance"], labels=feat_df.head(5)["Feature"], autopct='%1.1f%%')
    st.pyplot(fig)
