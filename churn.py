import streamlit as st
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# -----------------------------------------------------------------------------
# --- Load Model, Encoders, and Feature Names (cached for performance)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_assets():
    """
    Loads the ML model, label encoders, and feature names from pickle files.
    """
    try:
        with open("customer_churn_model.pkl", "rb") as f:
            model_data = pickle.load(f)
        with open("encoders.pkl", "rb") as f:
            encoders = pickle.load(f)
        return model_data["model"], encoders, model_data["features_names"]
    except FileNotFoundError:
        return None, None, None

rfc_model, encoders, feature_names = load_assets()

# -----------------------------------------------------------------------------
# --- Error Handling
# -----------------------------------------------------------------------------
if rfc_model is None or encoders is None:
    st.error("⚠️ Model or encoder files not found.")
    st.write("Please place `customer_churn_model.pkl` and `encoders.pkl` in the same folder as this `app.py`.")
    st.stop()

# -----------------------------------------------------------------------------
# --- Streamlit UI
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Customer Churn Prediction", page_icon="🔮", layout="centered")
st.title("🔮 Customer Churn Prediction App")
st.write("Predict whether a customer is likely to churn based on their account details.")

# --- Sidebar Inputs ---
st.sidebar.header("🧾 Customer Details")
input_data = {}

# Demographics
input_data['gender'] = st.sidebar.selectbox("Gender", ["Male", "Female"])
input_data['SeniorCitizen'] = st.sidebar.selectbox("Senior Citizen", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
input_data['Partner'] = st.sidebar.selectbox("Partner", ["Yes", "No"])
input_data['Dependents'] = st.sidebar.selectbox("Dependents", ["Yes", "No"])
input_data['tenure'] = st.sidebar.slider("Tenure (months)", 0, 72, 12)

# Services
input_data['PhoneService'] = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
input_data['MultipleLines'] = st.sidebar.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
input_data['InternetService'] = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
input_data['OnlineSecurity'] = st.sidebar.selectbox("Online Security", ["No", "Yes", "No internet service"])
input_data['OnlineBackup'] = st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"])
input_data['DeviceProtection'] = st.sidebar.selectbox("Device Protection", ["No", "Yes", "No internet service"])
input_data['TechSupport'] = st.sidebar.selectbox("Tech Support", ["No", "Yes", "No internet service"])
input_data['StreamingTV'] = st.sidebar.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
input_data['StreamingMovies'] = st.sidebar.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

# Billing
input_data['Contract'] = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
input_data['PaperlessBilling'] = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
input_data['PaymentMethod'] = st.sidebar.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)
input_data['MonthlyCharges'] = st.sidebar.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0)
input_data['TotalCharges'] = st.sidebar.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=1500.0)

# -----------------------------------------------------------------------------
# --- Prediction Logic
# -----------------------------------------------------------------------------
if st.button("🔍 Predict Churn", use_container_width=True):
    # Convert input to DataFrame
    input_df = pd.DataFrame([input_data])

    # Encode categorical features
    for col, encoder in encoders.items():
        if col in input_df.columns:
            input_df[col] = encoder.transform(input_df[col])

    # Match column order to model training
    input_df = input_df[feature_names]

    # Predict
    prediction = rfc_model.predict(input_df)
    prediction_proba = rfc_model.predict_proba(input_df)

    # -----------------------------------------------------------------------------
    # --- Display Result
    # -----------------------------------------------------------------------------
    st.subheader("📊 Prediction Result")
    if prediction[0] == 1:
        st.error("Prediction: **Customer will CHURN** 😞")
    else:
        st.success("Prediction: **Customer will NOT CHURN** 😊")

    # Probabilities
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Probability of NOT Churning", f"{prediction_proba[0][0]:.0%}")
    with col2:
        st.metric("Probability of Churning", f"{prediction_proba[0][1]:.0%}")

    # Optional: confidence info
    st.caption(f"Confidence Level: {max(prediction_proba[0])*100:.2f}%")

# -----------------------------------------------------------------------------
# --- Footer
# -----------------------------------------------------------------------------
st.markdown("---")
st.caption("Developed with ❤️ using Streamlit and Random Forest Classifier")
