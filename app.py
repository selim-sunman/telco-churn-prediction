import streamlit as st
import requests

# ------------------------------------------------------
# 1. PAGE SETTINGS AND API INFORMATION
# ------------------------------------------------------
st.set_page_config(page_title="Telco Churn Prediction", page_icon="📡", layout="wide")

# the endpoint you specified in your api.py file
API_URL = "http://localhost:8000/predict" 

# ------------------------------------------------------
# 2. MAIN SCREEN TITLE
# ------------------------------------------------------
st.title("📡 Telecommunications Customer Churn Estimation Tool")
st.markdown("""
    Please enter your customer profile, services used, and billing details in the menu on the left and click the **Estimate** button. 
    The machine learning model in the background will analyze this data and calculate the customer's churn risk.
""")
st.divider()

# ------------------------------------------------------
# 3. GETTING DATA FROM THE USER (SIDEBAR)
# CustomerData in api.py is configured according to the Pydantic class.
# ------------------------------------------------------
st.sidebar.header("1. Demographic Information")
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior_citizen_input = st.sidebar.checkbox("Senior Citizen")
# api.py expects the value of SeniorCitizen to be 1 or 0 (int)
senior_citizen = 1 if senior_citizen_input else 0 
partner = st.sidebar.selectbox("Does he/she have a partner? (Partner)", ["Yes", "No"])
dependents = st.sidebar.selectbox("Does he/she have dependents?", ["Yes", "No"])

st.sidebar.header("2. Customer Services")
phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.sidebar.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
online_backup = st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"])
device_protection = st.sidebar.selectbox("Device Protection", ["Yes", "No", "No internet service"])
tech_support = st.sidebar.selectbox("Technical Support (TechSupport)", ["Yes", "No", "No internet service"])
streaming_tv = st.sidebar.selectbox("TV Broadcast (Streaming TV)", ["Yes", "No", "No internet service"])
streaming_movies = st.sidebar.selectbox("Movie Streaming", ["Yes", "No", "No internet service"])

st.sidebar.header("3. Contract and Invoice")
tenure = st.sidebar.number_input("Tenure (Months)", min_value=0, max_value=100, value=12)
contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
payment_method = st.sidebar.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])
monthly_charges = st.sidebar.number_input("Monthly Charges", min_value=0.0, value=70.0)
total_charges = st.sidebar.number_input("Total Charges", min_value=0.0, value=840.0)

# ------------------------------------------------------
# 4. PREPARING JSON DATA FOR API (PAYLOAD)
# The keys are exactly the same as the CustomerData properties in api.py.
# ------------------------------------------------------
payload = {
    "tenure": tenure,
    "SeniorCitizen": senior_citizen,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    "gender": gender,
    "Partner": partner,
    "Dependents": dependents,
    "PhoneService": phone_service,
    "MultipleLines": multiple_lines,
    "InternetService": internet_service,
    "OnlineSecurity": online_security,
    "OnlineBackup": online_backup,
    "DeviceProtection": device_protection,
    "TechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_movies,
    "Contract": contract,
    "PaperlessBilling": paperless_billing,
    "PaymentMethod": payment_method
}

# We're dividing the interface into two parts.
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Customer Data to be Sent")
    st.json(payload)

# ------------------------------------------------------
# 5. SENDING A REQUEST TO THE API AND SHOWING THE RESULT
# ------------------------------------------------------
with col2:
    st.subheader("Model Prediction")
    
    if st.button("Churn Prediction", type="primary", use_container_width=True):
        
        with st.spinner("Connecting to the API and running the model..."):
            try:
                response = requests.post(API_URL, json=payload)
                
                if response.status_code == 200:
                    api_answer = response.json()
                    
                    #We are reading based on the keywords returned from api.py.
                    prediction = api_answer.get("churn_prediction", "Unknown")
                    probability = api_answer.get("churn_probability", 0.0)
                    
                    if prediction == "Yes":
                        st.error("⚠️ **HIGH-RISK CUSTOMER!** This customer is expected to cancel their subscription.")
                        st.metric(label="Probability of Secession (Churn)", value=f"%{probability * 100:.0f}")
                        st.warning("Offering a promotion or discount is recommended to retain this customer.")
                    else:
                        st.success("✅ **SAFE.** This customer is expected to remain in the system.")
                        st.metric(label="Probability of Secession (Churn)", value=f"%{probability * 100:.0f}")
                        st.balloons()
                        
                elif response.status_code == 422:
                     st.error("⚠️ There is an error in the submitted data format. Please check all fields.")
                     st.write("Error Details:", response.json())
                else:
                    st.error(f"Server Error! Status Code: {response.status_code}")
                    st.write(response.text)
                    
            except requests.exceptions.ConnectionError:
                st.error("🚨 The API is unreachable. Please ensure your FastAPI server is running at `http://localhost:8000`.")