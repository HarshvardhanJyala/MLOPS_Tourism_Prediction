import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="JyalaHarsha-2025/MLOPS_Tourism_Prediction", filename="tourism_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Machine Failure Prediction
st.title("Tourism Product Prediction App")
st.write("""
This application predicts the likelihood of a Customer will be taking the tourism package based on its operational parameters.
Please enter the sensor and configuration data below to get a prediction.
""")

# ---- CUSTOMER INFORMATION SECTION ----
st.header("🧍 Customer Information")

age = st.number_input("Age", min_value=10, max_value=100, value=30)
typeofcontact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
citytier = st.selectbox("City Tier", [1, 2, 3])
occupation = st.selectbox("Occupation", ["Salaried", "Freelancer", "Small Business", "Large Business"])
gender = st.selectbox("Gender", ["Male", "Female"])
numberofpersonvisiting = st.slider("Number of Persons Visiting", min_value=1, max_value=10, value=2)
preferredpropertystar = st.selectbox("Preferred Property Star", [1.0, 2.0, 3.0, 4.0, 5.0])
maritalstatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
numberoftrips = st.number_input("Number of Trips per Year", min_value=0, max_value=100, value=2)
passport = st.selectbox("Do they have a Passport?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
owncar = st.selectbox("Do they own a Car?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
numberofchildrenvisiting = st.slider("Number of Children Visiting", min_value=0, max_value=5, value=0)
designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
monthlyincome = st.number_input("Monthly Income", min_value=0.0, value=50000.0, step=1000.0)

# ---- INTERACTION DATA SECTION ----
st.header("📊 Sales Interaction Data")

pitchsatisfactionscore = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
productpitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
numberoffollowups = st.slider("Number of Follow-ups", min_value=0, max_value=10, value=2)
durationofpitch = st.slider("Duration of Pitch (minutes)", min_value=0, max_value=60, value=15)

# ---- Assemble input data ----
input_data = pd.DataFrame([{
    'age': age,
    'typeofcontact': typeofcontact,
    'citytier': citytier,
    'occupation': occupation,
    'gender': gender,
    'numberofpersonvisiting': numberofpersonvisiting,
    'preferredpropertystar': preferredpropertystar,
    'maritalstatus': maritalstatus,
    'numberoftrips': numberoftrips,
    'passport': passport,
    'owncar': owncar,
    'numberofchildrenvisiting': numberofchildrenvisiting,
    'designation': designation,
    'monthlyincome': monthlyincome,
    'pitchsatisfactionscore': pitchsatisfactionscore,
    'productpitched': productpitched,
    'numberoffollowups': numberoffollowups,
    'durationofpitch': durationofpitch
}])

st.subheader("📥 Input Summary")
st.dataframe(input_data)


if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    result = "Product Taken" if prediction == 1 else "Product Not Taken"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
