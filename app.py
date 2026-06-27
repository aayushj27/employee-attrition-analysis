import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the LITE model and scaler
model = joblib.load('attrition_model_lite.pkl')
scaler = joblib.load('scaler_lite.pkl')

st.title("🏢 Employee Attrition Predictor")
st.markdown("Enter employee details below to predict their dynamic flight risk.")

st.sidebar.header("Employee Input Features")

# Interactive Sliders
age = st.sidebar.slider("Age", 18, 65, 30)
monthly_income = st.sidebar.slider("Monthly Income ($)", 1000, 20000, 5000)
overtime_input = st.sidebar.selectbox("Works Overtime?", ["No", "Yes"])
years_at_company = st.sidebar.slider("Years at Company", 0, 40, 5)

# Convert Overtime to numerical (1 for Yes, 0 for No) to match our dataset encoding
overtime = 1 if overtime_input == "Yes" else 0

if st.button("Predict Attrition Risk"):
    # 1. Package the inputs into a dataframe that matches the lite model's training structure
    input_data = pd.DataFrame({
        'Age': [age],
        'MonthlyIncome': [monthly_income],
        'OverTime': [overtime],
        'YearsAtCompany': [years_at_company]
    })
    
    # 2. Scale the data
    input_scaled = scaler.transform(input_data)
    
    # 3. Get the dynamic probability from the model!
    # predict_proba returns [prob_stay, prob_leave]. We want prob_leave [0][1].
    probability_leave = model.predict_proba(input_scaled)[0][1]
    risk_percentage = round(probability_leave * 100, 1)
    
    # 4. Display the results dynamically based on our 30% threshold
    st.subheader(f"Calculated Flight Risk: {risk_percentage}%")
    
    # Dynamic Progress Bar
    st.progress(int(risk_percentage))
    
    if risk_percentage >= 30.0:
        st.error(f"⚠️ High Risk of Resignation")
        st.markdown("**HR Action:** Schedule a 1-on-1. Consider workload reduction and a salary review.")
    else:
        st.success(f"✅ Low Risk of Resignation")