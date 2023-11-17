import streamlit as st
import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import StandardScaler

#loading the trained model
loaded_model = load_model('/Users/naakoshie/Downloads/my_model (2).h5')

#Function to preprocess data
def preprocess_input(data):
   
    scaler = StandardScaler()
    numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    data[numerical_features] = scaler.fit_transform(data[numerical_features])
    
    categorical_features = ['Contract', 'PaymentMethod', 'OnlineSecurity', 'TechSupport', 'gender', 'OnlineBackup', 'PaperlessBilling']
    for col in categorical_features:
        data[col], _ = pd.factorize(data[col])

    return data

# Function to make predictions
def predict_churn(data):
    scaler = StandardScaler()
    scaled_data = scaler .fit_transform(data)  
    prediction = loaded_model.predict(scaled_data)
    confidence = np.max(prediction)

    return prediction, confidence


# Streamlit App
st.title("Churn Prediction App")

# Collect user input
tenure = st.slider("Enter Tenure", min_value=0, max_value=100, step=1)
monthly_charges = st.number_input("Enter Monthly Charges", min_value=0.0, max_value=500.0, step=1.0)
total_charges = st.number_input("Enter Total Charges", min_value=0.0, max_value=5000.0, step=1.0)
contract = st.selectbox("Select Contract", ['Month-to-month', 'One year', 'Two year'])
payment_method = st.selectbox("Select Payment Method", ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
online_security = st.selectbox("Select Online Security", ['No', 'Yes'])
tech_support = st.selectbox("Select Tech Support", ['No', 'Yes'])
gender = st.selectbox("Select Gender", ['Male', 'Female'])
online_backup = st.selectbox("Select Online Backup", ['No', 'Yes'])
paperless_billing = st.selectbox("Select Paperless Billing", ['No', 'Yes'])

# Prepare user input as a DataFrame
user_data = pd.DataFrame({
    'tenure': [tenure],
    'MonthlyCharges': [monthly_charges],
    'TotalCharges': [total_charges],
    'Contract': [contract],
    'PaymentMethod': [payment_method],
    'OnlineSecurity': [online_security],
    'TechSupport': [tech_support],
    'gender': [gender],
    'OnlineBackup': [online_backup],
    'PaperlessBilling': [paperless_billing]
})

if st.button("Predict"):
        
        processed_data = preprocess_input(user_data)
        # Makes a prediction
        prediction, confidence = predict_churn(processed_data)

        # Displays the prediction result
        st.subheader("Prediction Result:")
        if confidence > 0.5:
            st.success(f"The customer is likely to stay (not churn) with confidence {confidence:.2%}")
        else:
            st.error(f"The customer is likely to churn with confidence {1 - confidence:.2%}")
