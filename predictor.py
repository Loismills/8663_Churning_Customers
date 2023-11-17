
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.models import load_model

# Load the pre-trained model
model_path = '/Users/naakoshie/Downloads/my_model (2).h5'
loaded_model = load_model(model_path)

# Define the Streamlit app
def main():
    # Set the title of the web app
    st.title("Customer Churn Prediction App")

    # Create input fields for user data
    tenure = st.slider("Tenure", 0, 72, 0)
    monthly_charges = st.slider("Monthly Charges", 0.0, 118.75, 0.0)
    total_charges = st.slider("Total Charges", 0.0, 8684.8, 0.0)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two years"])
    payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    gender = st.selectbox("Gender", ["Female", "Male"])
    online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])

    # Convert categorical features to numerical values
    contract_mapping = {"Month-to-month": 0, "One year": 1, "Two years": 2}
    payment_method_mapping = {"Electronic check": 0, "Mailed check": 1, "Bank transfer (automatic)": 2, "Credit card (automatic)": 3}
    yes_no_mapping = {"No": 0, "Yes": 1, "No internet service": 2}
    gender_mapping = {"Female": 0, "Male": 1}

    contract = contract_mapping[contract]
    payment_method = payment_method_mapping[payment_method]
    online_security = yes_no_mapping[online_security]
    tech_support = yes_no_mapping[tech_support]
    online_backup = yes_no_mapping[online_backup]
    gender = gender_mapping[gender]
    paperless_billing = yes_no_mapping[paperless_billing]

    # Create a DataFrame with the user input
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

    # Standardize the input data
    scaler = StandardScaler()
    scaled_user_data = scaler.fit_transform(user_data)

    if st.button("Predict"):
        # Make a prediction
        prediction = loaded_model.predict(scaled_user_data)
        confidence_factor = prediction[0][0]  # Probability of not churn class

        # Display the prediction result
        st.subheader("Prediction Result:")
        if confidence_factor > 0.5:
            st.success(f"The customer is likely to stay (not churn) with confidence {confidence_factor:.2%}")
        else:
            st.error(f"The customer is likely to churn with confidence {1 - confidence_factor:.2%}")

if __name__ == '__main__':
    main()
