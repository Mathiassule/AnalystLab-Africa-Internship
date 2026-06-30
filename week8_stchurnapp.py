import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
import os

# ==========================================
# Application Setup & Data Loading
# ==========================================
st.set_page_config(page_title="Churn Predictor", page_icon="📉", layout="wide")

@st.cache_resource
def load_model():
    """Loads the predictive model from the pkl file."""
    model_path = "churn_model.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        st.warning(f"⚠️ Model file '{model_path}' not found! Using a dummy model for demonstration. Please place your .pkl file in the same directory.")
        # Dummy model class to prevent the app from crashing before you upload the pkl
        class DummyModel:
            def predict_proba(self, X):
                # Returns dummy probability: 65% chance of churn
                return np.array([[0.35, 0.65]]) 
        return DummyModel()

model = load_model()

# Dummy Feature Importance (Replace this with actual model.feature_importances_ if available)
feature_importance_df = pd.DataFrame({
    'Feature': ['Contract_Month-to-month', 'Tenure', 'InternetService_Fiber optic', 'MonthlyCharges', 'TechSupport_No'],
    'Importance': [0.25, 0.18, 0.15, 0.12, 0.08]
}).sort_values(by='Importance', ascending=True)

def preprocess_inputs(raw_data):
    """
    Transforms the 5 UI inputs into the 30-feature array expected by the model.
    You will need to adjust the exact indices/column names based on your training data.
    """
    # Initialize an array of 30 zeros to represent default/baseline values
    features = np.zeros(30)
    
    # 1. Scale/Assign Numerical variables (Assuming they go in the first few slots)
    features[0] = raw_data["Tenure"]
    features[1] = raw_data["MonthlyCharges"]
    
    # 2. Encode Categorical variables (Example mappings)
    # Contract: Let's assume Month-to-month is feature idx 5, One year is 6, Two year is 7
    if raw_data["Contract"] == "Month-to-month":
        features[5] = 1
    elif raw_data["Contract"] == "One year":
        features[6] = 1
    elif raw_data["Contract"] == "Two year":
        features[7] = 1
        
    # Internet Service: Fiber optic idx 8, DSL idx 9
    if raw_data["InternetService"] == "Fiber optic":
        features[8] = 1
    elif raw_data["InternetService"] == "DSL":
        features[9] = 1
        
    # Tech Support: Yes idx 10, No idx 11
    if raw_data["TechSupport"] == "Yes":
        features[10] = 1
    elif raw_data["TechSupport"] == "No":
        features[11] = 1

    # Return as a 2D array: shape (1, 30)
    return features.reshape(1, -1)


# ==========================================
# Sidebar UI
# ==========================================
with st.sidebar.form("prediction_form"):
    st.image("https://cdn-icons-png.flaticon.com/512/3126/3126647.png", width=50) # Optional logo
    st.header("Customer Profile")
    
    # Numerical Inputs
    st.subheader("Core Account Metrics")
    tenure = st.slider("Tenure (Months)", min_value=0, max_value=72, value=12)
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=65.0)
    
    # Categorical Inputs
    st.subheader("Key Services & Contract")
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])

    submit_button = st.form_submit_button(label="Predict Churn Risk", use_container_width=True)

# ==========================================
# Main Application Body
# ==========================================
st.title("Customer Churn Prediction Dashboard")
st.markdown("Analyze customer profiles and predict their likelihood of churning using our machine learning model.")

if submit_button:
    # 1. Collect form data 
    raw_data = {
        "Tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "Contract": contract,
        "InternetService": internet_service,
        "TechSupport": tech_support
    }

    # 2. Preprocess into 30-feature array
    processed_data = preprocess_inputs(raw_data)
    
    # 3. Generate Prediction (Probability of Churn / Class 1)
    # Using [0][1] to get the probability of the positive class (Churn = Yes)
    prediction_proba = model.predict_proba(processed_data)[0][1] 

    # 4. Display Results in Columns
    st.divider()
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Churn Probability")
        
        # Determine gauge color based on risk level
        gauge_color = "darkgreen"
        if prediction_proba > 0.6:
            gauge_color = "darkred"
        elif prediction_proba > 0.3:
            gauge_color = "darkorange"

        # Create Plotly Gauge
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prediction_proba * 100,
            number = {'suffix': "%", 'valueformat': ".1f"},
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Risk Level", 'font': {'size': 24}},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': gauge_color},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 30], 'color': "rgba(0, 255, 0, 0.1)"},
                    {'range': [30, 60], 'color': "rgba(255, 165, 0, 0.1)"},
                    {'range': [60, 100], 'color': "rgba(255, 0, 0, 0.1)"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        fig_gauge.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col2:
        st.subheader("Top Risk Factors (Global)")
        
        # Create Plotly Bar Chart for Feature Importance
        fig_bar = px.bar(
            feature_importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            color='Importance',
            color_continuous_scale="Reds"
        )
        fig_bar.update_layout(
            height=350, 
            margin=dict(l=20, r=20, t=30, b=20),
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # 5. Business Logic & Actionable Recommendations
    st.subheader("Recommended Actions")
    if prediction_proba >= 0.6:
        st.error("🚨 **High Churn Risk!** Immediate intervention required. Offer an exclusive retention discount or a free service upgrade to a long-term contract.")
    elif prediction_proba >= 0.3:
        st.warning("⚠️ **Moderate Churn Risk.** Reach out with a personalized check-in email. Highlight the benefits of renewing their contract and verify their satisfaction with tech support.")
    else:
        st.success("✅ **Low Churn Risk.** Maintain current engagement strategy. Customer is likely stable.")

else:
    # State before submission
    st.info("👈 Please adjust the customer profile in the sidebar and click **Predict Churn Risk** to evaluate the customer.")