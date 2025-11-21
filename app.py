import streamlit as st
import pandas as pd
import numpy as np
try:
    import joblib
except ImportError as e:
    st.error(f"joblib import error: {e}. Please check requirements.txt")

# Set page config first
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide"
)

# Load the model with caching
@st.cache_resource
def load_model():
    try:
        model = joblib.load("gb_model.pkl")
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

# Load label encoders if available
@st.cache_resource
def load_encoders():
    try:
        encoders = joblib.load("label_encoders.pkl")
        return encoders
    except Exception as e:
        st.warning(f"Label encoders not found: {e}")
        return None

# Initialize the app
def main():
    st.title("❤️ Heart Disease Prediction App")
    st.write("Enter patient details below to predict heart disease risk.")
    
    # Load model
    model = load_model()
    encoders = load_encoders()
    
    if model is None:
        st.error("""
        ⚠️ **Model failed to load.** Please ensure:
        - `gb_model_fixed.pkl` is in the root directory
        - The file is not corrupted
        - All dependencies are installed
        """)
        return
    
    # Input mappings
    sex_map = {"Female": 0, "Male": 1}
    cp_map = {
        "ASY (Asymptomatic)": 0,
        "NAP (Non-Anginal Pain)": 1,
        "ATA (Atypical Angina)": 2,
        "TA (Typical Angina)": 3
    }
    restecg_map = {"Normal": 0, "ST": 1, "LVH": 2}
    exang_map = {"No": 0, "Yes": 1}
    slope_map = {"Down": 0, "Flat": 1, "Up": 2}

    # Create input form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Patient Demographics")
            age = st.number_input("Age", min_value=1, max_value=120, value=50)
            sex = st.selectbox("Sex", list(sex_map.keys()))
            restingbp = st.number_input("Resting BP (mmHg)", min_value=0, max_value=300, value=120)
            
        with col2:
            st.subheader("Clinical Parameters")
            chol = st.number_input("Cholesterol (mg/dl)", min_value=0, max_value=600, value=200)
            maxhr = st.number_input("Max Heart Rate", min_value=0, max_value=250, value=150)
            cp = st.selectbox("Chest Pain Type", list(cp_map.keys()))
        
        col3, col4 = st.columns(2)
        
        with col3:
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", ["No", "Yes"])
            restecg = st.selectbox("Resting ECG", list(restecg_map.keys()))
            
        with col4:
            exa = st.selectbox("Exercise Angina", list(exang_map.keys()))
            oldpeak = st.number_input("Oldpeak (ST Depression)", min_value=-5.0, max_value=8.0, value=0.0, step=0.1)
            slope = st.selectbox("ST Slope", list(slope_map.keys()))
        
        # Submit button
        submitted = st.form_submit_button("🔍 Predict Heart Disease", type="primary", use_container_width=True)
    
    # When form is submitted
    if submitted:
        try:
            # Prepare input data
            input_data = {
                "Age": float(age),
                "Sex": sex_map[sex],
                "ChestPainType": cp_map[cp],
                "RestingBP": float(restingbp),
                "Cholesterol": float(chol),
                "FastingBS": 1 if fbs == "Yes" else 0,
                "RestingECG": restecg_map[restecg],
                "MaxHR": float(maxhr),
                "ExerciseAngina": exang_map[exa],
                "Oldpeak": float(oldpeak),
                "ST_Slope": slope_map[slope]
            }
            
            # Feature order (must match training)
            feature_order = [
                "Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol", 
                "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope"
            ]
            
            # Prepare input array
            input_values = [input_data[feature] for feature in feature_order]
            input_array = np.array([input_values], dtype=np.float32)
            
            # Make prediction
            prediction = model.predict(input_array)[0]
            prediction_proba = model.predict_proba(input_array)[0]
            
            # Display results
            st.markdown("---")
            st.subheader("🎯 Prediction Results")
            
            # Result columns
            col_result, col_prob = st.columns(2)
            
            with col_result:
                if prediction == 1:
                    st.error("""
                    ⚠️ **HEART DISEASE DETECTED**
                    
                    The model predicts a **high risk** of heart disease.
                    Please consult with a healthcare professional.
                    """)
                else:
                    st.success("""
                    ✅ **NO HEART DISEASE DETECTED**
                    
                    The model predicts a **low risk** of heart disease.
                    Maintain a healthy lifestyle!
                    """)
            
            with col_prob:
                # Confidence score
                confidence = max(prediction_proba) * 100
                st.metric(
                    label="Prediction Confidence", 
                    value=f"{confidence:.1f}%"
                )
                
                # Probability breakdown
                st.write("**Probability Breakdown:**")
                prob_data = {
                    'Condition': ['No Heart Disease', 'Heart Disease'],
                    'Probability': [f"{prediction_proba[0]*100:.1f}%", f"{prediction_proba[1]*100:.1f}%"]
                }
                st.dataframe(prob_data, use_container_width=True, hide_index=True)
                
        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")
            st.info("💡 Please check all input values and try again.")

    # Sidebar info
    with st.sidebar:
        st.header("ℹ️ About")
        st.write("""
        This app uses a Gradient Boosting Classifier to predict 
        heart disease risk based on clinical parameters.
        
        **Disclaimer:** For educational purposes only. 
        Always consult healthcare professionals for medical advice.
        """)
        
        st.header("📊 Input Summary")
        if submitted:
            input_summary = pd.DataFrame([input_data])
            st.dataframe(input_summary, use_container_width=True)

# Run the app
if __name__ == "__main__":
    main()

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit | For educational purposes only")