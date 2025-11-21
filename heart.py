import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the fixed model
try:
    model = joblib.load("gb_model.pkl")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# Try to load label encoders if they exist
try:
    label_encoders = joblib.load("label_encoders.pkl")
 
except:
    label_encoders = None


st.title("❤️ Heart Disease Prediction App")
st.write("### Enter patient details below:")

# --- Mappings (same as used in training) ---
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

# --- Input fields ---
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=105, value=50, step=1)
    sex = st.selectbox("Sex", list(sex_map.keys()))
    cp = st.selectbox("Chest Pain Type", list(cp_map.keys()))
    restingbp = st.number_input("Resting BP (mmHg)", min_value=0, max_value=300, value=120, step=1)
    chol = st.number_input("Cholesterol (mg/dl)", min_value=0, max_value=600, value=200, step=1)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", ["No", "Yes"])

with col2:
    restecg = st.selectbox("Resting ECG", list(restecg_map.keys()))
    maxhr = st.number_input("Max Heart Rate", min_value=0, max_value=250, value=150, step=1)
    exa = st.selectbox("Exercise Angina", list(exang_map.keys()))
    oldpeak = st.number_input("Oldpeak (ST Depression)", min_value=-5.0, max_value=8.0, value=0.0, step=0.1)
    slope = st.selectbox("ST Slope", list(slope_map.keys()))

# --- Prepare input data ---
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

# Convert to DataFrame with the same feature order as training
feature_order = [
    "Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol", 
    "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope"
]

# Create input array in the correct order
input_values = [input_data[feature] for feature in feature_order]

# --- Prediction Section ---
st.markdown("---")

if st.button("🔍 Predict Heart Disease", type="primary", use_container_width=True):
    try:
        # Method 1: Convert to numpy array with explicit float32 type
        input_array = np.array([input_values], dtype=np.float32)
        
        # Make prediction
        prediction = model.predict(input_array)[0]
        prediction_proba = model.predict_proba(input_array)[0]
        
        # Display results
        st.write("## 🔍 Prediction Results")
        
        # Create result columns
        col_result, col_prob = st.columns(2)
        
        with col_result:
            if prediction == 1:
                st.error("""
                **HEART DISEASE DETECTED**
                
                The model predicts a **high risk** of heart disease.
                Please consult with a healthcare professional.
                """)
            else:
                st.success("""
                **NO HEART DISEASE DETECTED**
                
                The model predicts a **low risk** of heart disease.
                Maintain a healthy lifestyle!
                """)
        
        with col_prob:
            st.metric(
                label="Confidence Score", 
                value=f"{max(prediction_proba)*100:.1f}%",
                delta=f"Risk: {'High' if prediction == 1 else 'Low'}"
            )
            
            # Show probability breakdown
            st.write("**Probability Breakdown:**")
            prob_df = pd.DataFrame({
                'Condition': ['No Heart Disease', 'Heart Disease'],
                'Probability': [f"{prediction_proba[0]*100:.1f}%", f"{prediction_proba[1]*100:.1f}%"]
            })
            st.dataframe(prob_df, hide_index=True, use_container_width=True)
        
        # Show input summary
        with st.expander("📋 View Input Summary"):
            input_summary = pd.DataFrame([input_data])
            st.dataframe(input_summary, use_container_width=True)
            
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")
        
        # Debug information
        with st.expander("🔧 Debug Information"):
            st.write("Input values:", input_values)
            st.write("Input array dtype:", input_array.dtype if 'input_array' in locals() else 'Not created')
            st.write("Input array shape:", input_array.shape if 'input_array' in locals() else 'Not created')
            
            # Try alternative prediction method
            st.write("### Trying alternative prediction method...")
            try:
                # Method 2: Use DataFrame
                input_df = pd.DataFrame([input_data])[feature_order].astype(np.float32)
                alt_prediction = model.predict(input_df)[0]
                st.success(f"Alternative method successful! Prediction: {alt_prediction}")
            except Exception as e2:
                st.error(f"Alternative method also failed: {str(e2)}")

