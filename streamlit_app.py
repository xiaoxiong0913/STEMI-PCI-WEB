import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Load model, scaler, and feature list
model_path = r"gbm_model.pkl"
scaler_path = r"scaler.pkl"
features_path = r"features.txt"

try:
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    with open(scaler_path, 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    if not isinstance(scaler, StandardScaler):
        raise ValueError("Loaded scaler is not an instance of StandardScaler.")
    with open(features_path, 'r') as f:
        features = [line.strip().replace(" ", "_") for line in f]
except Exception as e:
    st.error(f"Error loading model, scaler, or features: {e}")
    st.stop()

st.set_page_config(layout="wide", page_icon="❤️")
st.title("STEMI Post-PCI 3-Year Mortality Prediction System")

# Custom CSS styling for beautification
st.write("""
<style>
.protocol-card {
    padding: 15px;
    border-radius: 10px;
    margin: 10px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.critical-card { border-left: 5px solid #dc3545; background-color: #fff5f5; }
.warning-card  { border-left: 5px solid #ffc107; background-color: #fff9e6; }
.green-card    { border-left: 5px solid #28a745; background-color: #f0fff4; }
.blue-card     { border-left: 5px solid #17a2b8; background-color: #d1ecf1; }
.result-card   { border-radius: 10px; padding: 20px; background-color: #f8f9fa; margin: 20px 0; }
</style>
""", unsafe_allow_html=True)

# Introduction section
st.write("# Introduction")
st.write("""
This clinical decision support tool integrates clinical, laboratory, and procedural parameters 
to predict 3-year mortality risk in STEMI patients after primary PCI. Validated with **AUC 0.91 (0.867-0.944)** and **93.4% accuracy**.
""")

# Clinical pathway cards
cols = st.columns([2, 3])
with cols[0]:
    st.write("""
    <div class='protocol-card critical-card'>
        <h4 style='color:#dc3545;'>High Risk Criteria</h4>
        <ul style='padding-left:20px'>
            <li>Probability ≥14.7%</li>
            <li>Age >75 years</li>
            <li>Hb <100 g/L</li>
            <li>AST >200 U/L</li>
        </ul>
    </div>
    <div class='protocol-card blue-card'>
        <h4 style='color:#17a2b8;'>Laboratory Alerts</h4>
        <ul style='padding-left:20px'>
            <li>Hb <100 g/L → Hematology consult</li>
            <li>AST >200 U/L → Cardiac enzyme monitoring</li>
            <li>Creatinine >150 μmol/L → Renal function assessment</li>
        </ul>
    </div>
    <div class='protocol-card warning-card'>
        <h4 style='color:#ffc107;'>Medication Recommendations</h4>
        <ul style='padding-left:20px'>
            <li>High-risk patients: High-intensity statin therapy</li>
            <li>Beta-blocker continuation unless contraindicated</li>
            <li>Dual antiplatelet therapy for ≥12 months</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
with cols[1]:
    st.write("""
    <div class='protocol-card green-card'>
        <h4 style='color:#28a745;'>Post-PCI Monitoring Protocol</h4>
        <ul style='padding-left:20px'>
            <li>Cardiac telemetry for 24-48 hours post-PCI</li>
            <li>Serial troponin measurements every 6 hours × 3</li>
            <li>Echocardiography within 24 hours</li>
            <li>Lipid profile assessment before discharge</li>
            <li>Cardiac rehabilitation referral</li>
            <li>Follow-up at 1, 3, 6, and 12 months</li>
            <li>Annual stress testing for high-risk patients</li>
            <li>Lifestyle modification counseling</li>
            <li>Smoking cessation support</li>
            <li>Diabetes management optimization</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Define normal value ranges for continuous variables
normal_ranges = {
    'Age': (18, 120),
    'Hb': (130, 175),
    'AST': (10, 40),
}

with st.sidebar:
    st.write("## Patient Parameters")
    with st.form("input_form"):
        inputs = {}
        inputs['Age'] = st.slider("Age (Years)", 30, 100, 65)
        inputs['Hb']  = st.slider("Hb (g/L)", 60, 200, 130)
        inputs['AST'] = st.slider("AST (U/L)", 5, 600, 30)
        inputs['Respiratory_support'] = st.selectbox("Respiratory support", ["No", "Yes"])
        inputs['Beta_blocker']        = st.selectbox("Beta blocker", ["No", "Yes"])
        inputs['Cardiotonics']        = st.selectbox("Cardiotonics", ["No", "Yes"])
        inputs['Statins']             = st.selectbox("Statins", ["No", "Yes"])
        # Numeric options for Stent_for_IRA
        inputs['Stent_for_IRA']       = st.selectbox("Stent for IRA", [0, 1, 2])
        submitted = st.form_submit_button("Predict Risk")

if submitted:
    try:
        # Map inputs to numeric values
        input_data = {}
        for k, v in inputs.items():
            if isinstance(v, str):
                input_data[k] = 1 if v == "Yes" else 0
            else:
                input_data[k] = v

        # Build DataFrame in feature order
        df = pd.DataFrame([input_data], columns=features)

        # Scale only the continuous features
        cont_feats = list(scaler.feature_names_in_)  # ['Age','Hb','AST']
        df_cont = df[cont_feats]
        df_cont_scaled = scaler.transform(df_cont)

        # Concatenate scaled continuous and raw categorical/numeric features
        df_cat = df.drop(columns=cont_feats)
        X = np.hstack([df_cont_scaled, df_cat.values])

        # Predict
        prob = model.predict_proba(X)[:, 1][0]
        risk_status = "High Risk" if prob >= 0.147 else "Low Risk"
        color = "#dc3545" if risk_status == "High Risk" else "#28a745"
        risk_message = (
            "High risk of mortality within 3 years."
            if risk_status == "High Risk"
            else "Low risk of mortality within 3 years."
        )

        # Check for abnormal values
        advice = []
        display_names = {
            'Age': 'Age', 'Hb': 'Hemoglobin', 'AST': 'AST',
            'Respiratory_support': 'Respiratory support',
            'Beta_blocker': 'Beta blocker', 'Cardiotonics': 'Cardiotonics',
            'Statins': 'Statins', 'Stent_for_IRA': 'Stent for IRA'
        }
        for var, val in inputs.items():
            if var in normal_ranges:
                lo, hi = normal_ranges[var]
                if val < lo or val > hi:
                    if var == 'Hb' and val < lo:
                        advice.append(f"<b>{display_names[var]} ({val} g/L)</b>: Below normal range (130–175). Consider anemia workup.")
                    elif var == 'Hb' and val > hi:
                        advice.append(f"<b>{display_names[var]} ({val} g/L)</b>: Above normal range. Evaluate for polycythemia.")
                    if var == 'AST' and val > hi:
                        advice.append(f"<b>{display_names[var]} ({val} U/L)</b>: Elevated above normal (10–40). May indicate myocardial injury.")
                    if var == 'Age' and val > 75:
                        advice.append(f"<b>{display_names[var]} ({val} years)</b>: Advanced age is an independent risk factor.")

        # Display result
        st.markdown(f"""
        <div class='result-card'>
            <h2 style='color:{color};'>Predicted 3-Year Mortality Risk: {prob*100:.1f}% ({risk_status})</h2>
            <p>{risk_message}</p>
        </div>
        """, unsafe_allow_html=True)

        # Display clinical advice if any
        if advice:
            st.markdown("<h4 style='color:red;'>Clinical Advisory:</h4>", unsafe_allow_html=True)
            for a in advice:
                st.markdown(f"<p style='color:red;'>{a}</p>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Prediction error: {e}")
        import traceback; st.text(traceback.format_exc())

# Footer
st.write("---")
st.write(
    "<div style='text-align:center; color:gray;'>"
    "Developed by Yichang Central People's Hospital</div>",
    unsafe_allow_html=True
)
