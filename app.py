import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# ==============================
# Page Config
# ==============================
st.set_page_config(page_title="Fraud Detection System", layout="centered")
st.title("🕵️ Financial Transaction Fraud Detector")
st.markdown("""
**Model**: Neural Network (MLP) — Highest ROC AUC  
**Dataset**: Synthetic Financial Transactions  
**Threshold**: 2% probability → flagged as fraud  
**Explainability**: SHAP (KernelExplainer)
""")

# ==============================
# Load Model Only
# ==============================
@st.cache_resource
def load_model():
    try:
        pipeline = joblib.load("best_fraud_detection_model.pkl")
        return pipeline
    except FileNotFoundError:
        st.error("Model file `best_fraud_detection_model.pkl` not found in the current directory.")
        st.stop()

best_pipeline = load_model()
preprocessor = best_pipeline.named_steps["prep"]
clf = best_pipeline.named_steps["clf"]

# Feature names
numeric_features = [
    "amount", "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest",
    "balanceDiffOrig", "balanceDiffDest"
]
cat_features = preprocessor.named_transformers_["cat"].get_feature_names_out(["type"])
feature_names = np.concatenate([numeric_features, list(cat_features)])

# ==============================
# Helper: Normalize SHAP values (same as your notebook)
# ==============================
def normalize_shap(s, X):
    s = np.asarray(s)
    n_samples = X.shape[0]
    n_features = X.shape[1]

    if isinstance(s, (list, tuple)):
        if len(s) > 1:
            s = np.asarray(s[1])
        else:
            s = np.asarray(s[0])

    if s.ndim == 3:
        if s.shape[2] > 1:
            s = s[:, :, 1]
        else:
            s = s[:, :, 0]

    if s.ndim == 1:
        if s.size == n_features:
            s = s.reshape(1, -1)
        else:
            s = s.reshape(1, -1)
    elif s.ndim == 2:
        if s.shape[0] == n_features and s.shape[1] == n_samples:
            s = s.T

    if s.ndim != 2 or s.shape[1] != n_features:
        s = s.reshape(n_samples, n_features)

    if s.shape[0] != n_samples:
        s = s[:n_samples, :]

    return s

# ==============================
# User Input Form
# ==============================
st.header("Enter Transaction Details")

with st.form("transaction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        trans_type = st.selectbox("Transaction Type", 
                                  ["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"])
        amount = st.number_input("Amount", min_value=0.0, value=10000.0, step=100.0)
        oldbalanceOrg = st.number_input("Origin Balance Before", min_value=0.0, value=50000.0)
        newbalanceOrig = st.number_input("Origin Balance After", min_value=0.0, value=40000.0)
    
    with col2:
        oldbalanceDest = st.number_input("Destination Balance Before", min_value=0.0, value=0.0)
        newbalanceDest = st.number_input("Destination Balance After", min_value=0.0, value=10000.0)
    
    submitted = st.form_submit_button("🔍 Analyze Transaction")

# ==============================
# Prediction & SHAP Explanation
# ==============================
if submitted:
    # Compute derived features
    balanceDiffOrig = oldbalanceOrg - newbalanceOrig
    balanceDiffDest = newbalanceDest - oldbalanceDest
    
    # Input DataFrame
    input_df = pd.DataFrame([{
        "type": trans_type,
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest,
        "balanceDiffOrig": balanceDiffOrig,
        "balanceDiffDest": balanceDiffDest
    }])
    
    st.subheader("Transaction Summary")
    st.dataframe(input_df.T.rename(columns={0: "Value"}), use_container_width=True)
    
    # Predict probability
    fraud_prob = best_pipeline.predict_proba(input_df)[0, 1]
    FRAUD_THRESHOLD = 0.02
    is_fraud = fraud_prob >= FRAUD_THRESHOLD
    
    st.subheader("Prediction Result")
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Fraud Probability", f"{fraud_prob*100:.2f}%")
    with col_b:
        if is_fraud:
            st.error("🚨 **HIGH RISK: Likely Fraudulent**")
        else:
            st.success("✅ **Low Risk: Appears Legitimate**")
    
    # ==============================
    # SHAP Explanation (Exactly like your notebook test cell)
    # ==============================
    st.subheader("SHAP Explanation")
    st.markdown("Computing SHAP values for this transaction... (may take 10–30 seconds)")

    with st.spinner("Running SHAP KernelExplainer..."):
        # Transform input
        X_transformed = preprocessor.transform(input_df)
        
        # Create small background (same logic as notebook: sample from training-like data)
        # We simulate a realistic background without needing saved file
        np.random.seed(42)
        dummy_background = pd.DataFrame({
            "type": np.random.choice(["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"], 50),
            "amount": np.random.uniform(1, 200000, 50),
            "oldbalanceOrg": np.random.uniform(0, 1000000, 50),
            "newbalanceOrig": np.random.uniform(0, 1000000, 50),
            "oldbalanceDest": np.random.uniform(0, 1000000, 50),
            "newbalanceDest": np.random.uniform(0, 1000000, 50),
            "balanceDiffOrig": np.random.uniform(-50000, 50000, 50),
            "balanceDiffDest": np.random.uniform(-50000, 50000, 50),
        })
        bg_processed = preprocessor.transform(dummy_background)
        
        # Create explainer
        explainer = shap.KernelExplainer(clf.predict_proba, bg_processed)
        
        # Compute SHAP values
        raw_shap = explainer.shap_values(X_transformed, nsamples=100)
        
        # Normalize (exact same function from your notebook)
        shap_values = normalize_shap(raw_shap, X_transformed)
        
        # Expected value for fraud class
        base_value = explainer.expected_value[1] if hasattr(explainer.expected_value, '__len__') and len(explainer.expected_value) > 1 else explainer.expected_value

        # Force Plot
        st.markdown("**Force Plot: How features push the prediction**")
        fig_force = plt.figure(figsize=(12, 4))
        shap.force_plot(
            base_value,
            shap_values,
            X_transformed,
            feature_names=feature_names,
            matplotlib=True,
            show=False,
            figsize=(12, 3)
        )
        st.pyplot(fig_force)
        
        # Bar Plot
        st.markdown("**Top Features by Impact (Absolute SHAP Value)**")
        fig_bar, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, X_transformed, feature_names=feature_names, plot_type="bar", show=False)
        st.pyplot(fig_bar)

st.sidebar.header("About")
st.sidebar.info("""
This app uses your trained **Neural Network** model.  
SHAP explanations are computed on-the-fly using KernelExplainer  
with a small simulated background (no saved file needed).
""")
st.sidebar.caption("Threshold: 2% fraud probability → flagged")