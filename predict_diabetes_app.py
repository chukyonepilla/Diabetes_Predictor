#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import os

# --- File paths (update if needed) ---
MODEL_PATHS = {
    "Random Forest": "saved_models_phase1/random_forest_model_SM.pkl",
    "Logistic Regression": "saved_models_phase1/logistic_regression_model_SM.pkl",
    "XGBoost": "saved_models_phase1/xgboost_model_SM.pkl"
}
SCALER_PATH = "saved_models_phase1/scaler_SM.pkl"
HYPERPARAMS_PATH = "saved_models_phase1/best_hyperparameters_SM.pkl"

FEATURE_NAMES = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke", "HeartIssue",
    "PhysActivity", "Fruits", "Veggies", "AlcoholConsump", "AnyHealthcare",
    "MedCost", "GenHlth", "MentHlth", "PhysHlth", "DiffWalk", "Sex",
    "AgeBracket", "Education", "Income"
]

# --- Caching model/hyperparam/scaler loading ---
@st.cache_resource
def load_model_and_scaler_and_hyper(model_name):
    model = joblib.load(MODEL_PATHS[model_name])
    scaler = joblib.load(SCALER_PATH)
    hyper_all = joblib.load(HYPERPARAMS_PATH)
    # Hyperparameter keys are typically like 'RandomForest', 'LogisticRegression', 'XGBoost'
    key_map = {"Random Forest": "RandomForest", "Logistic Regression": "LogisticRegression", "XGBoost": "XGBoost"}
    hyper = hyper_all[key_map[model_name]]
    return model, scaler, hyper

# --- UI Feature input with clear clinical context ---
def collect_user_features():
    st.header("Enter Patient Information")
    cols1, cols2 = st.columns(2)
    with cols1:
        HighBP = st.radio(
            "Ever diagnosed with high blood pressure?",
            ["No", "Yes"], help="By a doctor, nurse, or other healthcare professional."
        )
        HighChol = st.radio(
            "Ever diagnosed with high cholesterol?",
            ["No", "Yes"]
        )
        CholCheck = st.radio(
            "Had cholesterol checked in the last 5 years?",
            ["No", "Yes"]
        )
        BMI = st.number_input(
            "Body Mass Index (BMI)", 10.0, 70.0, 25.0, format="%.2f",
            help="Weight(kg)/[height(m)]². If not sure, check with your healthcare provider."
        )
        Smoker = st.radio(
            "Smoked at least 100 cigarettes in your life?",
            ["No", "Yes"],
            help="Even if no longer smoking."
        )
        Stroke = st.radio(
            "Ever had a stroke (doctor diagnosis)?",
            ["No", "Yes"]
        )
        HeartIssue = st.radio(
            "Coronary heart disease or heart attack ever diagnosed?",
            ["No", "Yes"]
        )
        PhysActivity = st.radio(
            "Any exercise/physical activity in past 30 days (besides regular job)?",
            ["No", "Yes"]
        )
        Fruits = st.radio(
            "Eat fruit one or more times per day?",
            ["No", "Yes"]
        )
        Veggies = st.radio(
            "Eat veggies one or more times per day?",
            ["No", "Yes"]
        )
        AlcoholConsump = st.radio(
            "Currently a heavy drinker (male: >14/wk, female: >7/wk)?",
            ["No", "Yes"]
        )
    with cols2:
        AnyHealthcare = st.radio(
            "Have any kind of health care coverage?",
            ["No", "Yes"]
        )
        MedCost = st.radio(
            "Did cost prevent doctor visit in past year?",
            ["No", "Yes"]
        )
        GenHlth = st.selectbox(
            "How would you rate your general health?",
            [1, 2, 3, 4, 5],
            format_func=lambda x: ["Excellent", "Very good", "Good", "Fair", "Poor"][x-1]
        )
        MentHlth = st.number_input(
            "Number of days mental health was not good in last 30 days",
            min_value=0, max_value=30, value=0
        )
        PhysHlth = st.number_input(
            "Number of days physical health not good in last 30 days",
            min_value=0, max_value=30, value=0
        )
        DiffWalk = st.radio(
            "Serious difficulty walking/climbing stairs?",
            ["No", "Yes"]
        )
        Sex = st.radio(
            "Sex assigned at birth",
            ["Female", "Male"]
        )
        AgeBracket = st.selectbox(
            "What is your age group?",
            list(range(1, 14)),
            format_func=lambda x: [
                "18–24", "25–29", "30–34", "35–39", "40–44", "45–49", "50–54",
                "55–59", "60–64", "65–69", "70–74", "75–79", "80 or older"][x-1]
        )
        Education = st.selectbox(
            "Highest education completed",
            [1,2,3,4,5,6],
            format_func=lambda x: {
                1:"Elementary (1–8)", 2:"Some high school", 3:"High school grad",
                4:"Some college/tech", 5:"College grad (2yr)", 6:"College grad (4yr)"
            }[x]
        )
        Income = st.selectbox(
            "Approximate annual household income",
            [1,2,3,4,5,6,7,8],
            format_func=lambda x: {
                1:"< $10,000", 2:"$10,000–<$15,000", 3:"$15,000–<$20,000",
                4:"$20,000–<$25,000", 5:"$25,000–<$35,000", 6:"$35,000–<$50,000",
                7:"$50,000–<$75,000", 8:"$75,000 or more"
            }[x]
        )
    # Mapping radio/box values to model input numeric coding
    return np.array([
        1 if HighBP=="Yes" else 0,
        1 if HighChol=="Yes" else 0,
        1 if CholCheck=="Yes" else 0,
        BMI,
        1 if Smoker=="Yes" else 0,
        1 if Stroke=="Yes" else 0,
        1 if HeartIssue=="Yes" else 0,
        1 if PhysActivity=="Yes" else 0,
        1 if Fruits=="Yes" else 0,
        1 if Veggies=="Yes" else 0,
        1 if AlcoholConsump=="Yes" else 0,
        1 if AnyHealthcare=="Yes" else 0,
        1 if MedCost=="Yes" else 0,
        GenHlth,
        MentHlth,
        PhysHlth,
        1 if DiffWalk=="Yes" else 0,
        1 if Sex=="Male" else 0,
        AgeBracket,
        Education,
        Income
    ]).reshape(1, -1)

# --- App main UI ---
st.set_page_config(page_title="Diabetes Risk Prediction (BRFSS)", layout="centered")
st.title("Diabetes Risk Prediction App")

st.info("Answer all questions. When ready, select a model and click Predict to view risk results and feature explanations.")

with st.sidebar:
    model_option = st.selectbox(
        "Choose a model for prediction",
        ["Random Forest", "Logistic Regression", "XGBoost"],
        help="Choose from three established ML models."
    )

model, scaler, hyperparams = load_model_and_scaler_and_hyper(model_option)

X_input = collect_user_features()

if st.button("Predict Diabetes Risk"):
    # --- Model inference pipeline ---
    X_scaled = scaler.transform(X_input)  # StandardScaler means you must scale!
    pred = model.predict(X_scaled)[0]
    prob = model.predict_proba(X_scaled)[0][1]

    model_names_friendly = {
        "Random Forest": "Random Forest (ensemble tree-based model)",
        "Logistic Regression": "Logistic Regression (statistical probability model)",
        "XGBoost": "Extreme Gradient Boosting (boosted tree model)"
    }
    st.subheader("Prediction Results")
    st.markdown(f"**Model used:** `{model_names_friendly[model_option]}`")
    st.markdown("**Model hyperparameters used for this prediction:**")
    st.code(
        "\n".join([f"{k}: {v}" for k, v in hyperparams.items()]),
        language="yaml"
    )

    st.markdown(
        f"""
**Outcome:** {"🔴 Positive diabetes risk" if pred==1 else "🟢 Negative diabetes risk"}  
**Estimated probability patient would be classified as diabetic:** `{prob:.2%}`
"""
    )

    st.divider()
    st.subheader("Most Influential Features for This Prediction")
    # --- Feature importance for current prediction (patient) ---
    try:
        # Use SHAP for per-patient/top feature contribution
        if model_option == "Logistic Regression":
            explainer = shap.LinearExplainer(model, X_scaled)
        else:
            explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_scaled)
        # Take correct class if tree model shap returns a list
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        contrib = pd.Series(shap_values[0], index=FEATURE_NAMES)
        top_feats = contrib.abs().sort_values(ascending=False).head(5)

        for fname in top_feats.index:
            effect = "raises risk" if contrib[fname]>0 else "lowers risk"
            st.markdown(f"- **{fname}**: {effect} ({contrib[fname]:+.2f} contribution)")
        st.caption("Computed using SHAP for this individual input. Positive values mean higher risk.")
    except Exception as ex:
        st.warning("Could not compute local feature importances (SHAP).")
        st.text(f"Error detail: {ex}")

    st.divider()
    st.caption(
        """
**Interpretation:**  
Above results are machine-generated risk estimates using a validated ML model trained on a large U.S. population health dataset (BRFSS, with SMOTE+ENN balancing).  
Highest-influence features above describe why the model classified this patient as at risk or not, *with signs indicating which way each factor pushes the result*.  
This risk score is for education and support only—it does not replace a provider diagnosis or lab testing.
"""
    )

else:
    st.info("Complete all required fields and click Predict to view results.")


# In[ ]:




