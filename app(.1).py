import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ------------------------ LOAD MODEL & DATA ------------------------
model = joblib.load("electricity_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Load dataset
df = pd.read_csv("cost.csv")
df["structure type"] = label_encoder.inverse_transform(label_encoder.transform(df["structure type"]))

# Streamlit Page Config
st.set_page_config(page_title="Electricity Cost Dashboard", layout="wide")
st.title("⚡ Electricity Cost Prediction Dashboard")
st.markdown("An **interactive machine learning dashboard** for predicting monthly electricity costs.")

# Sidebar Navigation (Removed "Data Insights")
menu = st.sidebar.radio(
    "📌 Select Page",
    ["Prediction", "Bulk Prediction", "Model Performance"]
)

# ------------------------ PAGE 1: SINGLE PREDICTION ------------------------
if menu == "Prediction":
    st.header("🔹 Predict Monthly Electricity Cost")

    col1, col2, col3 = st.columns(3)

    with col1:
        site_area = st.number_input("Site Area (sq. meters)", min_value=0)
        water_consumption = st.number_input("Water Consumption (liters/day)", min_value=0)
        utilisation_rate = st.slider("Utilisation Rate (%)", 0, 100, 50)
    with col2:
        structure_type = st.selectbox("Structure Type", ["Residential", "Commercial", "Mixed-use", "Industrial"])
        recycling_rate = st.slider("Recycling Rate (%)", 0, 100, 50)
        air_quality_index = st.number_input("Air Quality Index (AQI)", min_value=0)
    with col3:
        resident_count = st.number_input("Resident Count", min_value=0)

    if st.button("🚀 Predict Electricity Cost"):
        structure_encoded = label_encoder.transform([structure_type])[0]
        input_data = np.array([[site_area, structure_encoded, water_consumption, recycling_rate,
                                utilisation_rate, air_quality_index, resident_count]])
        prediction = model.predict(input_data)
        st.success(f"💡 Predicted Monthly Electricity Cost: **${prediction[0]:,.2f}**")

# ------------------------ PAGE 2: BULK PREDICTION ------------------------
elif menu == "Bulk Prediction":
    st.header("📂 Upload CSV for Bulk Predictions")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)

        # Encode structure type column if present
        if "structure type" in input_df.columns:
            input_df["structure type"] = label_encoder.transform(input_df["structure type"])

        # Predict electricity cost
        predictions = model.predict(input_df)
        input_df["Predicted Electricity Cost"] = predictions

        st.success("✅ Bulk predictions completed!")
        st.dataframe(input_df)

        # Download option
        csv = input_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Predictions", csv, "bulk_predictions.csv", "text/csv")

# ------------------------ PAGE 3: MODEL PERFORMANCE ------------------------
elif menu == "Model Performance":
    st.header("📈 Model Performance Metrics")

    # Separate features & target
    X = df.drop(columns=["electricity cost"])
    X["structure type"] = label_encoder.transform(X["structure type"])
    y = df["electricity cost"]

    # Predictions
    y_pred = model.predict(X)

    # Metrics
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)

    col1, col2, col3 = st.columns(3)
    col1.metric("R² Score", f"{r2:.3f}")
    col2.metric("Mean Squared Error", f"{mse:,.2f}")
    col3.metric("Mean Absolute Error", f"{mae:,.2f}")

    # Feature Importance Visualization
    st.subheader("🔹 Feature Importance")
    try:
        feature_importances = model.feature_importances_
        features = X.columns
        importance_df = pd.DataFrame({"Feature": features, "Importance": feature_importances})
        importance_df = importance_df.sort_values(by="Importance", ascending=False)

        fig = px.bar(importance_df, x="Feature", y="Importance", color="Importance", title="Feature Importance")
        st.plotly_chart(fig)
    except:
        st.warning("⚠️ Feature importance not available for this model.")
