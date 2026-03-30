import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import genextreme as gev
from properscoring import crps_ensemble

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(layout="wide")
st.title("🌾 FULL INTEGRATED LIVE DASHBOARD (ALL IN ONE)")

# =========================
# FILE UPLOAD
# =========================
uploaded_file = st.file_uploader("Upload your dataset (CSV)")

if uploaded_file:

    df_original = pd.read_csv(uploaded_file)

    st.subheader("📊 Dataset Preview")
    st.dataframe(df_original.head())

    # =========================
    # PREPROCESSING
    # =========================
    df = df_original.copy()
    df = df.dropna(subset=['Crop_Yield_MT_per_HA'])

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())

    for c in cat_cols:
        df[c] = df[c].fillna(df[c].mode()[0])

    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    X = df.drop(columns=['Crop_Yield_MT_per_HA'])
    y = df['Crop_Yield_MT_per_HA']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # =========================
    # TRAIN MODEL
    # =========================
    st.info("Training models...")

    quantiles = [0.1, 0.5, 0.9]
    models, preds = {}, {}

    for q in quantiles:
        model = lgb.LGBMRegressor(
            objective='quantile',
            alpha=q,
            n_estimators=300
        )
        model.fit(X_train, y_train)
        preds[q] = model.predict(X_test)
        models[q] = model

    y_lo, y_med, y_hi = preds[0.1], preds[0.5], preds[0.9]

    st.success("Models Trained Successfully!")

    # =========================
    # METRICS
    # =========================
    rmse = np.sqrt(mean_squared_error(y_test, y_med))
    r2 = r2_score(y_test, y_med)
    picp = np.mean((y_test >= y_lo) & (y_test <= y_hi))
    sharpness = np.mean(y_hi - y_lo)
    crps_score = np.mean(
        crps_ensemble(y_test.values, np.vstack([y_lo, y_med, y_hi]).T)
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("RMSE", round(rmse, 3))
    col2.metric("R²", round(r2, 3))
    col3.metric("Coverage %", round(picp * 100, 2))
    col4.metric("Interval Width", round(sharpness, 3))

    # =========================
    # SIDEBAR INPUTS
    # =========================
    st.sidebar.header("🌱 Interactive Input Controls")

    important_features = [
        'Year',
        'Average_Temperature_C',
        'Total_Precipitation_mm',
        'CO2_Emissions_MT',
        'Extreme_Weather_Events',
        'Irrigation_Access_%',
        'Pesticide_Use_KG_per_HA',
        'Fertilizer_Use_KG_per_HA'
    ]

    input_data = {}

    for col in important_features:
        if col in df_original.columns:
            val = float(df_original[col].mean())
            input_data[col] = st.sidebar.slider(
                col,
                float(df_original[col].min()),
                float(df_original[col].max()),
                val
            )

    # =========================
    # FIX FEATURE MISMATCH
    # =========================
    input_full = {}

    for col in X.columns:
        if col in input_data:
            input_full[col] = input_data[col]
        else:
            input_full[col] = 0  # for dummy variables

    input_df = pd.DataFrame([input_full])
    input_df = input_df[X.columns]

    # =========================
    # PREDICTION
    # =========================
    pred = models[0.5].predict(input_df)[0]
    lo = models[0.1].predict(input_df)[0]
    hi = models[0.9].predict(input_df)[0]

    st.subheader("📈 Real-Time Yield Prediction")
    st.write(f"### 🌾 {round(pred,2)} MT/ha")
    st.write(f"Uncertainty Range: {round(lo,2)} — {round(hi,2)}")

    # =========================
    # VISUALS
    # =========================
    st.subheader("📊 All Visuals")

    fig1 = px.scatter(x=y_med, y=y_test,
                      labels={'x': 'Predicted', 'y': 'Observed'},
                      title="Actual vs Predicted")
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.bar(x=["PICP", "Width"], y=[picp, sharpness],
                  title="Coverage vs Interval Width")
    st.plotly_chart(fig2, use_container_width=True)

    residuals = y_test - y_med
    fig3 = px.histogram(residuals, nbins=30,
                        title="Residual Distribution")
    st.plotly_chart(fig3, use_container_width=True)

    x_vals = np.linspace(min(residuals), max(residuals), 100)
    fig4 = px.line(
        x=x_vals,
        y=gev.pdf(x_vals, *gev.fit(residuals)),
        title="GEV Tail Fit"
    )
    st.plotly_chart(fig4, use_container_width=True)

    # =========================
    # SHAP EXPLANATION
    # =========================
    st.subheader("🔍 SHAP Explanation")

    explainer = shap.Explainer(models[0.5])
    shap_values = explainer(input_df)

    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(plt.gcf())
    plt.clf()