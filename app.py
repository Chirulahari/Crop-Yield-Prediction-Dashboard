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
st.title("🌾 ADVANCED CROP YIELD PREDICTION DASHBOARD")

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

    # Store the categories BEFORE one-hot encoding so we can use them for input building
    cat_categories = {c: sorted(df[c].unique().tolist()) for c in cat_cols}

    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    X = df.drop(columns=['Crop_Yield_MT_per_HA'])
    y = df['Crop_Yield_MT_per_HA']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # =========================
    # TRAIN MODEL (cached so sidebar changes don't retrain)
    # =========================
    @st.cache_resource
    def train_models(_X_train, _y_train):
        quantiles = [0.1, 0.5, 0.9]
        models = {}
        for q in quantiles:
            model = lgb.LGBMRegressor(
                objective='quantile',
                alpha=q,
                n_estimators=400,
                learning_rate=0.05
            )
            model.fit(_X_train, _y_train)
            models[q] = model
        return models

    with st.spinner("Training models..."):
        models = train_models(X_train, y_train)
    st.success("Models Trained Successfully!")

    preds = {q: models[q].predict(X_test) for q in [0.1, 0.5, 0.9]}
    y_lo, y_med, y_hi = preds[0.1], preds[0.5], preds[0.9]

    # =========================
    # METRICS
    # =========================
    rmse = np.sqrt(mean_squared_error(y_test, y_med))
    r2 = r2_score(y_test, y_med)
    picp = np.mean((y_test >= y_lo) & (y_test <= y_hi))
    sharpness = np.mean(y_hi - y_lo)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("RMSE", round(rmse, 3))
    col2.metric("R² Score", round(r2, 3))
    col3.metric("Coverage %", round(picp * 100, 2))
    col4.metric("Interval Width", round(sharpness, 3))

    # =========================
    # SIDEBAR INPUTS
    # =========================
    st.sidebar.header("🌱 Input Controls")

    input_data = {}

    # ---------- NUMERICAL ----------
    numeric_inputs = [
        'Year',
        'Average_Temperature_C',
        'Total_Precipitation_mm',
        'CO2_Emissions_MT',
        'Extreme_Weather_Events',
        'Irrigation_Access_%',
        'Pesticide_Use_KG_per_HA',
        'Fertilizer_Use_KG_per_HA',
        'Soil_Health_Index',
        'Economic_Impact_Million_USD',
    ]

    for col in numeric_inputs:
        if col in df_original.columns:
            input_data[col] = st.sidebar.slider(
                col,
                float(df_original[col].min()),
                float(df_original[col].max()),
                float(df_original[col].mean())
            )

    # ---------- CATEGORICAL ----------
    # Store selected value per category column
    cat_selected = {}   # e.g. {'Crop_Type': 'Rice', 'Region': 'West Bengal', ...}

    for col in cat_cols:
        options = cat_categories[col]
        cat_selected[col] = st.sidebar.selectbox(col, options)

    # =========================
    # CREATE FULL INPUT — FIX IS HERE
    # =========================
    # Start with zeros for all one-hot columns, median for numeric
    input_full = {}

    for col in X.columns:
        # Identify if this column is a one-hot dummy (contains an underscore that
        # matches one of the original categorical columns)
        is_dummy = any(col.startswith(f"{c}_") for c in cat_cols)
        input_full[col] = 0.0 if is_dummy else float(X[col].median())

    # Override numeric inputs from sliders
    for k, v in input_data.items():
        if k in input_full:
            input_full[k] = v

    # Override categorical dummies correctly:
    # For each categorical column, set the correct dummy to 1 (and leave all others 0)
    # Note: drop_first=True drops the first (alphabetically sorted) category per column.
    for col, selected_val in cat_selected.items():
        sorted_cats = cat_categories[col]   # same sort order pandas used
        dropped_cat = sorted_cats[0]        # drop_first drops the first sorted category

        if selected_val == dropped_cat:
            # This is the reference category — all dummies for this column stay 0
            pass
        else:
            dummy_col = f"{col}_{selected_val}"
            if dummy_col in input_full:
                input_full[dummy_col] = 1.0

    input_df = pd.DataFrame([input_full])
    input_df = input_df[X.columns]

    # =========================
    # PREDICTION
    # =========================
    pred = models[0.5].predict(input_df)[0]
    lo = models[0.1].predict(input_df)[0]
    hi = models[0.9].predict(input_df)[0]

    st.subheader("📈 Real-Time Yield Prediction")
    st.write(f"### 🌾 {round(pred, 2)} MT/ha")
    st.write(f"Range: {round(lo, 2)} — {round(hi, 2)}")

    # =========================
    # VISUALS
    # =========================
    st.subheader("📊 Visualizations")

    fig1 = px.scatter(x=y_med, y=y_test, title="Actual vs Predicted",
                      labels={"x": "Predicted", "y": "Actual"})
    st.plotly_chart(fig1, use_container_width=True)

    residuals = y_test - y_med

    fig2 = px.histogram(residuals, title="Residual Distribution")
    st.plotly_chart(fig2, use_container_width=True)

    x_vals = np.linspace(min(residuals), max(residuals), 100)

    fig3 = px.line(
        x=x_vals,
        y=gev.pdf(x_vals, *gev.fit(residuals)),
        title="GEV Tail Risk"
    )
    st.plotly_chart(fig3, use_container_width=True)

    # =========================
    # SHAP
    # =========================
    st.subheader("🔍 SHAP Explanation")

    explainer = shap.TreeExplainer(models[0.5])
    shap_values = explainer.shap_values(input_df)

    shap.plots._waterfall.waterfall_legacy(
        explainer.expected_value,
        shap_values[0],
        feature_names=input_df.columns
    )

    st.pyplot(plt.gcf())
    plt.clf()
