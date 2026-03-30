# 🌾 ML-Powered Crop Yield Prediction & Risk Analysis Dashboard  

![Made with Python](https://img.shields.io/badge/Made%20with-Python-blue?logo=python)  
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red?logo=streamlit)  
![LightGBM](https://img.shields.io/badge/Model-LightGBM-green)  

An end-to-end **Machine Learning + Interactive Dashboard Application** that predicts crop yield and analyzes agricultural risk using climate, environmental, and farming factors.

The system uses **Quantile LightGBM models** for probabilistic prediction and provides **real-time insights, uncertainty estimation, and SHAP-based explainability** through an interactive **Streamlit dashboard**.


## 📖 Overview
This project provides a **data-driven solution for crop yield prediction** using machine learning techniques.

It uses environmental and agricultural features such as:

- 🌡️ Temperature  
- 🌧️ Rainfall  
- 🌿 Fertilizer & Pesticide Usage  
- 🌍 CO₂ Emissions  
- 🌪️ Extreme Weather Events  
- 💧 Irrigation Access  

The model predicts:
- ✅ Crop Yield (MT/ha)  
- 📊 Prediction Interval (Uncertainty Range)  
- ⚠️ Risk Metrics (PICP, CRPS, Sharpness)  

The dashboard enables **real-time interaction and visualization** for better decision-making.

---

## 💡 Motivation
Agriculture is highly affected by **climate change and environmental factors**.

This project aims to:
- Provide **accurate yield predictions**
- Analyze **risk and uncertainty**
- Help farmers and policymakers make **data-driven decisions**

It is designed as a complete **end-to-end ML system**, from data processing to deployment-ready dashboard.

---

## 📂 Project Structure
```bash
Crop-Yield-Prediction-Dashboard/
│
├── app.py
├── climate_change_impact_on_agriculture_2024.csv
├── requirements.txt
├── README.md
├── .gitignore
```

---

## 🛠 Technologies Used

- **Python (pandas, numpy, scikit-learn)** → Data processing  
- **LightGBM** → Quantile regression model  
- **Streamlit** → Interactive dashboard  
- **Plotly** → Interactive visualizations  
- **SHAP** → Model explainability  
- **SciPy** → Statistical modeling (GEV distribution)  
- **Proper Scoring (CRPS)** → Probabilistic evaluation  

---

## 🚀 Features

### 🔹 Machine Learning
- Quantile Regression using LightGBM  
- Predicts median + uncertainty intervals  
- Handles real-world noisy data  

### 🔹 Risk Analysis
- 📉 RMSE & R² Score  
- 📊 PICP (Prediction Interval Coverage Probability)  
- 📏 Sharpness (interval width)  
- 📉 CRPS (probabilistic accuracy)  

### 🔹 Interactive Dashboard
- Real-time predictions using sliders  
- Dataset preview  
- Single-page UI  

### 🔹 Visualization
- Actual vs Predicted Plot  
- Residual Distribution  
- Coverage vs Width  
- Tail Risk Modeling (GEV)  

### 🔹 Explainability
- SHAP Waterfall Plot  
- Feature contribution analysis  

---

## ⚙️ Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/Crop-Yield-Prediction-Dashboard.git
cd Crop-Yield-Prediction-Dashboard
```
### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
### 3️⃣ Run the Application
```bash
streamlit run app.py
```
