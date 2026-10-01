# ⚡ AI-Based Smart Energy Consumption Forecasting and Anomaly Detection System

## 📌 Project Overview

This project is a machine learning based smart energy monitoring system.

It predicts household energy consumption and detects unusual energy consumption patterns using machine learning.

The system provides:

- Energy consumption prediction
- Anomaly detection
- Real-time simulation
- Interactive dashboard
- Prediction error monitoring
- Historical data analysis
- SQLite prediction history
- CSV data export

## 🧠 Machine Learning Models

### 1. Energy Forecasting

**XGBoost Regressor** is used to predict household energy consumption.

The model uses time-based and historical energy features such as:

- Hour
- Day
- Month
- Day of Week
- Previous Hour Energy
- Previous 24 Hour Energy

### 2. Anomaly Detection

**Isolation Forest** is used to detect unusual energy consumption patterns.

It identifies readings that differ significantly from normal consumption behavior.

## 📊 Dataset

The project uses the **UCI Individual Household Electric Power Consumption Dataset**.

The dataset contains household electricity measurements collected over time.

The current system uses historical data and replays selected readings to simulate a real-time monitoring environment.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn
- Plotly
- Flask
- Streamlit
- SQLite
- Joblib
- Requests
- Git
- GitHub

## 📂 Project Structure

```text
smart-energy-ml/
│
├── README.md
├── .gitignore
├── requirements.txt
│
└── data/
    ├── app.py
    ├── dashboard.py
    ├── database.py
    ├── live_simulator.py
    ├── energy_model.pkl
    ├── anomaly_model.pkl
    │
    └── src/
        ├── preprocess.py
        ├── eda.py
        ├── features.py
        ├── train.py
        └── anomaly.py