
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --------------------------------
# Page Configuration
# --------------------------------

st.set_page_config(
    page_title="Smart Energy Monitoring",
    page_icon="⚡",
    layout="wide"
)

API_URL = "https://smart-energy-ml.onrender.com/history"

# --------------------------------
# Auto Refresh
# --------------------------------

st_autorefresh(
    interval=5000,
    key="energy_refresh"
)

# --------------------------------
# Fetch Data from Render API
# --------------------------------

@st.cache_data(ttl=4)
def fetch_history():
    response = requests.get(API_URL, timeout=60)
    response.raise_for_status()
    return response.json()

# --------------------------------
# Dashboard Header
# --------------------------------

st.title("⚡ Smart Energy Consumption Monitoring")
st.caption("AI-Based Energy Forecasting and Anomaly Detection")
st.divider()

# --------------------------------
# Load History
# --------------------------------

try:
    records = fetch_history()

except requests.exceptions.RequestException as e:
    st.error("Unable to connect to the Render API.")
    st.write("Please check whether the Render service is running.")
    st.code(str(e))
    st.stop()

if not records:
    st.warning("No prediction data received yet.")
    st.info(
        "Run live_simulator.py on your computer "
        "to send energy readings to the API."
    )
    st.stop()

df = pd.DataFrame(records)

# --------------------------------
# Prepare Data
# --------------------------------

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce"
)

df["actual_energy"] = pd.to_numeric(
    df["actual_energy"],
    errors="coerce"
)

df["predicted_energy"] = pd.to_numeric(
    df["predicted_energy"],
    errors="coerce"
)

df = df.dropna(
    subset=[
        "timestamp",
        "actual_energy",
        "predicted_energy"
    ]
)

df = df.sort_values("timestamp").reset_index(drop=True)

df["error"] = (
    df["actual_energy"] - df["predicted_energy"]
)

df["absolute_error"] = df["error"].abs()

if df.empty:
    st.warning("No valid prediction records are available.")
    st.stop()

latest = df.iloc[-1]

# --------------------------------
# System Status
# --------------------------------

st.subheader("🟢 System Status")

if latest["status"] == "Anomaly":
    st.error("Anomaly Detected in Latest Reading")
else:
    st.success("System Operating Normally")

st.caption(
    f"Latest reading: {latest['timestamp']}"
)

# --------------------------------
# Key Metrics
# --------------------------------

st.subheader("📊 Energy Overview")

total_readings = len(df)

anomaly_count = int(
    (df["status"] == "Anomaly").sum()
)

average_energy = df["actual_energy"].mean()

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Current Energy",
    f"{latest['actual_energy']:.3f} kW"
)

col2.metric(
    "Predicted Energy",
    f"{latest['predicted_energy']:.3f} kW"
)

col3.metric(
    "Average Energy",
    f"{average_energy:.3f} kW"
)

col4.metric(
    "Anomalies Detected",
    anomaly_count
)

st.caption(f"Total readings received: {total_readings}")

st.divider()

# --------------------------------
# Model Performance
# --------------------------------

st.subheader("🎯 Live Prediction Performance")

if len(df) >= 2:
    mae = mean_absolute_error(
        df["actual_energy"],
        df["predicted_energy"]
    )

    rmse = np.sqrt(
        mean_squared_error(
            df["actual_energy"],
            df["predicted_energy"]
        )
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("MAE", f"{mae:.4f}")
    col2.metric("RMSE", f"{rmse:.4f}")

    if df["actual_energy"].nunique() > 1:
        r2 = r2_score(
            df["actual_energy"],
            df["predicted_energy"]
        )
        col3.metric("R² Score", f"{r2:.4f}")
    else:
        col3.metric("R² Score", "N/A")
else:
    st.info("More readings are needed to calculate live performance metrics.")

st.caption(
    "Metrics are calculated from the prediction records received so far, "
    "not from a separate test dataset."
)

st.divider()

# --------------------------------
# Actual vs Predicted Graph
# --------------------------------

st.subheader("📈 Actual vs Predicted Energy")

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=df["timestamp"],
        y=df["actual_energy"],
        mode="lines",
        name="Actual Energy"
    )
)

fig.add_trace(
    go.Scatter(
        x=df["timestamp"],
        y=df["predicted_energy"],
        mode="lines",
        name="Predicted Energy"
    )
)

anomalies = df[df["status"] == "Anomaly"]

if not anomalies.empty:
    fig.add_trace(
        go.Scatter(
            x=anomalies["timestamp"],
            y=anomalies["actual_energy"],
            mode="markers",
            name="Anomaly",
            marker=dict(
                symbol="x",
                size=10
            )
        )
    )

fig.update_layout(
    xaxis_title="Time",
    yaxis_title="Energy (kW)",
    hovermode="x unified",
    legend_title="Readings",
    height=450
)

st.plotly_chart(fig, use_container_width=True)

# --------------------------------
# Prediction Error Graph
# --------------------------------

st.subheader("📉 Prediction Error")

error_fig = go.Figure()

error_fig.add_trace(
    go.Scatter(
        x=df["timestamp"],
        y=df["error"],
        mode="lines",
        name="Prediction Error"
    )
)

error_fig.add_hline(
    y=0,
    line_dash="dash"
)

error_fig.update_layout(
    xaxis_title="Time",
    yaxis_title="Error (kW)",
    height=350
)

st.plotly_chart(error_fig, use_container_width=True)

st.divider()

# --------------------------------
# Anomaly Details
# --------------------------------

st.subheader("🚨 Anomaly Detection Results")

if not anomalies.empty:
    st.dataframe(
        anomalies[
            [
                "timestamp",
                "actual_energy",
                "predicted_energy",
                "status"
            ]
        ].sort_values(
            "timestamp",
            ascending=False
        ),
        use_container_width=True,
        hide_index=True
    )
else:
    st.success("No anomalies have been detected in the received readings.")

# --------------------------------
# Energy Summary
# --------------------------------

st.subheader("📋 Energy Summary")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Maximum Energy",
    f"{df['actual_energy'].max():.3f} kW"
)

col2.metric(
    "Minimum Energy",
    f"{df['actual_energy'].min():.3f} kW"
)

col3.metric(
    "Anomaly Rate",
    f"{(anomaly_count / total_readings) * 100:.2f}%"
)

st.divider()

# --------------------------------
# All Readings
# --------------------------------

st.subheader("📑 Complete Energy History")

st.dataframe(
    df[
        [
            "id",
            "timestamp",
            "actual_energy",
            "predicted_energy",
            "status",
            "error"
        ]
    ].sort_values(
        "timestamp",
        ascending=False
    ),
    use_container_width=True,
    hide_index=True
)

# --------------------------------
# Download CSV
# --------------------------------

csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="📥 Download Energy History CSV",
    data=csv,
    file_name="energy_history.csv",
    mime="text/csv"
)

# --------------------------------
# About
# --------------------------------

st.divider()

st.subheader("ℹ️ About This Project")

st.write(
    """
    This project uses machine learning to forecast household
    energy consumption and detect unusual energy usage.

    - XGBoost Regressor: predicts energy consumption.
    - Isolation Forest: detects anomalous readings.
    - Flask API on Render: receives predictions and stores history.
    - Streamlit: displays the monitoring dashboard.

    The real-time view is a simulation that replays historical
    energy readings at regular intervals. It is not a live
    smart-meter connection.
    """
)

st.caption("Developed by Tanvi Nagar | B.Tech CSE")