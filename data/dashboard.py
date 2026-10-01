import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
from database import get_predictions


# =================================================
# PAGE CONFIGURATION
# =================================================

st.set_page_config(
    page_title="Smart Energy Monitoring",
    page_icon="⚡",
    layout="wide"
)


# =================================================
# AUTO REFRESH
# =================================================

st_autorefresh(
    interval=5000,
    key="energy_refresh"
)


# =================================================
# TITLE
# =================================================

st.title("⚡ Smart Energy Consumption Monitoring")

st.write(
    "Real-Time Energy Forecasting and Anomaly Detection System"
)

st.caption(
    "ML-powered household energy monitoring dashboard"
)

st.divider()


# =================================================
# LOAD DATABASE
# =================================================

try:

    history = get_predictions()

except Exception as e:

    st.error(f"Database error: {e}")
    st.stop()


if not history:

    st.warning(
        "Waiting for incoming energy readings..."
    )

    st.info(
        "Please make sure Flask and live_simulator.py "
        "are running."
    )

    st.stop()


# =================================================
# CREATE DATAFRAME
# =================================================

realtime_df = pd.DataFrame(
    history,
    columns=[
        "ID",
        "Timestamp",
        "Actual Energy",
        "Predicted Energy",
        "Status"
    ]
)


# =================================================
# DATA CLEANING
# =================================================

realtime_df["Timestamp"] = pd.to_datetime(
    realtime_df["Timestamp"]
)

realtime_df = realtime_df.sort_values(
    "Timestamp"
).reset_index(drop=True)


# =================================================
# PREDICTION ERROR
# =================================================

realtime_df["Prediction Error"] = (
    realtime_df["Actual Energy"]
    - realtime_df["Predicted Energy"]
).abs()


# =================================================
# SIDEBAR CONTROLS
# =================================================

st.sidebar.title("⚙️ Dashboard Controls")


# -------------------------------------------------
# READING RANGE
# -------------------------------------------------

st.sidebar.subheader("📊 Reading Range")

time_range = st.sidebar.selectbox(
    "Select Range",
    [
        "All Data",
        "Last 10 Readings",
        "Last 25 Readings",
        "Last 50 Readings"
    ]
)


# -------------------------------------------------
# DATE FILTER
# -------------------------------------------------

st.sidebar.subheader("📅 Date Filter")

min_date = realtime_df[
    "Timestamp"
].min().date()

max_date = realtime_df[
    "Timestamp"
].max().date()


start_date = st.sidebar.date_input(
    "Start Date",
    value=min_date,
    min_value=min_date,
    max_value=max_date
)


end_date = st.sidebar.date_input(
    "End Date",
    value=max_date,
    min_value=min_date,
    max_value=max_date
)


if start_date > end_date:

    st.sidebar.error(
        "Start Date cannot be after End Date."
    )

    st.stop()


# =================================================
# APPLY DATE FILTER
# =================================================

filtered_df = realtime_df[
    (realtime_df["Timestamp"].dt.date >= start_date)
    &
    (realtime_df["Timestamp"].dt.date <= end_date)
].copy()


if filtered_df.empty:

    st.warning(
        "No energy readings found for the selected dates."
    )

    st.stop()


# =================================================
# APPLY READING RANGE
# =================================================

if time_range == "Last 10 Readings":

    display_df = filtered_df.tail(10).copy()

elif time_range == "Last 25 Readings":

    display_df = filtered_df.tail(25).copy()

elif time_range == "Last 50 Readings":

    display_df = filtered_df.tail(50).copy()

else:

    display_df = filtered_df.copy()


# =================================================
# LATEST READING
# =================================================

latest = realtime_df.iloc[-1]

latest_time = latest["Timestamp"]

latest_actual = latest["Actual Energy"]

latest_predicted = latest["Predicted Energy"]

latest_status = latest["Status"]


# =================================================
# LIVE SYSTEM STATUS
# =================================================

st.header("🔴 Live System Status")


if latest_status == "Anomaly":

    st.error(
        f"🔴 ANOMALY DETECTED  |  "
        f"Last Update: "
        f"{latest_time.strftime('%Y-%m-%d %H:%M:%S')}"
    )

else:

    st.success(
        f"🟢 SYSTEM NORMAL  |  "
        f"Last Update: "
        f"{latest_time.strftime('%Y-%m-%d %H:%M:%S')}"
    )


st.divider()


# =================================================
# DASHBOARD METRICS
# =================================================

total_readings = len(realtime_df)


anomaly_count = int(
    (realtime_df["Status"] == "Anomaly").sum()
)


normal_count = (
    total_readings
    - anomaly_count
)


average_energy = realtime_df[
    "Actual Energy"
].mean()


current_error = abs(
    latest_actual
    - latest_predicted
)


col1, col2, col3, col4 = st.columns(4)


col1.metric(
    "⚡ Current Energy",
    f"{latest_actual:.2f} kW"
)


col2.metric(
    "🔮 Predicted Energy",
    f"{latest_predicted:.2f} kW"
)


col3.metric(
    "📊 Average Energy",
    f"{average_energy:.2f} kW"
)


col4.metric(
    "🚨 Anomalies",
    anomaly_count
)


st.divider()


# =================================================
# ACTUAL VS PREDICTED WITH ANOMALIES
# =================================================

st.header("📈 Actual vs Predicted Energy")


fig = go.Figure()


# -------------------------------------------------
# ACTUAL ENERGY
# -------------------------------------------------

fig.add_trace(
    go.Scatter(
        x=display_df["Timestamp"],
        y=display_df["Actual Energy"],
        mode="lines+markers",
        name="Actual Energy",
        hovertemplate=(
            "<b>Actual Energy</b><br>"
            "Time: %{x}<br>"
            "Energy: %{y:.2f} kW"
            "<extra></extra>"
        )
    )
)


# -------------------------------------------------
# PREDICTED ENERGY
# -------------------------------------------------

fig.add_trace(
    go.Scatter(
        x=display_df["Timestamp"],
        y=display_df["Predicted Energy"],
        mode="lines+markers",
        name="Predicted Energy",
        hovertemplate=(
            "<b>Predicted Energy</b><br>"
            "Time: %{x}<br>"
            "Energy: %{y:.2f} kW"
            "<extra></extra>"
        )
    )
)


# -------------------------------------------------
# ANOMALY POINTS
# -------------------------------------------------

anomaly_points = display_df[
    display_df["Status"] == "Anomaly"
]


if not anomaly_points.empty:

    fig.add_trace(
        go.Scatter(
            x=anomaly_points["Timestamp"],
            y=anomaly_points["Actual Energy"],
            mode="markers",
            name="Anomaly",
            marker=dict(
                size=12,
                symbol="x"
            ),
            text=[
                f"Energy: {value:.2f} kW"
                for value
                in anomaly_points["Actual Energy"]
            ],
            hovertemplate=(
                "<b>🚨 Anomaly</b><br>"
                "Time: %{x}<br>"
                "%{text}<br>"
                "<extra></extra>"
            )
        )
    )


# -------------------------------------------------
# CHART SETTINGS
# -------------------------------------------------

fig.update_layout(
    title="Energy Consumption Monitoring",
    xaxis_title="Time",
    yaxis_title="Energy (kW)",
    hovermode="x unified",
    height=500
)


st.plotly_chart(
    fig,
    use_container_width=True
)


st.caption(
    "❌ marks represent readings identified as anomalies."
)


st.divider()


# =================================================
# PREDICTION ERROR
# =================================================

st.header("📉 Prediction Error")


error_chart = display_df[
    [
        "Timestamp",
        "Prediction Error"
    ]
].copy()


error_chart = error_chart.set_index(
    "Timestamp"
)


st.line_chart(
    error_chart,
    height=300
)


st.metric(
    "Current Prediction Error",
    f"{current_error:.2f} kW"
)


st.divider()


# =================================================
# ANOMALY DETECTION
# =================================================

st.header("🚨 Anomaly Detection")


selected_anomalies = display_df[
    display_df["Status"] == "Anomaly"
].copy()


if len(selected_anomalies) > 0:

    st.warning(
        f"{len(selected_anomalies)} anomaly reading(s) "
        "detected in the selected range."
    )


    st.dataframe(
        selected_anomalies[
            [
                "ID",
                "Timestamp",
                "Actual Energy",
                "Predicted Energy",
                "Prediction Error",
                "Status"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )


else:

    st.success(
        "✅ No anomalies detected in the selected range."
    )


st.divider()


# =================================================
# ENERGY STATUS SUMMARY
# =================================================

st.header("📊 Energy Status Summary")


summary_df = pd.DataFrame(
    {
        "Status": [
            "Normal",
            "Anomaly"
        ],
        "Count": [
            normal_count,
            anomaly_count
        ]
    }
)


st.bar_chart(
    summary_df.set_index("Status")
)


st.divider()


# =================================================
# SELECTED DATA STATISTICS
# =================================================

st.header("📌 Selected Data Statistics")


stat_col1, stat_col2, stat_col3 = st.columns(3)


stat_col1.metric(
    "Selected Readings",
    len(display_df)
)


stat_col2.metric(
    "Selected Average Energy",
    f"{display_df['Actual Energy'].mean():.2f} kW"
)


stat_col3.metric(
    "Selected Average Error",
    f"{display_df['Prediction Error'].mean():.2f} kW"
)


st.divider()


# =================================================
# ENERGY READINGS TABLE
# =================================================

st.header("📋 Energy Readings")


table_df = display_df.sort_values(
    "Timestamp",
    ascending=False
).copy()


st.dataframe(
    table_df[
        [
            "ID",
            "Timestamp",
            "Actual Energy",
            "Predicted Energy",
            "Prediction Error",
            "Status"
        ]
    ],
    use_container_width=True,
    hide_index=True
)


st.divider()


# =================================================
# DOWNLOAD CSV
# =================================================

st.header("📥 Download Prediction History")


csv_data = realtime_df.to_csv(
    index=False
)


st.download_button(
    label="⬇️ Download CSV",
    data=csv_data,
    file_name="energy_prediction_history.csv",
    mime="text/csv"
)


st.divider()


# =================================================
# ABOUT DASHBOARD
# =================================================

with st.expander("ℹ️ About This Dashboard"):

    st.write(
        """
        This dashboard monitors household energy consumption
        using machine learning.

        The system provides:

        • Real-time energy readings
        • Energy consumption prediction
        • Prediction error monitoring
        • Anomaly detection
        • Interactive energy charts
        • Anomaly highlighting
        • Historical filtering
        • Automatic dashboard refresh
        • CSV data export

        The current real-time stream is a simulation that
        replays historical dataset readings.
        """
    )


# =================================================
# FOOTER
# =================================================

st.caption(
    "⚡ Smart Energy ML System | "
    "Dashboard refreshes automatically every 5 seconds"
)