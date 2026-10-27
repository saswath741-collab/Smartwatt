"""Optional Streamlit dashboard: streamlit run dashboard.py"""
import pandas as pd
import streamlit as st

from smartwatt.data import generate_synthetic
from smartwatt.evaluate import run_once

st.title("SmartWatt: electricity anomaly detection")
src = st.sidebar.radio("Data", ["Synthetic", "Upload CSV"])
if src == "Synthetic":
    seed = st.sidebar.number_input("Seed", 0, 999, 0)
    df = generate_synthetic(seed=int(seed))
else:
    up = st.sidebar.file_uploader("CSV with timestamp and consumption columns")
    if up is None:
        st.info("Upload a CSV, or switch to synthetic data.")
        st.stop()
    df = pd.read_csv(up, parse_dates=["timestamp"])
    if "is_anomaly" not in df:
        df["is_anomaly"] = -1

table, feat, preds, cut = run_once(df)
method = st.sidebar.selectbox("Detector", list(preds))
if table is not None:
    st.subheader("Test-set metrics")
    st.dataframe(table.style.format("{:.3f}"))

feat = feat.assign(flagged=preds[method])
days = st.sidebar.slider("Days to show", 3, 60, 14)
view = feat.iloc[cut:cut + days * 24].set_index("timestamp")
st.subheader(f"Consumption and detections ({method})")
st.line_chart(view["consumption"])
st.write(f"Flagged {int(view['flagged'].sum())} readings in this window.")
st.dataframe(view.loc[view["flagged"] == 1, ["consumption", "is_anomaly"]])
