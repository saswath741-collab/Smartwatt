"""Time-based feature extraction. All features use only past information."""
import numpy as np
import pandas as pd

FEATURES = ["consumption", "hour_sin", "hour_cos", "is_weekend",
            "lag_1", "diff_1", "roll_mean_24", "roll_std_24", "ratio_to_roll"]


def add_features(df):
    d = df.copy()
    ts = pd.to_datetime(d["timestamp"])
    d["hour"] = ts.dt.hour
    d["is_weekend"] = (ts.dt.day_of_week >= 5).astype(int)
    d["hour_sin"] = np.sin(2 * np.pi * d["hour"] / 24)
    d["hour_cos"] = np.cos(2 * np.pi * d["hour"] / 24)
    c = d["consumption"]
    d["lag_1"] = c.shift(1)
    d["diff_1"] = c - d["lag_1"]
    past = c.shift(1)
    d["roll_mean_24"] = past.rolling(24, min_periods=6).mean()
    d["roll_std_24"] = past.rolling(24, min_periods=6).std()
    d["ratio_to_roll"] = c / (d["roll_mean_24"] + 1e-6)
    return d.dropna(subset=FEATURES).reset_index(drop=True)
