import pandas as pd

def compute_kpis(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["aov"] = df["revenue"] / df["orders"]
    df["revenue_7d_ma"] = df["revenue"].rolling(7).mean()
    return df
