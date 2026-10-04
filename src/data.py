"""Load and validate the Hillstrom randomized e-mail experiment."""
import pandas as pd

EXPECTED_COLUMNS = ["recency", "history_segment", "history", "mens", "womens", "zip_code",
                    "newbie", "channel", "segment", "visit", "conversion", "spend"]
ARMS = {"No E-Mail", "Mens E-Mail", "Womens E-Mail"}


def load(path):
    df = pd.read_csv(path)
    validate(df)
    df["zip_code"] = df["zip_code"].replace({"Surburban": "Suburban"})  # typo in the source
    return df


def validate(df):
    """Fail fast if the file is not the dataset the analysis expects."""
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if set(df["segment"].unique()) != ARMS:
        raise ValueError(f"Unexpected arms: {sorted(df['segment'].unique())}")
    for col in ["visit", "conversion", "mens", "womens", "newbie"]:
        if not df[col].isin([0, 1]).all():
            raise ValueError(f"{col} must be 0/1")
    if (df["spend"] < 0).any():
        raise ValueError("spend must be non-negative")
    if ((df["conversion"] == 1) & (df["visit"] == 0)).any():
        raise ValueError("a conversion without a visit is impossible")


def feature_matrix(df):
    """Numeric features for modelling: pre-campaign customer attributes only."""
    X = df[["recency", "history", "mens", "womens", "newbie"]].copy()
    X = X.join(pd.get_dummies(df[["zip_code", "channel"]], dtype=int))
    return X
