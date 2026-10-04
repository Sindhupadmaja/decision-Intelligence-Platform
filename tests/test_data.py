import pytest

from src.data import feature_matrix, validate


def test_valid_data_passes(experiment_df):
    validate(experiment_df)
    X = feature_matrix(experiment_df)
    assert "segment" not in X.columns and "spend" not in X.columns   # no outcome leakage


def test_conversion_without_visit_is_rejected(experiment_df):
    bad = experiment_df.copy()
    i = bad.index[bad["visit"] == 0][0]
    bad.loc[i, "conversion"] = 1
    with pytest.raises(ValueError):
        validate(bad)


def test_unknown_arm_is_rejected(experiment_df):
    bad = experiment_df.copy()
    bad.loc[0, "segment"] = "SMS"
    with pytest.raises(ValueError):
        validate(bad)
