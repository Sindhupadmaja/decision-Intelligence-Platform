import numpy as np

from src import experiment


def test_holm_matches_hand_calculation():
    adj = experiment.holm([0.01, 0.04, 0.03])
    assert np.allclose(adj, [0.03, 0.06, 0.06])


def test_lift_detects_known_effect(experiment_df):
    r = experiment.lift(experiment_df, "No E-Mail", "Mens E-Mail", "visit")
    assert 0.15 < r["absolute_lift"] < 0.25
    assert r["ci_low"] < r["absolute_lift"] < r["ci_high"]
    assert r["p_value"] < 0.001


def test_sample_ratio_check_on_even_split(experiment_df):
    assert experiment.sample_ratio_check(experiment_df)["p_value"] == 1.0
