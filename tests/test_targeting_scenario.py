import numpy as np
import pandas as pd

from src import scenario, targeting


def test_ipw_value_of_logging_policy_is_mean_reward():
    arm = np.array(["A", "B", "A", "B"])
    reward = np.array([1.0, 3.0, 2.0, 4.0])
    prop = {"A": 0.5, "B": 0.5}
    assert targeting.ipw_value(arm, arm, reward, prop) == reward.mean()
    # Always choosing A is scored on the customers who actually received A.
    assert targeting.ipw_value(np.repeat("A", 4), arm, reward, prop) == 1.5


def test_break_even_cost_and_profit():
    readout = pd.DataFrame([{"treatment": "Mens E-Mail", "metric": "spend",
                             "absolute_lift": 0.8, "ci_low": 0.5, "ci_high": 1.1}])
    s = scenario.campaign_scenarios(readout, customers=1000, margin=0.3, cost=0.1).iloc[0]
    assert np.isclose(s["break_even_email_cost"], 0.24)
    assert np.isclose(s["incremental_profit"], (0.3 * 0.8 - 0.1) * 1000)
