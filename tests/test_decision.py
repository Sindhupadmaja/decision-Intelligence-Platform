from src.scenario import scenario_revenue

def test_neutral_scenario():
    assert scenario_revenue(1000) == 1000

def test_positive_scenario():
    assert scenario_revenue(1000, traffic_change=0.10) == 1100
