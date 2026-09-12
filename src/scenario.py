def scenario_revenue(baseline_revenue, traffic_change=0, conversion_change=0, aov_change=0):
    return baseline_revenue * (1 + traffic_change) * (1 + conversion_change) * (1 + aov_change)

def run_scenarios(baseline):
    return {
        "downside": scenario_revenue(baseline, -0.10, -0.05),
        "base": scenario_revenue(baseline),
        "upside": scenario_revenue(baseline, 0.10, 0.05)
    }
