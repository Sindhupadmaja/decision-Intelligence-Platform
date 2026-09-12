def executive_insight(kpis, experiment, scenarios):
    latest = kpis.iloc[-1]
    return {
        "latest_revenue": float(latest["revenue"]),
        "experiment_effect": experiment["absolute_effect"],
        "experiment_p_value": experiment["p_value"],
        "scenario_range": scenarios,
        "decision_note": "Use effect size, uncertainty, and business impact together."
    }
