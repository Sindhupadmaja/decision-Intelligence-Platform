"""Translate measured lifts into business scenarios under explicit cost and margin assumptions."""
import pandas as pd


def campaign_scenarios(readout, customers, margin, cost):
    """Incremental revenue and profit of e-mailing a whole customer base with each campaign."""
    spend = readout[readout["metric"] == "spend"]
    rows = []
    for _, r in spend.iterrows():
        rows.append({
            "campaign": r["treatment"],
            "incremental_revenue": r["absolute_lift"] * customers,
            "revenue_ci_low": r["ci_low"] * customers,
            "revenue_ci_high": r["ci_high"] * customers,
            "incremental_profit": (margin * r["absolute_lift"] - cost) * customers,
            "profit_ci_low": (margin * r["ci_low"] - cost) * customers,
            "profit_ci_high": (margin * r["ci_high"] - cost) * customers,
            "break_even_email_cost": margin * r["absolute_lift"],
        })
    return pd.DataFrame(rows)


def sensitivity_grid(spend_lift, customers, margins, costs):
    """Incremental profit for one campaign across margin and e-mail cost assumptions."""
    return pd.DataFrame({f"${c:.2f}/email": [(m * spend_lift - c) * customers for m in margins]
                         for c in costs}, index=[f"{int(m * 100)}% margin" for m in margins])
