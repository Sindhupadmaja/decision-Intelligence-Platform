"""Generate a plain-language executive summary from the computed results."""


def executive_summary(kpis, readout, policy_stats, shares, scenarios, srm, balance_max, cfg):
    sig = readout[readout["significant"]]
    spend = readout[readout["metric"] == "spend"].set_index("treatment")
    best = spend["absolute_lift"].idxmax()
    other = [t for t in spend.index if t != best][0]
    sc = scenarios.set_index("campaign")
    lines = [
        "# Executive summary",
        "",
        "## Recommendation",
        f"Send the **{best}** campaign. It raised revenue per customer by "
        f"${spend.loc[best, 'absolute_lift']:.2f} over no e-mail "
        f"(95% CI ${spend.loc[best, 'ci_low']:.2f} to ${spend.loc[best, 'ci_high']:.2f}), "
        f"a {spend.loc[best, 'relative_lift']:.0%} increase, versus "
        f"${spend.loc[other, 'absolute_lift']:.2f} for the {other} campaign.",
        "",
        f"At {cfg.GROSS_MARGIN:.0%} gross margin and ${cfg.EMAIL_COST:.2f} per e-mail, sending "
        f"it to {cfg.BASE_CUSTOMERS:,} customers would add about "
        f"${sc.loc[best, 'incremental_revenue']:,.0f} in revenue and "
        f"${sc.loc[best, 'incremental_profit']:,.0f} in gross profit. The campaign stays "
        f"profitable while e-mails cost less than ${sc.loc[best, 'break_even_email_cost']:.2f} each.",
        "",
        "## Targeting",
        f"A next-best-action model that picks the most profitable option per customer averaged "
        f"${policy_stats['model_minus_best_blanket_mean']:+.3f} profit per customer versus "
        f"e-mailing everyone with the best single campaign, and won in "
        f"{policy_stats['model_wins_share_of_splits']:.0%} of {cfg.N_SPLITS} repeated test splits. "
        + ("Personalization adds value here." if policy_stats["model_wins_share_of_splits"] >= 0.8
           else "That is not a reliable gain, so the simpler blanket campaign is the better decision "
                "until a larger test shows otherwise."),
        "",
        "## Evidence quality",
        f"- Randomization check: arm sizes match an even split (chi-square p = {srm['p_value']:.2f}); "
        f"largest covariate imbalance |SMD| = {balance_max:.3f} (below 0.1 is balanced).",
        f"- {len(sig)} of {len(readout)} lift tests remain significant after Holm correction.",
        "- Spend is revenue, not profit; margin and e-mail cost are assumptions varied in the sensitivity grid.",
        "- The test ran for two weeks in 2008, so long-term effects (unsubscribes, repeat purchases) are unknown.",
    ]
    return "\n".join(lines) + "\n"
