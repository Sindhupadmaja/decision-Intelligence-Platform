"""End-to-end run: validate -> experiment readout -> targeting -> scenarios -> executive summary.

Run from the repository root:  python -m src.pipeline
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import config as cfg
from src import experiment, insights, scenario, targeting
from src.data import feature_matrix, load


def plot_lifts(readout, path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, metric in zip(axes, cfg.METRICS):
        r = readout[readout["metric"] == metric]
        y = np.arange(len(r))
        ax.errorbar(r["absolute_lift"], y, xerr=[r["absolute_lift"] - r["ci_low"],
                    r["ci_high"] - r["absolute_lift"]], fmt="o", capsize=4)
        ax.axvline(0, color="grey", linestyle="--")
        ax.set_yticks(y, r["treatment"])
        ax.set_title(f"{metric} lift vs no e-mail (95% CI)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_policies(summary, path):
    fig, ax = plt.subplots(figsize=(8, 3.8))
    s = summary.sort_values("mean")
    ax.barh(s.index, s["mean"], xerr=s["std"], capsize=4)
    ax.set_xlabel(f"Gross profit per customer ($), mean ± sd over {cfg.N_SPLITS} test splits")
    ax.set_title("Offline evaluation of targeting policies")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_deciles(deciles, treatment, path):
    fig, ax = plt.subplots(figsize=(8, 3.8))
    x = np.arange(1, len(deciles) + 1)
    ax.bar(x - 0.2, deciles["predicted_uplift"], width=0.4, label="predicted")
    ax.bar(x + 0.2, deciles["observed_uplift"], width=0.4, label="observed")
    ax.set_xticks(x)
    ax.set_xlabel("Decile of predicted uplift (1 = highest)")
    ax.set_ylabel("Spend uplift per customer ($)")
    ax.set_title(f"{treatment}: predicted vs observed uplift on held-out customers")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    cfg.FIGURES.mkdir(parents=True, exist_ok=True)
    df = load(cfg.RAW_CSV)
    X = feature_matrix(df)

    # 1. Experiment readout
    srm = experiment.sample_ratio_check(df)
    balance = experiment.covariate_balance(df, cfg.CONTROL, cfg.TREATMENTS,
                                           ["recency", "history", "mens", "womens", "newbie"])
    kpis = experiment.kpi_table(df)
    readout = experiment.readout(df, cfg.CONTROL, cfg.TREATMENTS, cfg.METRICS,
                                 cfg.ALPHA, cfg.N_BOOTSTRAP, cfg.RANDOM_STATE)
    mde = {m: experiment.minimum_detectable_effect(df, cfg.CONTROL, m) for m in cfg.METRICS}
    kpis.to_csv(cfg.REPORTS / "kpi_by_arm.csv")
    readout.to_csv(cfg.REPORTS / "experiment_readout.csv", index=False)
    print(kpis.round(4).to_string())
    print(readout.round(4).to_string(index=False))

    # 2. Targeting / next best action
    arms = [cfg.CONTROL] + cfg.TREATMENTS
    results, shares = targeting.evaluate_policies(X, df, arms, cfg.CONTROL, cfg.GROSS_MARGIN,
                                                  cfg.EMAIL_COST, cfg.N_SPLITS, cfg.RANDOM_STATE)
    policy_summary, policy_stats = targeting.summarize_policies(results)
    best_spend = readout[readout["metric"] == "spend"].set_index("treatment")["absolute_lift"].idxmax()
    deciles = targeting.uplift_by_decile(df, X, best_spend, cfg.CONTROL, cfg.RANDOM_STATE)
    groups = {"womens": "Bought womens last year", "mens": "Bought mens last year",
              "newbie": "New customer", "channel": "Channel"}
    seg = {t: targeting.segment_lifts(df, t, cfg.CONTROL, groups) for t in cfg.TREATMENTS}
    policy_summary.to_csv(cfg.REPORTS / "policy_evaluation.csv")
    deciles.to_csv(cfg.REPORTS / "uplift_deciles.csv", index=False)
    for t, s in seg.items():
        s.to_csv(cfg.REPORTS / f"segment_lifts_{t.split()[0].lower()}.csv", index=False)
    print(policy_summary.round(4).to_string())
    print(json.dumps(policy_stats, indent=2), shares.round(3).to_dict())

    # 3. Scenarios
    scen = scenario.campaign_scenarios(readout, cfg.BASE_CUSTOMERS, cfg.GROSS_MARGIN, cfg.EMAIL_COST)
    lift_best = readout.query("metric == 'spend' and treatment == @best_spend")["absolute_lift"].iloc[0]
    grid = scenario.sensitivity_grid(lift_best, cfg.BASE_CUSTOMERS, cfg.MARGIN_GRID, cfg.EMAIL_COST_GRID)
    scen.to_csv(cfg.REPORTS / "campaign_scenarios.csv", index=False)
    grid.round(0).to_csv(cfg.REPORTS / "profit_sensitivity.csv")
    print(scen.round(2).to_string(index=False))
    print(grid.round(0).to_string())

    # 4. Figures and executive summary
    plot_lifts(readout, cfg.FIGURES / "lifts.png")
    plot_policies(policy_summary, cfg.FIGURES / "policy_evaluation.png")
    plot_deciles(deciles, best_spend, cfg.FIGURES / "uplift_deciles.png")
    summary = insights.executive_summary(kpis, readout, policy_stats, shares, scen, srm,
                                         float(balance["smd"].abs().max()), cfg)
    (cfg.REPORTS / "executive_summary.md").write_text(summary)
    print(summary)

    metrics = {
        "sample_ratio_check": srm,
        "max_abs_smd": float(balance["smd"].abs().max()),
        "kpis": kpis.round(4).to_dict(orient="index"),
        "readout": readout.round(5).to_dict(orient="records"),
        "minimum_detectable_effect": mde,
        "policy_summary": policy_summary.round(4).to_dict(orient="index"),
        "policy_stats": policy_stats,
        "model_policy_action_shares": shares.round(4).to_dict(),
        "scenarios": scen.round(2).to_dict(orient="records"),
    }
    with open(cfg.REPORTS / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=float)


if __name__ == "__main__":
    main()
