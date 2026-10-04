"""Next-best-action targeting: predict each customer's response to every option, pick the
most profitable one, and evaluate policies offline using the randomized experiment."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


def fit_t_learner(X, arm, y, arms, random_state=42):
    """T-learner: one spend model per arm, each trained only on customers in that arm."""
    models = {}
    for a in arms:
        m = arm == a
        model = HistGradientBoostingRegressor(loss="poisson", max_depth=3, learning_rate=0.05,
                                              max_iter=200, min_samples_leaf=200,
                                              random_state=random_state)
        models[a] = model.fit(X[m], y[m])
    return models


def predict_spend(models, X):
    return pd.DataFrame({a: m.predict(X) for a, m in models.items()}, index=X.index)


def expected_profit(pred_spend, control, margin, cost):
    """Gross profit per customer for each option: margin * spend minus the cost of e-mailing."""
    profit = pred_spend * margin
    for a in profit.columns:
        if a != control:
            profit[a] -= cost
    return profit


def ipw_value(chosen, arm, reward, propensity):
    """Self-normalized inverse propensity estimate of a policy's average reward per customer.

    Because arms were assigned at random, customers whose actual arm matches the policy's
    choice are an unbiased sample of what would happen if the policy were deployed."""
    arm = np.asarray(arm)
    match = (np.asarray(chosen) == arm).astype(float)
    w = match / np.array([propensity[a] for a in arm])
    return float((w * reward).sum() / w.sum())


def evaluate_policies(X, df, arms, control, margin, cost, n_splits, seed):
    """Repeat 50/50 train/test splits; fit on train, score every policy on test."""
    rng = np.random.default_rng(seed)
    propensity = df["segment"].value_counts(normalize=True).to_dict()
    reward_all = margin * df["spend"] - cost * (df["segment"] != control)
    rows, share_rows = [], []
    for split in range(n_splits):
        test = rng.random(len(df)) < 0.5
        tr, te = ~test, test
        models = fit_t_learner(X[tr], df["segment"][tr].to_numpy(), df["spend"][tr].to_numpy(),
                               arms, random_state=seed + split)
        profit = expected_profit(predict_spend(models, X[te]), control, margin, cost)
        arm_te, reward_te = df["segment"][te], reward_all[te].to_numpy()
        n = te.sum()
        policies = {
            "No e-mail": np.repeat(control, n),
            **{f"Everyone: {a}": np.repeat(a, n) for a in arms if a != control},
            "Random": rng.choice(arms, n),
            "Model (next best action)": profit.idxmax(axis=1).to_numpy(),
        }
        for name, chosen in policies.items():
            rows.append({"split": split, "policy": name,
                         "profit_per_customer": ipw_value(chosen, arm_te.to_numpy(), reward_te, propensity)})
        share_rows.append(pd.Series(policies["Model (next best action)"]).value_counts(normalize=True))
    results = pd.DataFrame(rows)
    shares = pd.DataFrame(share_rows).fillna(0).mean()
    return results, shares


def summarize_policies(results):
    piv = results.pivot(index="split", columns="policy", values="profit_per_customer")
    summary = piv.agg(["mean", "std"]).T.sort_values("mean", ascending=False)
    blanket = [c for c in piv.columns if c.startswith("Everyone")]
    best_blanket = piv[blanket].mean().idxmax()
    diff = piv["Model (next best action)"] - piv[best_blanket]
    return summary, {
        "best_blanket_policy": best_blanket,
        "model_minus_best_blanket_mean": float(diff.mean()),
        "model_minus_best_blanket_p05": float(diff.quantile(0.05)),
        "model_minus_best_blanket_p95": float(diff.quantile(0.95)),
        "model_wins_share_of_splits": float((diff > 0).mean()),
    }


def uplift_by_decile(df, X, treatment, control, seed=42):
    """Rank held-out customers by predicted spend uplift and compare with observed uplift."""
    rng = np.random.default_rng(seed)
    test = rng.random(len(df)) < 0.5
    keep = df["segment"].isin([treatment, control]).to_numpy()
    tr, te = ~test & keep, test & keep
    models = fit_t_learner(X[tr], df["segment"][tr].to_numpy(), df["spend"][tr].to_numpy(),
                           [control, treatment], random_state=seed)
    pred = predict_spend(models, X[te])
    d = df[te].assign(pred_uplift=(pred[treatment] - pred[control]).to_numpy())
    d["decile"] = pd.qcut(d["pred_uplift"].rank(method="first"), 10, labels=False)
    g = d.groupby(["decile", "segment"])["spend"].mean().unstack()
    out = pd.DataFrame({"predicted_uplift": d.groupby("decile")["pred_uplift"].mean(),
                        "observed_uplift": g[treatment] - g[control]})
    return out.sort_index(ascending=False).reset_index(drop=True)


def segment_lifts(df, treatment, control, groups):
    """Observed spend lift by customer group (exploratory; not corrected for multiple tests)."""
    rows = []
    for col, label in groups.items():
        for val, sub in df.groupby(col):
            t = sub.loc[sub["segment"] == treatment, "spend"]
            c = sub.loc[sub["segment"] == control, "spend"]
            se = np.sqrt(t.var() / len(t) + c.var() / len(c))
            d = t.mean() - c.mean()
            rows.append({"group": label, "value": str(val), "customers": len(sub),
                         "spend_lift": d, "ci_low": d - 1.96 * se, "ci_high": d + 1.96 * se})
    return pd.DataFrame(rows)
