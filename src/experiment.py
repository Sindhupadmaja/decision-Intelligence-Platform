"""A/B/C test readout: randomization checks, KPI table, lifts with confidence intervals."""
import numpy as np
import pandas as pd
from scipy import stats


def sample_ratio_check(df, arm_col="segment"):
    """Chi-square test that customers were split evenly across arms (sample ratio mismatch)."""
    counts = df[arm_col].value_counts().sort_index()
    chi2, p = stats.chisquare(counts)
    return {"counts": counts.to_dict(), "chi2": float(chi2), "p_value": float(p)}


def covariate_balance(df, control, treatments, covariates):
    """Standardized mean differences; |SMD| < 0.1 is the usual threshold for good balance."""
    rows = []
    c = df[df["segment"] == control]
    for t in treatments:
        g = df[df["segment"] == t]
        for col in covariates:
            pooled = np.sqrt((c[col].var() + g[col].var()) / 2)
            rows.append({"treatment": t, "covariate": col,
                         "smd": float((g[col].mean() - c[col].mean()) / pooled)})
    return pd.DataFrame(rows)


def kpi_table(df):
    g = df.groupby("segment")
    out = pd.DataFrame({
        "customers": g.size(),
        "visit_rate": g["visit"].mean(),
        "conversion_rate": g["conversion"].mean(),
        "revenue_per_customer": g["spend"].mean(),
        "total_revenue": g["spend"].sum(),
    })
    buyers = df[df["conversion"] == 1].groupby("segment")["spend"].mean()
    out["avg_order_value"] = buyers
    return out


def bootstrap_diff_ci(t, c, n_boot, alpha, rng):
    """Percentile bootstrap CI for a difference in means (robust to heavy-tailed spend)."""
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        diffs[i] = rng.choice(t, t.size).mean() - rng.choice(c, c.size).mean()
    return np.quantile(diffs, [alpha / 2, 1 - alpha / 2])


def lift(df, control, treatment, metric, alpha=0.05, n_boot=2000, rng=None):
    rng = rng or np.random.default_rng(0)
    t = df.loc[df["segment"] == treatment, metric].to_numpy(float)
    c = df.loc[df["segment"] == control, metric].to_numpy(float)
    diff = t.mean() - c.mean()
    if metric in ("visit", "conversion"):
        # Two-proportion z-test with an unpooled Wald interval.
        p_pool = (t.sum() + c.sum()) / (t.size + c.size)
        se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / t.size + 1 / c.size))
        p_value = 2 * stats.norm.sf(abs(diff) / se_pool)
        se = np.sqrt(t.mean() * (1 - t.mean()) / t.size + c.mean() * (1 - c.mean()) / c.size)
        z = stats.norm.ppf(1 - alpha / 2)
        lo, hi = diff - z * se, diff + z * se
        test = "two-proportion z-test"
    else:
        p_value = stats.ttest_ind(t, c, equal_var=False).pvalue
        lo, hi = bootstrap_diff_ci(t, c, n_boot, alpha, rng)
        test = "Welch t-test, bootstrap CI"
    return {"treatment": treatment, "metric": metric, "control_mean": c.mean(),
            "treatment_mean": t.mean(), "absolute_lift": diff,
            "relative_lift": diff / c.mean(), "ci_low": float(lo), "ci_high": float(hi),
            "p_value": float(p_value), "test": test}


def holm(p_values):
    """Holm-Bonferroni adjusted p-values (controls family-wise error across many tests)."""
    p = np.asarray(p_values, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def readout(df, control, treatments, metrics, alpha, n_boot, seed):
    rng = np.random.default_rng(seed)
    res = pd.DataFrame([lift(df, control, t, m, alpha, n_boot, rng)
                        for t in treatments for m in metrics])
    res["p_holm"] = holm(res["p_value"])
    res["significant"] = res["p_holm"] < alpha
    return res


def minimum_detectable_effect(df, control, metric, alpha=0.05, power=0.80):
    """Smallest absolute lift this sample size could reliably detect for one arm vs control."""
    n = (df["segment"] == control).sum()
    sd = df.loc[df["segment"] == control, metric].std()
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return float(z * sd * np.sqrt(2 / n))
