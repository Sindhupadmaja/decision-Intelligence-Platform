import numpy as np
from scipy import stats

def compare_groups(control, treatment):
    control, treatment = np.asarray(control), np.asarray(treatment)
    diff = treatment.mean() - control.mean()
    _, p = stats.ttest_ind(treatment, control, equal_var=False)
    return {
        "control_mean": float(control.mean()),
        "treatment_mean": float(treatment.mean()),
        "absolute_effect": float(diff),
        "relative_effect": float(diff / control.mean()),
        "p_value": float(p)
    }
