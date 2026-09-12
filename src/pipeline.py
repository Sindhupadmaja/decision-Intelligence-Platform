from pathlib import Path
import json
import pandas as pd
from kpi import compute_kpis
from experiment import compare_groups
from scenario import run_scenarios
from insights import executive_insight

ROOT = Path(__file__).resolve().parents[1]
kpis = compute_kpis(pd.read_csv(ROOT / "data/raw/kpi_daily.csv"))
exp = pd.read_csv(ROOT / "data/raw/experiment.csv")
control = exp.loc[exp.variant=="control","outcome"]
treatment = exp.loc[exp.variant=="treatment","outcome"]
experiment = compare_groups(control, treatment)
scenarios = run_scenarios(float(kpis.revenue.mean()))
insight = executive_insight(kpis, experiment, scenarios)

(ROOT / "reports").mkdir(exist_ok=True)
kpis.to_csv(ROOT / "reports/kpi_monitoring.csv", index=False)
with open(ROOT / "reports/executive_insight.json", "w") as f:
    json.dump(insight, f, indent=2)
print(json.dumps(insight, indent=2))
