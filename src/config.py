"""Central configuration for the campaign decision intelligence pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data/raw/hillstrom.csv"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

CONTROL = "No E-Mail"
TREATMENTS = ["Mens E-Mail", "Womens E-Mail"]
METRICS = ["visit", "conversion", "spend"]

ALPHA = 0.05
N_BOOTSTRAP = 2000
N_SPLITS = 20             # repeated train/test splits for evaluating targeting policies
RANDOM_STATE = 42

# Business assumptions for scenario analysis (not in the data; varied in the sensitivity grid).
BASE_CUSTOMERS = 100_000
EMAIL_COST = 0.10         # $ per email sent
GROSS_MARGIN = 0.30       # share of revenue kept as gross profit
EMAIL_COST_GRID = [0.02, 0.05, 0.10, 0.20, 0.30]
MARGIN_GRID = [0.20, 0.30, 0.40, 0.50]
