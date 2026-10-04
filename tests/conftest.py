import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def experiment_df():
    """Synthetic 3-arm experiment where Mens e-mail clearly lifts spend."""
    rng = np.random.default_rng(0)
    n = 3000
    arms = np.repeat(["No E-Mail", "Mens E-Mail", "Womens E-Mail"], n // 3)
    visit = rng.random(n) < np.select([arms == "Mens E-Mail", arms == "Womens E-Mail"], [0.3, 0.2], 0.1)
    conversion = visit & (rng.random(n) < 0.2)
    spend = np.where(conversion, 100.0, 0.0)
    return pd.DataFrame({
        "recency": rng.integers(1, 13, n), "history_segment": "1) $0 - $100",
        "history": rng.uniform(30, 500, n).round(2), "mens": rng.integers(0, 2, n),
        "womens": rng.integers(0, 2, n), "zip_code": rng.choice(["Urban", "Surburban", "Rural"], n),
        "newbie": rng.integers(0, 2, n), "channel": rng.choice(["Web", "Phone", "Multichannel"], n),
        "segment": arms, "visit": visit.astype(int), "conversion": conversion.astype(int),
        "spend": spend,
    })
