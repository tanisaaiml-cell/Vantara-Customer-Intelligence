"""Small accounting fixtures and the real trained customer snapshot."""

import pandas as pd
import pytest

from src.utils.config import ROOT


@pytest.fixture
def transactions_sample():
    """Include purchases, returns and exact date-boundary events."""
    rows = [
        ["1", "10001", "RED MUG", 2, "2011-01-01", 5, 1, "UK"],
        ["2", "10002", "HEART CANDLE", 1, "2011-02-01", 10, 1, "UK"],
        ["C3", "10001", "RED MUG", -1, "2011-02-02", 5, 1, "UK"],
        ["4", "10001", "RED MUG", 1, "2011-03-01", 5, 2, "FR"],
        ["5", "10001", "RED MUG", 3, "2011-04-01", 5, 1, "UK"],
        ["6", "10001", "RED MUG", 1, "2011-07-01", 5, 3, "UK"],
    ]
    return pd.DataFrame(
        rows,
        columns=["invoice", "stock_code", "description", "quantity", "invoice_date", "price", "customer_id", "country"],
    )


@pytest.fixture
def real_customers():
    """Load the actual artifact-backed records."""
    return pd.read_parquet(ROOT / "data/processed/scored_customers.parquet")
