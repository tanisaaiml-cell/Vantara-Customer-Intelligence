"""Check error arithmetic, frozen test isolation and target-free exports."""

import json

import numpy as np
import pandas as pd
import pytest

from src.data.exports import processed_table
from src.features.build import FEATURES
from src.models.diagnostics import error_rows, error_summary, held_out_diagnostics
from src.utils.config import ROOT, artifacts


def test_known_error_arithmetic():
    rows = error_rows(np.array([0, 10, 20]), np.array([2, 8, 24]))
    metrics = error_summary(rows)
    assert rows.error.tolist() == [2, -2, 4]
    assert metrics["mae"] == pytest.approx(8 / 3)
    assert metrics["mse"] == pytest.approx(8)
    assert metrics["rmse"] == pytest.approx(np.sqrt(8))
    assert metrics["mean_error"] == pytest.approx(4 / 3)
    assert metrics["r2"] == pytest.approx(0.88)
    assert metrics["median_absolute_error"] == 2
    assert metrics["worst_10_squared_error_share"] == 1


def test_perfect_predictions_and_zero_targets():
    metrics = error_summary(error_rows(np.array([0, 0, 5]), np.array([0, 0, 5])))
    assert metrics["mse"] == 0 and metrics["worst_10_squared_error_share"] == 0
    assert metrics["r2"] == 1


@pytest.mark.parametrize("actual,predicted", [([], []), ([1], [1, 2]), ([np.nan], [0]), ([1], [np.inf])])
def test_invalid_error_inputs(actual, predicted):
    with pytest.raises(ValueError):
        error_rows(np.array(actual), np.array(predicted))


def test_actual_snapshot_test_ids_and_metrics():
    result = held_out_diagnostics()
    splits = json.loads((artifacts() / "split.json").read_text())
    table = processed_table("modeling")
    ids = set(table.iloc[splits["test"]].customer_id.astype(str))
    training_ids = set(table.iloc[splits["train"]].customer_id.astype(str))
    assert result["count"] == 788
    for task in ("value", "timing"):
        rows = pd.DataFrame(result[task]["rows"])
        assert set(rows.customer_id) == ids
        assert not set(rows.customer_id) & training_ids
        assert result[task]["summary"]["mse"] == pytest.approx(((rows.predicted - rows.actual) ** 2).mean())
    assert result["value"]["summary"]["mae"] == pytest.approx(709.9866254284246)
    assert result["churn"]["incorrect"] == 212
    assert result["churn"]["false_positives"] == 165
    assert result["churn"]["false_negatives"] == 47


def test_feature_export_has_no_targets():
    data = processed_table()
    assert data.columns.tolist() == ["customer_id", *FEATURES]
    assert len(data) == 5249 and data.customer_id.is_unique
    assert not {"churn", "clv_90d", "next_days", "next_category", "partition"} & set(data.columns)
    exported = pd.read_csv(ROOT / "data/processed/customer_features.csv", dtype={"customer_id": str})
    expected = data.copy()
    expected.customer_id = expected.customer_id.astype(str)
    pd.testing.assert_frame_equal(exported, expected, check_dtype=False)
    with pytest.raises(ValueError):
        processed_table("../../config")
