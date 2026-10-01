"""Held-out diagnostics from frozen predictions; never fit or select a model here."""

import hashlib
import json

import numpy as np
import pandas as pd

from src.models.evaluate import classification, regression
from src.utils.config import ROOT, artifacts


def error_rows(actual: np.ndarray, predicted: np.ndarray) -> pd.DataFrame:
    """Use predicted minus actual: positive error means overprediction."""
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if actual.ndim != 1 or actual.shape != predicted.shape or not len(actual):
        raise ValueError("Actual and predicted must be matching, nonempty 1-D arrays")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Evaluation values must be finite")
    error = predicted - actual
    return pd.DataFrame(
        {
            "actual": actual,
            "predicted": predicted,
            "error": error,
            "absolute_error": np.abs(error),
            "squared_error": error**2,
        }
    )


def error_summary(rows: pd.DataFrame) -> dict:
    """Summarize full, untrimmed errors in the target's original units."""
    squared_total = float(rows.squared_error.sum())
    return {
        **regression(rows.actual, rows.predicted),
        "median_absolute_error": float(rows.absolute_error.median()),
        "mean_error": float(rows.error.mean()),
        "worst_10_squared_error_share": (
            float(rows.squared_error.nlargest(10).sum() / squared_total) if squared_total else 0.0
        ),
        "count": len(rows),
    }


def held_out_diagnostics() -> dict:
    """Join frozen test IDs to saved predictions, rejecting mismatched artifacts."""
    folder = artifacts()
    metadata = json.loads((folder / "metadata.json").read_text())
    source = ROOT / "data/processed/customers.parquet"
    if hashlib.sha256(source.read_bytes()).hexdigest() != metadata["feature_table_sha256"]:
        raise ValueError("Processed features differ from the trained snapshot; rebuild matching artifacts")
    data = pd.read_parquet(source)
    split = json.loads((folder / "split.json").read_text())
    flattened = [i for part in split.values() for i in part]
    if len(flattened) != len(data) or sorted(flattened) != list(range(len(data))):
        raise ValueError("Split indices must partition the feature table exactly once")
    test = data.iloc[split["test"]].copy()
    scores = pd.read_parquet(ROOT / "data/processed/scored_customers.parquet")
    test.customer_id = test.customer_id.astype(str)
    scores.customer_id = scores.customer_id.astype(str)
    prediction_columns = ["predicted_clv_90d", "predicted_next_days", "churn_probability", "partition"]
    test = test.merge(scores[["customer_id", *prediction_columns]], on="customer_id", validate="one_to_one")
    if len(test) != metadata["split_sizes"]["test"] or not test.partition.eq("test").all():
        raise ValueError("Saved scores do not match the frozen test partition")
    output = {}
    for task, target, prediction, unit in [
        ("value", "clv_90d", "predicted_clv_90d", "GBP"),
        ("timing", "next_days", "predicted_next_days", "days"),
    ]:
        rows = error_rows(test[target].to_numpy(), test[prediction].to_numpy())
        rows.insert(0, "customer_id", test.customer_id.to_numpy())
        output[task] = {"unit": unit, "summary": error_summary(rows), "rows": rows.to_dict("records")}
    champion = next(r for r in metadata["clv_runs"] if r["name"] == metadata["clv_champion"])
    for task, reported in [("value", champion["test"]), ("timing", metadata["lstm"]["test"])]:
        for key in ("mae", "rmse", "r2"):
            if not np.isclose(output[task]["summary"][key], reported[key], rtol=1e-6, atol=1e-7):
                raise ValueError("Saved regression predictions disagree with the training metrics")
    churn = classification(test.churn, test.churn_probability, metadata["threshold"])
    recorded = next(r for r in metadata["churn_runs"] if r["name"] == metadata["champion"])["test"]
    if churn["confusion_matrix"] != recorded["confusion_matrix"] or not np.isclose(
        churn["roc_auc"], recorded["roc_auc"]
    ):
        raise ValueError("Saved churn predictions disagree with the training metrics")
    tn, fp, fn, tp = np.asarray(churn["confusion_matrix"]).ravel()
    output["churn"] = {
        **churn,
        "error_rate": float((fp + fn) / len(test)),
        "incorrect": int(fp + fn),
        "correct": int(tn + tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
    }
    output["partition"] = "test"
    output["count"] = len(test)
    output["cutoff"] = metadata["cutoff"]
    return output
