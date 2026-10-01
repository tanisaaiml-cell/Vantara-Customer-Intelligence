"""Prepare modeling data from the original source workbook."""

import json

import numpy as np

from src.data.exports import export_csv_files
from src.data.pipeline import clean_transactions, load_workbook
from src.features.build import FEATURES, make_features, make_sequences, make_targets
from src.utils.config import ROOT, config, log


def prepare() -> None:
    """Create clean ledger, feature/label table, sequences and upload examples."""
    cfg = config()
    log("loading_workbook")
    raw = load_workbook(ROOT / cfg["raw_path"])
    frame, audit = clean_transactions(raw, cfg)
    extreme = (raw.quantity.abs() > cfg["max_quantity"]) | (raw.price > cfg["max_price"])
    raw[extreme].drop_duplicates().to_csv(ROOT / "data/interim/quarantined_extremes.csv", index=False)
    frame.to_parquet(ROOT / "data/interim/transactions.parquet", index=False)
    (ROOT / "docs/data_audit.json").write_text(json.dumps(audit, indent=2))
    x = make_features(frame, cfg["cutoff"], cfg)
    y = make_targets(frame, x.customer_id, cfg["cutoff"], cfg["horizon_days"])
    x.merge(y, on="customer_id").to_parquet(ROOT / "data/processed/customers.parquet", index=False)
    export_csv_files()
    values, lengths = make_sequences(frame, x.customer_id, cfg["cutoff"], cfg["sequence_length"])
    np.savez_compressed(ROOT / "data/processed/sequences.npz", values=values, lengths=lengths)
    x[["customer_id"]].head(5).to_csv(ROOT / "data/examples/customer_ids.csv", index=False)
    x[["customer_id", *FEATURES]].head(5).to_csv(ROOT / "data/examples/customer_features.csv", index=False)
    log("data_prepared", audit=audit, customers=len(x), churn_rate=float(y.churn.mean()))


if __name__ == "__main__":
    prepare()
