"""Readable exports of the model-ready table, with targets separated from inputs."""

import json

import pandas as pd

from src.features.build import FEATURES
from src.utils.config import ROOT, artifacts


def processed_table(kind: str = "features") -> pd.DataFrame:
    """Feature-only is safe for inference; modeling includes future outcome labels."""
    if kind not in {"features", "modeling", "scored"}:
        raise ValueError("Unknown processed dataset")
    name = "scored_customers.parquet" if kind == "scored" else "customers.parquet"
    data = pd.read_parquet(ROOT / "data/processed" / name)
    return data[["customer_id", *FEATURES]] if kind == "features" else data


def export_csv_files() -> dict:
    """Write human-readable copies without modifying canonical Parquet artifacts."""
    folder = ROOT / "data/processed"
    for kind, name in [("features", "customer_features.csv"), ("modeling", "customers_with_targets.csv")]:
        processed_table(kind).to_csv(folder / name, index=False)
    return {"rows": len(processed_table()), "features": len(FEATURES)}


def dataset_preview() -> dict:
    """Expose provenance, schema and the first 100 feature rows through HTTP."""
    data = processed_table()
    full = processed_table("modeling")
    metadata = json.loads((artifacts() / "metadata.json").read_text())
    return {
        "rows": len(data),
        "feature_count": len(FEATURES),
        "cutoff": metadata["cutoff"],
        "horizon_days": metadata["horizon_days"],
        "feature_columns": FEATURES,
        "target_columns": [c for c in full.columns if c not in data.columns],
        "source": "data/processed/customers.parquet",
        "preview": json.loads(data.head(100).to_json(orient="records")),
    }
