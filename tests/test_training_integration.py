"""Real-data retraining in an isolated temporary directory."""

import json
import shutil

import numpy as np
import pandas as pd
import pytest

from src.utils.config import ROOT


@pytest.mark.slow
def test_full_training_reproducibility(tmp_path, monkeypatch):
    import src.models.train as training

    (tmp_path / "data/processed").mkdir(parents=True)
    (tmp_path / "docs").mkdir()
    for name in ["customers.parquet", "sequences.npz"]:
        shutil.copy2(ROOT / "data/processed" / name, tmp_path / "data/processed" / name)
    monkeypatch.setattr(training, "ROOT", tmp_path)
    monkeypatch.setenv("MODEL_ARTIFACT_PATH", str(tmp_path / "models_artifacts"))
    training.train()
    before = pd.read_parquet(ROOT / "data/processed/scored_customers.parquet")
    after = pd.read_parquet(tmp_path / "data/processed/scored_customers.parquet")
    for col in ["churn_probability", "predicted_clv_90d", "predicted_next_days"]:
        if col == "predicted_next_days":
            # Float32 neural inference may differ by a few ULPs across CPU kernels.
            # At the 90-day cap this permits <9 seconds, not a meaningful model change.
            np.testing.assert_allclose(before[col], after[col], rtol=1e-6, atol=1e-5)
        else:
            np.testing.assert_allclose(before[col], after[col], rtol=1e-7, atol=1e-8)
    meta = json.loads((tmp_path / "models_artifacts/metadata.json").read_text())
    original_meta = json.loads((ROOT / "models_artifacts/metadata.json").read_text())
    for metric in ["mae", "rmse", "r2"]:
        np.testing.assert_allclose(
            meta["lstm"]["test"][metric], original_meta["lstm"]["test"][metric], rtol=1e-6, atol=1e-7
        )
    assert len(meta["churn_runs"]) == 7 and len(meta["lstm"]["cv"]) == 5
    assert all(len(r["cv_scores"]) == 5 for r in meta["churn_runs"])
    for name in ["ann.pt", "lstm.pt", "autoencoder.pt", "segments.joblib", "clv.joblib"]:
        assert (tmp_path / "models_artifacts" / name).is_file()
