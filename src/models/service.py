"""Immutable model loading and validated batch inference."""

import json

import joblib
import numpy as np
import pandas as pd

from src.explainability.explain import Explainer
from src.features.build import CATEGORIES, FEATURES, NUMERIC
from src.segmentation.cluster import COLUMNS
from src.utils.config import artifacts


class ModelService:
    """Load artifacts once; dashboard must access this service through FastAPI."""

    def __init__(self) -> None:
        """Load the validation-selected models and explanation background."""
        folder = artifacts()
        self.metadata = json.loads((folder / "metadata.json").read_text())
        self.churn = joblib.load(folder / "churn.joblib")
        self.clv = joblib.load(folder / "clv.joblib")
        self.category = joblib.load(folder / "category.joblib")
        self.segments = joblib.load(folder / "segments.joblib")
        self.explainer = Explainer(self.churn, pd.read_parquet(folder / "explanation_background.parquet"))

    def score(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Validate feature CSV records and return scores without using target fields."""
        missing = set(FEATURES + ["customer_id"]) - set(frame.columns)
        if missing:
            raise ValueError("Missing columns: " + ", ".join(sorted(missing)))
        if frame.empty or len(frame) > 5000:
            raise ValueError("Provide between 1 and 5000 customer rows")
        frame = frame.copy()
        if (
            frame.customer_id.isna().any()
            or frame.customer_id.astype(str).str.strip().eq("").any()
            or frame.customer_id.astype(str).duplicated().any()
        ):
            raise ValueError("Customer IDs must be present and unique")
        for col in NUMERIC:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        if not np.isfinite(frame[NUMERIC].to_numpy()).all():
            raise ValueError("All numerical features must be finite numbers")
        if (frame[[c for c in NUMERIC if c not in {"historical_clv", "frequency_trend"}]] < 0).any().any():
            raise ValueError("Unexpected negative feature")
        if frame.country.isna().any() or frame.country.astype(str).str.strip().eq("").any():
            raise ValueError("Country cannot be empty")
        for col in ["discount_proxy", "seasonal_concentration", *["affinity_" + c for c in CATEGORIES]]:
            if (frame[col] > 1).any():
                raise ValueError(f"{col} must be between zero and one")
        if (frame.engagement > 100).any():
            raise ValueError("Engagement must be between 0 and 100")
        output = pd.DataFrame({"customer_id": frame.customer_id.astype(str)})
        output["churn_probability"] = self.churn.predict_proba(frame[FEATURES])[:, 1]
        output["purchase_probability"] = 1 - output.churn_probability
        output["predicted_clv_90d"] = np.maximum(0, self.clv.predict(frame[FEATURES]))
        output["predicted_category"] = self.category.predict(frame[FEATURES])
        labels = self.segments["kmeans"].predict(self.segments["scaler"].transform(np.log1p(frame[COLUMNS])))
        output["segment"] = [self.segments["names"][int(i)] for i in labels]
        output["high_risk"] = output.churn_probability >= self.metadata["threshold"]
        return output
