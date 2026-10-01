"""Consistent global and local SHAP/LIME explanations."""

import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer

from src.features.build import FEATURES


class Explainer:
    """Explain a persisted classification pipeline on transformed features."""

    def __init__(self, pipeline: object, background: pd.DataFrame) -> None:
        """Build explainers using only saved training background rows."""
        self.pipeline = pipeline
        self.background = pipeline["preprocess"].transform(background[FEATURES])
        self.names = pipeline["preprocess"].get_feature_names_out().tolist()
        model = pipeline["model"]
        if hasattr(model, "feature_importances_"):
            self.shap = shap.TreeExplainer(model)
        elif hasattr(model, "coef_"):
            self.shap = shap.LinearExplainer(model, self.background)
        else:
            self.shap = shap.Explainer(
                lambda x: model.predict_proba(x)[:, 1],
                self.background,
                feature_names=self.names,
                algorithm="permutation",
            )
        self.lime = LimeTabularExplainer(
            self.background,
            feature_names=self.names,
            class_names=["Purchase", "Churn"],
            random_state=42,
            mode="classification",
        )

    def values(self, records: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return signed churn contributions and transformed values."""
        x = self.pipeline["preprocess"].transform(records[FEATURES])
        explanation = self.shap(x)
        v = explanation.values
        if v.ndim == 3:
            v = v[:, :, 1]
        return v, x

    def individual(self, record: pd.DataFrame) -> dict:
        """Explain one prediction with attributions and plain-language associations."""
        values, x = self.values(record)
        order = np.argsort(np.abs(values[0]))[::-1][:10]
        contributions = [
            {
                "feature": self.names[i].replace("numeric__", "").replace("country__", ""),
                "value": float(x[0, i]),
                "contribution": float(values[0, i]),
            }
            for i in order
        ]
        local = self.lime.explain_instance(x[0], self.pipeline["model"].predict_proba, num_features=8, num_samples=1200)
        labels = {
            "recency": "days since the last purchase",
            "spend_90d": "recent spending",
            "historical_clv": "historical net spending",
            "frequency": "order frequency",
            "monetary": "historical gross spending",
            "orders_90d": "recent order count",
            "engagement": "the engagement score",
            "gap_variance": "irregular purchase intervals",
        }
        strongest = [
            labels.get(c["feature"], c["feature"].replace("_", " ")) for c in contributions if c["contribution"] > 0
        ][:3]
        sentence = (
            "The model associates " + ", ".join(strongest) + " with higher churn risk relative to its baseline."
            if strongest
            else "The strongest signals reduce churn risk relative to the model baseline."
        )
        return {
            "shap": contributions,
            "lime": [{"feature": k, "contribution": float(v)} for k, v in local.as_list()],
            "lime_local_fit": float(local.score),
            "explanation": sentence,
            "units": "Signed model-output contribution. Random Forest explains probability; some other models use log-odds. Associations are not causal effects.",
        }
