"""Inference consistency, metrics, neural padding and malformed-record checks."""

import joblib
import numpy as np
import pytest
import torch

from src.features.build import FEATURES
from src.models.deep import ANN, Autoencoder, PurchaseLSTM, infer, seed_everything, train_network
from src.models.evaluate import choose_threshold, classification, regression
from src.models.service import ModelService
from src.utils.config import artifacts, config


@pytest.fixture(scope="module")
def service():
    return ModelService()


def test_metrics():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.3, 0.4, 0.9])
    t = choose_threshold(y, p, 0.7)
    assert classification(y, p, t)["confusion_matrix"] == [[2, 0], [0, 2]]
    assert regression(y, y)["r2"] == 1


def test_inference_matches_saved(service, real_customers):
    x = real_customers.sample(8, random_state=42)
    out = service.score(x)
    np.testing.assert_allclose(out.churn_probability, x.churn_probability, rtol=1e-10)
    np.testing.assert_allclose(out.predicted_clv_90d, x.predicted_clv_90d, rtol=1e-10)
    np.testing.assert_allclose(out.churn_probability + out.purchase_probability, 1)


@pytest.mark.parametrize(
    "case", ["missing", "empty", "nan", "duplicate", "negative", "country", "affinity", "engagement"]
)
def test_invalid_features(service, real_customers, case):
    x = real_customers.head(2).copy()
    if case == "missing":
        x = x.drop(columns="frequency")
    if case == "empty":
        x = x.iloc[:0]
    if case == "nan":
        x.loc[0, "recency"] = np.nan
    if case == "duplicate":
        x.loc[1, "customer_id"] = x.loc[0, "customer_id"]
    if case == "negative":
        x.loc[0, "monetary"] = -1
    if case == "country":
        x.loc[0, "country"] = ""
    if case == "affinity":
        x.loc[0, "affinity_home"] = 2
    if case == "engagement":
        x.loc[0, "engagement"] = 101
    with pytest.raises(ValueError):
        service.score(x)


def test_unseen_country(service, real_customers):
    x = real_customers.head(1).copy()
    x.country = "Unknown new country"
    assert 0 <= service.score(x).churn_probability.iloc[0] <= 1


def test_explanations(service, real_customers):
    result = service.explainer.individual(real_customers.head(1))
    assert len(result["shap"]) == 10 and result["lime"]
    assert all(np.isfinite(r["contribution"]) for r in result["shap"])


def test_deep_artifacts_and_padding(real_customers):
    seed_everything(42)
    cfg = config()
    prep = joblib.load(artifacts() / "neural_preprocessing.joblib")
    x = prep.transform(real_customers.head(8)[FEATURES]).astype("float32")
    ann = ANN(x.shape[1], cfg["dl"]["hidden"], cfg["dl"]["dropout"])
    ann.load_state_dict(torch.load(artifacts() / "ann.pt", weights_only=True))
    ae = Autoencoder(x.shape[1])
    ae.load_state_dict(torch.load(artifacts() / "autoencoder.pt", weights_only=True))
    assert ((infer(ann, x, sigmoid=True) >= 0) & (infer(ann, x, sigmoid=True) <= 1)).all()
    assert infer(ae, x).shape == x.shape
    net = PurchaseLSTM()
    net.load_state_dict(torch.load(artifacts() / "lstm.pt", weights_only=True))
    seq = np.ones((2, 5, 8), dtype="float32")
    lengths = np.array([2, 3])
    before = infer(net, seq, lengths)
    seq[0, 2:] = 999
    seq[1, 3:] = 999
    np.testing.assert_allclose(before, infer(net, seq, lengths))


def test_training_loop():
    seed_everything(42)
    cfg = config()
    cfg["dl"].update(epochs=3, batch_size=8, patience=2)
    x = np.random.normal(size=(32, 6)).astype("float32")
    y = np.array([0, 1] * 16)
    net = ANN(6)
    curve = train_network(net, x, y, x, y, cfg, "ann")
    assert len(curve) <= 3 and not net.training and np.isfinite(curve[-1]["validation_loss"])
