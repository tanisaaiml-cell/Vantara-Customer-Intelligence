"""Full HTTP contracts using a real API with isolated persisted storage."""

import io
import os

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from api.main import app
from src.data.database import batch_scores, seed_database


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    old = os.environ.get("DATABASE_URL")
    os.environ["DATABASE_URL"] = "sqlite:///" + str(tmp_path_factory.mktemp("db") / "test.db")
    with TestClient(app) as c:
        yield c
    if old is None:
        os.environ.pop("DATABASE_URL", None)
    else:
        os.environ["DATABASE_URL"] = old


def test_health_customer_metadata(client):
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/metadata").json()["customers"] == 5249
    rows = client.get("/customers?limit=2").json()
    assert len(rows) == 2
    cid = rows[0]["customer_id"]
    assert client.post("/predict", json={"customer_id": cid}).json()["customer_id"] == cid
    assert client.get("/customers/" + cid).status_code == 200
    assert client.get("/customers?country=Nowhere").json() == []
    assert client.get("/customers?limit=0").status_code == 422
    seed_database(app.state.db, app.state.service.metadata["cutoff"])


def test_invalid_customer(client):
    assert client.get("/customers/99999999").status_code == 404
    assert client.post("/predict", json={"customer_id": "abc"}).status_code == 422
    assert client.post("/predict", json={"customer_id": "123", "extra": 1}).status_code == 422


def test_batch_persistence(client, real_customers):
    payload = real_customers.head(3).to_csv(index=False).encode()
    response = client.post("/predict/batch", files={"file": ("x.csv", payload)})
    assert response.status_code == 200 and response.json()["count"] == 3
    with app.state.db.connect() as conn:
        assert conn.execute(select(func.count()).select_from(batch_scores)).scalar() >= 1
    ids = real_customers[["customer_id"]].head(2).to_csv(index=False).encode()
    assert client.post("/predict/batch", files={"file": ("ids.csv", ids)}).json()["count"] == 2


@pytest.mark.parametrize("payload", [b"bad\n1\n", b"customer_id\n", b"customer_id\n123\n123\n", b"\xff\xfe\x00"])
def test_bad_csv(client, payload):
    assert client.post("/predict/batch", files={"file": ("bad.csv", payload)}).status_code == 422


def test_oversize_csv(client):
    assert client.post("/predict/batch", files={"file": ("big.csv", b"a" * (5 * 1024 * 1024 + 1))}).status_code == 413


def test_explanations_catalog_trends_pdf(client, real_customers):
    cid = str(real_customers.customer_id.iloc[0])
    assert client.get("/customers/" + cid + "/explain").json()["shap"]
    assert "items" in client.get("/recommendations/" + cid).json()
    trend = client.get("/trends").json()
    assert len(trend["forecast"]) == 3 and max(r["month"] for r in trend["actual"]) < "2011-09-01"
    assert client.get("/global-importance").json()
    assert client.get("/report/" + cid).content.startswith(b"%PDF")


def test_errors_and_processed_downloads(client):
    errors = client.get("/evaluation/errors")
    assert errors.status_code == 200
    assert errors.json()["count"] == 788
    assert errors.json()["value"]["summary"]["mse"] > 0
    info = client.get("/datasets/processed").json()
    assert info["rows"] == 5249 and info["feature_count"] == 23 and len(info["preview"]) == 100
    for kind in ("features", "modeling", "scored"):
        response = client.get("/datasets/processed/download", params={"kind": kind})
        assert response.status_code == 200 and "attachment" in response.headers["content-disposition"]
        table = pd.read_csv(io.BytesIO(response.content))
        assert len(table) == 5249
        assert ("clv_90d" in table.columns) == (kind != "features")
    assert client.get("/datasets/processed/download?kind=../../config").status_code == 422


def test_stale_evaluation_artifacts_fail_clearly(client, monkeypatch):
    def stale():
        raise ValueError("Snapshot changed")

    monkeypatch.setattr("api.main.held_out_diagnostics", stale)
    assert client.get("/evaluation/errors").status_code == 409
