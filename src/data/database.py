"""PostgreSQL persistence with a native SQLite development fallback."""

import json
import os
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    select,
)
from sqlalchemy.engine import Engine

from src.utils.config import ROOT

metadata = MetaData()
customers = Table(
    "customers",
    metadata,
    Column("customer_id", String, primary_key=True),
    Column("country", String),
    Column("features", JSON, nullable=False),
)
segments = Table(
    "segments", metadata, Column("segment_id", Integer, primary_key=True), Column("name", String, nullable=False)
)
predictions = Table(
    "predictions",
    metadata,
    Column("customer_id", String, ForeignKey("customers.customer_id"), primary_key=True),
    Column("as_of", String, nullable=False),
    Column("segment_id", Integer, ForeignKey("segments.segment_id")),
    Column("churn_probability", Float),
    Column("predicted_clv_90d", Float),
    Column("payload", JSON, nullable=False),
)
transactions = Table(
    "transactions",
    metadata,
    Column("row_id", Integer, primary_key=True, autoincrement=True),
    Column("customer_id", String, ForeignKey("customers.customer_id")),
    Column("invoice", String),
    Column("invoice_date", DateTime),
    Column("amount", Float),
)
batch_scores = Table(
    "batch_scores",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("created_at", DateTime, default=lambda: datetime.now(timezone.utc)),
    Column("payload", JSON),
)


def engine() -> Engine:
    """Connect through DATABASE_URL or use a local SQLite file."""
    url = os.getenv("DATABASE_URL", f"sqlite:///{ROOT}/data/vantara.db")
    return create_engine(
        url, pool_pre_ping=True, connect_args={"check_same_thread": False} if url.startswith("sqlite") else {}
    )


def seed_database(db: Engine, cutoff: str) -> None:
    """Idempotently seed snapshot records and pre-cutoff invoice aggregates."""
    metadata.create_all(db)
    with db.begin() as conn:
        if conn.execute(select(predictions.c.customer_id).limit(1)).first():
            return
        frame = pd.read_parquet(ROOT / "data/processed/scored_customers.parquet")
        records = json.loads(frame.to_json(orient="records"))
        conn.execute(
            segments.insert(),
            [
                {"segment_id": int(r.cluster), "name": r.segment}
                for r in frame[["cluster", "segment"]].drop_duplicates().itertuples()
            ],
        )
        conn.execute(
            customers.insert(),
            [{"customer_id": str(r["customer_id"]), "country": r["country"], "features": r} for r in records],
        )
        conn.execute(
            predictions.insert(),
            [
                {
                    "customer_id": str(r["customer_id"]),
                    "as_of": cutoff,
                    "segment_id": int(r["cluster"]),
                    "churn_probability": r["churn_probability"],
                    "predicted_clv_90d": r["predicted_clv_90d"],
                    "payload": r,
                }
                for r in records
            ],
        )
        raw = pd.read_parquet(ROOT / "data/interim/transactions.parquet")
        raw = raw[(raw.invoice_date < pd.Timestamp(cutoff)) & raw.customer_id.isin(frame.customer_id)]
        invoice = (
            raw.groupby(["customer_id", "invoice"])
            .agg(invoice_date=("invoice_date", "min"), amount=("amount", "sum"))
            .reset_index()
        )
        rows = [
            {
                "customer_id": str(r.customer_id),
                "invoice": r.invoice,
                "invoice_date": r.invoice_date.to_pydatetime(),
                "amount": float(r.amount),
            }
            for r in invoice.itertuples()
        ]
        for start in range(0, len(rows), 2000):
            conn.execute(transactions.insert(), rows[start : start + 2000])


def fetch_predictions(db: Engine) -> list[dict]:
    """Read saved scores for business views."""
    with db.connect() as conn:
        return list(conn.execute(select(predictions.c.payload)).scalars())
