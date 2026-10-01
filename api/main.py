"""REST API for stored predictions, batch inference and explanations."""

import io
import json
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator, Literal

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import Response
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from sqlalchemy import select, text

from api.schemas.customer import PredictionRequest
from src.data.database import batch_scores, engine, fetch_predictions, predictions, seed_database, transactions
from src.data.exports import dataset_preview, processed_table
from src.models.diagnostics import held_out_diagnostics
from src.models.service import ModelService
from src.utils.config import artifacts, log
from src.utils.pdf_fonts import register_fonts


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load models and initialize persistent storage before accepting traffic."""
    app.state.service = ModelService()
    app.state.db = engine()
    seed_database(app.state.db, app.state.service.metadata["cutoff"])
    yield
    app.state.db.dispose()


app = FastAPI(title="Vantara Customer Intelligence", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def request_log(request: Request, call_next: object) -> Response:
    """Log status and latency without recording personal input values."""
    start = time.perf_counter()
    response = await call_next(request)
    log(
        "api_request",
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        elapsed_ms=round((time.perf_counter() - start) * 1000, 2),
    )
    return response


@app.get("/health")
def health() -> dict:
    """Check model readiness and actual database connectivity."""
    with app.state.db.connect() as conn:
        conn.execute(text("SELECT 1"))
    return {"status": "ok", "model": app.state.service.metadata["champion"], "database": app.state.db.dialect.name}


@app.get("/metadata")
def model_metadata() -> dict:
    """Return measured results and versioned training metadata."""
    return app.state.service.metadata


@app.get("/evaluation/errors")
def evaluation_errors() -> dict:
    """Expose frozen test errors only; fail clearly on stale or missing artifacts."""
    try:
        return held_out_diagnostics()
    except FileNotFoundError as exc:
        raise HTTPException(503, "Evaluation artifacts are missing; restore the complete project") from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@app.get("/datasets/processed")
def processed_preview() -> dict:
    """Describe the feature-engineered snapshot and return a bounded preview."""
    return dataset_preview()


@app.get("/datasets/processed/download")
def processed_download(kind: Literal["features", "modeling", "scored"] = "features") -> Response:
    """Export the chosen snapshot as CSV using an allowlisted dataset selection."""
    names = {
        "features": "customer_features.csv",
        "modeling": "customers_with_targets.csv",
        "scored": "scored_customers.csv",
    }
    return Response(
        processed_table(kind).to_csv(index=False).encode("utf-8-sig"),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={names[kind]}"},
    )


@app.get("/customers")
def customer_list(
    segment: str | None = None,
    country: str | None = None,
    value_tier: str | None = None,
    limit: int = Query(6000, ge=1, le=10000),
) -> list[dict]:
    """Filter stored scores and rank retention priorities."""
    rows = fetch_predictions(app.state.db)
    rows = [
        r
        for r in rows
        if (segment is None or r["segment"] == segment)
        and (country is None or r["country"] == country)
        and (value_tier is None or r["value_tier"] == value_tier)
    ]
    return sorted(rows, key=lambda r: r["priority"], reverse=True)[:limit]


def lookup(customer_id: str) -> dict:
    """Fetch a saved snapshot record or return HTTP 404."""
    with app.state.db.connect() as conn:
        value = conn.execute(
            select(predictions.c.payload).where(predictions.c.customer_id == customer_id)
        ).scalar_one_or_none()
    if value is None:
        raise HTTPException(404, "Customer not found in the scored snapshot")
    return value


@app.post("/predict")
def predict(record: PredictionRequest) -> dict:
    """Serve a saved single-customer prediction."""
    return lookup(record.customer_id)


@app.get("/customers/{customer_id}")
def customer_detail(customer_id: str) -> dict:
    """Serve observations and scores for one customer."""
    return lookup(customer_id)


@app.get("/customers/{customer_id}/explain")
def explain(customer_id: str) -> dict:
    """Compute local SHAP and LIME explanations."""
    return app.state.service.explainer.individual(pd.DataFrame([lookup(customer_id)]))


@app.post("/predict/batch")
async def predict_batch(file: UploadFile = File(...)) -> dict:
    """Score feature CSVs or retrieve ID-only snapshot lists and save the batch."""
    content = await file.read(5 * 1024 * 1024 + 1)
    if len(content) > 5 * 1024 * 1024:
        raise HTTPException(413, "CSV exceeds 5 MB limit")
    try:
        frame = pd.read_csv(io.BytesIO(content), dtype={"customer_id": str})
        if not 1 <= len(frame) <= 5000:
            raise ValueError("CSV must contain 1 to 5000 rows")
        if frame.columns.tolist() == ["customer_id"]:
            if frame.customer_id.isna().any() or frame.customer_id.duplicated().any():
                raise ValueError("IDs must be present and unique")
            records = [lookup(i) for i in frame.customer_id]
        else:
            records = json.loads(app.state.service.score(frame).to_json(orient="records"))
        with app.state.db.begin() as conn:
            conn.execute(batch_scores.insert(), {"payload": records})
        return {"count": len(records), "results": records}
    except (ValueError, KeyError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        log("validation_failed", reason=str(exc))
        raise HTTPException(422, str(exc)) from exc


@app.get("/trends")
def revenue_trends() -> dict:
    """Return pre-cutoff revenue and an explicitly labeled naive forecast."""
    with app.state.db.connect() as conn:
        data = pd.read_sql(select(transactions.c.invoice_date, transactions.c.amount), conn)
    data.invoice_date = pd.to_datetime(data.invoice_date)
    monthly = data.set_index("invoice_date").amount.resample("MS").sum()
    future = pd.date_range(monthly.index[-1] + pd.offsets.MonthBegin(1), periods=3, freq="MS")
    return {
        "actual": [{"month": str(k.date()), "revenue": float(v)} for k, v in monthly.items()],
        "forecast": [{"month": str(k.date()), "revenue": float(monthly.tail(3).mean())} for k in future],
        "method": "Three-month moving-average baseline; no confidence interval. Identified snapshot customers only.",
    }


@app.get("/recommendations/{customer_id}")
def recommendations(customer_id: str) -> dict:
    """Suggest unseen popular products from a predicted proxy category."""
    row = lookup(customer_id)
    catalog = pd.read_parquet(artifacts() / "recommendations.parquet")
    chosen = catalog[catalog.category == row["predicted_category"]]
    if chosen.empty:
        chosen = catalog
    bought = json.loads((artifacts() / "purchased_products.json").read_text()).get(customer_id, [])
    return {
        "method": "Heuristic ranking of unseen popular products in a description-derived category.",
        "items": json.loads(chosen[~chosen.stock_code.isin(bought)].head(5).to_json(orient="records")),
    }


@app.get("/global-importance")
def global_importance() -> list[dict]:
    """Serve the saved mean absolute SHAP importance."""
    return json.loads((artifacts() / "global_importance.json").read_text())


@app.get("/report/{customer_id}")
def customer_pdf(customer_id: str) -> Response:
    """Generate a downloadable individual PDF from stored predictions."""
    row = lookup(customer_id)
    register_fonts()
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    pdf.setTitle("Vantara Customer Report")
    pdf.setFont("VantaraBold", 20)
    pdf.drawString(48, 785, "VANTARA | Customer report")
    pdf.setFont("VantaraSans", 10)
    lines = [
        f"Customer: {customer_id}",
        f"Prediction date: {app.state.service.metadata['cutoff']}",
        f"Country: {row['country']}",
        f"Segment: {row['segment']}",
        f"90-day churn risk: {row['churn_probability']:.1%}",
        f"Expected 90-day gross spend: GBP {row['predicted_clv_90d']:,.2f}",
        f"Purchase probability: {row['purchase_probability']:.1%}",
        "Historical demo snapshot. Predictions are estimates.",
    ]
    for i, line in enumerate(lines):
        pdf.drawString(48, 740 - i * 28, line)
    pdf.save()
    return Response(
        buffer.getvalue(),
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=customer_{customer_id}.pdf"},
    )
