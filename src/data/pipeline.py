"""Read both sheets; preserve signed returns and audit cleaning decisions."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

COLUMNS = ["invoice", "stock_code", "description", "quantity", "invoice_date", "price", "customer_id", "country"]


def load_workbook(path: str | Path) -> pd.DataFrame:
    """Read the exact two-sheet schema in chronological order."""
    sheets = pd.read_excel(path, sheet_name=None)
    if len(sheets) != 2:
        raise ValueError("Expected both Online Retail II sheets")
    frames = []
    for frame in sheets.values():
        if list(frame.columns) != [
            "Invoice",
            "StockCode",
            "Description",
            "Quantity",
            "InvoiceDate",
            "Price",
            "Customer ID",
            "Country",
        ]:
            raise ValueError("Unexpected workbook schema")
        frame.columns = COLUMNS
        frames.append(frame)
    return pd.concat(frames, ignore_index=True).sort_values("invoice_date").reset_index(drop=True)


def clean_transactions(raw: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict]:
    """Remove invalid accounting rows; retain legitimate IQR-flagged bulk purchases."""
    if not set(COLUMNS).issubset(raw.columns):
        raise ValueError("Missing required transaction columns")
    frame = raw.copy()
    frame["invoice_date"] = pd.to_datetime(frame.invoice_date, errors="coerce")
    if frame.invoice_date.isna().any():
        raise ValueError("Invalid transaction date")
    if not frame.invoice_date.between(cfg["minimum_date"], cfg["maximum_date"]).all():
        raise ValueError("Unexpected date range")
    missing = float(frame.customer_id.isna().mean())
    if missing > cfg["max_missing_customer_rate"]:
        raise ValueError("Customer ID null rate exceeds configured threshold")
    audit = {
        "raw_rows": len(frame),
        "missing_customer_rate": missing,
        "exact_duplicate_rows": int(frame.duplicated().sum()),
    }
    frame = frame.drop_duplicates().copy()
    for col in ["quantity", "price"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
        if not np.isfinite(frame[col]).all():
            raise ValueError(f"Nonfinite {col}")
        q1, q3 = frame[col].quantile([0.25, 0.75])
        audit[col + "_iqr_flagged"] = int((~frame[col].between(q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1))).sum())
    invalid = (frame.price <= 0) | (frame.quantity == 0)
    extreme = (frame.quantity.abs() > cfg["max_quantity"]) | (frame.price > cfg["max_price"])
    audit.update(
        invalid_price_or_zero_quantity=int(invalid.sum()), quarantined_extremes=int((extreme & ~invalid).sum())
    )
    frame = frame.loc[~invalid & ~extreme].copy()
    frame["invoice"] = frame.invoice.astype(str).str.upper().str.strip()
    frame["stock_code"] = frame.stock_code.astype(str).str.upper().str.strip()
    frame["description"] = frame.description.fillna("").str.upper().str.replace(r"\s+", " ", regex=True).str.strip()
    frame["country"] = frame.country.fillna("Unknown").astype(str)
    frame["customer_id"] = frame.customer_id.astype("Int64").astype("string")
    frame["is_return"] = (frame.quantity < 0) | frame.invoice.str.startswith("C")
    frame["quantity"] = np.where(frame.is_return, -frame.quantity.abs(), frame.quantity)
    frame["amount"] = frame.quantity * frame.price
    frame["is_product"] = frame.stock_code.str.match(r"^\d{5}[A-Z]*$")
    audit.update(
        clean_rows=len(frame),
        return_rows=int(frame.is_return.sum()),
        identified_customers=int(frame.customer_id.nunique()),
        min_date=str(frame.invoice_date.min()),
        max_date=str(frame.invoice_date.max()),
    )
    return frame.reset_index(drop=True), audit
