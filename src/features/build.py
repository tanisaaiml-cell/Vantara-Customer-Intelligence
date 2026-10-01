"""Point-in-time feature tables, complete future labels and padded purchase sequences."""

from typing import Any

import numpy as np
import pandas as pd

CATEGORIES = ["home", "kitchen", "seasonal", "accessories", "other"]
NUMERIC = [
    "recency",
    "frequency",
    "monetary",
    "average_spend",
    "historical_clv",
    "basket_size",
    "frequency_trend",
    "gap_variance",
    "seasonal_concentration",
    "return_rate",
    "discount_proxy",
    "engagement",
    "tenure",
    "orders_90d",
    "spend_90d",
    "unique_products",
    "sku_popularity",
    *["affinity_" + c for c in CATEGORIES],
]
FEATURES = [*NUMERIC, "country"]


def category(description: pd.Series) -> pd.Series:
    """Assign documented proxy categories, not a merchant-provided taxonomy."""
    rules = [
        r"HEART|CANDLE|LANTERN|FRAME|CLOCK|CUSHION",
        r"MUG|CUP|PLATE|BOWL|KITCHEN|TEA|CAKE",
        r"CHRISTMAS|XMAS|EASTER|SANTA|HALLOWEEN",
        r"BAG|NECKLACE|BRACELET|PURSE|RING",
    ]
    return pd.Series(
        np.select([description.str.contains(r, na=False) for r in rules], CATEGORIES[:-1], default="other"),
        index=description.index,
    )


def history(frame: pd.DataFrame, cutoff: str | pd.Timestamp) -> pd.DataFrame:
    """Normalize descriptions using only records known before prediction time."""
    past = frame.loc[(frame.invoice_date < pd.Timestamp(cutoff)) & frame.customer_id.notna()].copy()
    lookup = past[past.description.ne("")].groupby("stock_code").description.agg(lambda x: x.mode().iloc[0])
    past["description"] = past.stock_code.map(lookup).fillna("")
    past["category"] = category(past.description)
    return past


def make_features(frame: pd.DataFrame, cutoff: str, cfg: dict[str, Any]) -> pd.DataFrame:
    """Aggregate one row per purchasing customer, excluding all future records."""
    date = pd.Timestamp(cutoff)
    past = history(frame, date)
    purchases = past[~past.is_return]
    orders = (
        purchases.groupby(["customer_id", "invoice"])
        .agg(date=("invoice_date", "min"), amount=("amount", "sum"), units=("quantity", "sum"))
        .reset_index()
    )
    grouped = orders.groupby("customer_id")
    features = grouped.agg(
        frequency=("invoice", "nunique"),
        monetary=("amount", "sum"),
        average_spend=("amount", "mean"),
        basket_size=("units", "mean"),
    )
    features["recency"] = (date - grouped.date.max()).dt.total_seconds() / 86400
    features["tenure"] = (date - grouped.date.min()).dt.total_seconds() / 86400
    features["historical_clv"] = past.groupby("customer_id").amount.sum()
    features["country"] = purchases.sort_values("invoice_date").groupby("customer_id").country.last()
    gaps = orders.sort_values("date").groupby("customer_id").date.diff().dt.total_seconds() / 86400
    features["gap_variance"] = gaps.groupby(orders.customer_id).var().fillna(0)
    recent = orders[orders.date >= date - pd.Timedelta(days=90)].groupby("customer_id")
    features["orders_90d"] = recent.size()
    features["spend_90d"] = recent.amount.sum()
    bins = ((date - orders.date).dt.total_seconds() / (86400 * 30)).astype(int)
    monthly = pd.crosstab(orders.customer_id[bins < 6], bins[bins < 6]).reindex(
        index=features.index, columns=range(5, -1, -1), fill_value=0
    )
    features["frequency_trend"] = monthly.to_numpy() @ (np.arange(6) - 2.5) / 17.5
    seasons = pd.crosstab(orders.customer_id, orders.date.dt.month)
    features["seasonal_concentration"] = seasons.max(axis=1) / seasons.sum(axis=1)
    returns = past[past.is_return].groupby("customer_id").quantity.sum().abs()
    features["return_rate"] = returns / purchases.groupby("customer_id").quantity.sum()
    products = purchases[purchases.is_product].copy()
    prices = products.groupby("stock_code").price.transform("median")
    products["discount"] = (products.price < prices * 0.9).astype(float)
    products["popularity"] = products.stock_code.map(products.stock_code.value_counts(normalize=True))
    features["discount_proxy"] = products.groupby("customer_id").discount.mean()
    features["sku_popularity"] = products.groupby("customer_id").popularity.mean()
    features["unique_products"] = products.groupby("customer_id").stock_code.nunique()
    affinity = pd.crosstab(products.customer_id, products.category).reindex(columns=CATEGORIES, fill_value=0)
    affinity = affinity.div(affinity.sum(axis=1), axis=0)
    for col in CATEGORIES:
        features["affinity_" + col] = affinity[col]
    features["engagement"] = (
        (
            np.exp(-features.recency / 90)
            + (1 - np.exp(-features.frequency / 10))
            + (1 - np.exp(-features.monetary / 2500))
        )
        / 3
        * 100
    )
    return features.fillna(0).rename_axis("customer_id").reset_index()[["customer_id", *FEATURES]]


def make_targets(frame: pd.DataFrame, customers: pd.Series, cutoff: str, horizon: int) -> pd.DataFrame:
    """Label churn, gross 90-day spend, capped waiting time and next proxy category."""
    start = pd.Timestamp(cutoff)
    end = start + pd.Timedelta(days=horizon)
    if frame.invoice_date.max() < end:
        raise ValueError("Outcome window is incomplete; labels would be censored")
    future = frame[
        (frame.invoice_date >= start) & (frame.invoice_date < end) & ~frame.is_return & frame.customer_id.notna()
    ].copy()
    outcome = pd.DataFrame(index=pd.Index(customers, name="customer_id"))
    group = future.groupby("customer_id")
    outcome["churn"] = (~outcome.index.isin(future.customer_id)).astype(int)
    outcome["clv_90d"] = group.amount.sum().reindex(outcome.index, fill_value=0)
    outcome["next_days"] = (
        ((group.invoice_date.min() - start).dt.total_seconds() / 86400).reindex(outcome.index).fillna(horizon)
    )
    products = future[future.is_product].copy()
    products["category"] = category(products.description)
    first = products.sort_values(["invoice_date", "invoice", "stock_code"]).groupby("customer_id").category.first()
    outcome["next_category"] = first.reindex(outcome.index).fillna("no_purchase")
    return outcome.reset_index()


def make_sequences(
    frame: pd.DataFrame, customers: pd.Series, cutoff: str, length: int
) -> tuple[np.ndarray, np.ndarray]:
    """Build event vectors from log amount, log gap, event age and category mix."""
    past = history(frame, cutoff)
    past = past[~past.is_return & past.is_product]
    orders = (
        past.groupby(["customer_id", "invoice"])
        .agg(date=("invoice_date", "min"), amount=("amount", "sum"))
        .reset_index()
    )
    cats = pd.crosstab([past.customer_id, past.invoice], past.category).reindex(columns=CATEGORIES, fill_value=0)
    cats = cats.div(cats.sum(axis=1), axis=0).reset_index()
    orders = orders.merge(cats, on=["customer_id", "invoice"]).sort_values("date")
    orders["gap"] = orders.groupby("customer_id").date.diff().dt.total_seconds().fillna(0) / 86400
    orders["age"] = (pd.Timestamp(cutoff) - orders.date).dt.total_seconds() / 86400
    groups = {str(k): g for k, g in orders.groupby("customer_id")}
    result = np.zeros((len(customers), length, 8), dtype="float32")
    lengths = np.ones(len(customers), dtype="int64")
    for i, customer in enumerate(customers):
        events = groups.get(str(customer))
        if events is not None:
            events = events.tail(length)
            values = np.column_stack(
                [np.log1p(events.amount) / 10, np.log1p(events.gap) / 7, np.log1p(events.age) / 7, events[CATEGORIES]]
            )
            result[i, : len(values)] = values
            lengths[i] = len(values)
    return result, lengths
