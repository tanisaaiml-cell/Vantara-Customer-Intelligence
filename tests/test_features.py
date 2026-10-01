"""Arithmetic, date boundary, schema and leakage prevention tests."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.data.pipeline import clean_transactions
from src.features.build import FEATURES, category, make_features, make_sequences, make_targets
from src.utils.config import config


def test_rfm_and_returns(transactions_sample):
    f = clean_transactions(transactions_sample, config())[0]
    x = make_features(f, "2011-04-01", config()).set_index("customer_id")
    assert x.loc["1", "frequency"] == 2
    assert x.loc["1", "monetary"] == 20
    assert x.loc["1", "historical_clv"] == 15
    assert x.loc["1", "recency"] == 59
    assert x.loc["1", "return_rate"] == pytest.approx(1 / 3)
    assert x.loc["1", "affinity_kitchen"] == pytest.approx(0.5)


def test_future_mutation_invariant(transactions_sample):
    f = clean_transactions(transactions_sample, config())[0]
    changed = f.copy()
    mask = changed.invoice_date >= pd.Timestamp("2011-04-01")
    changed.loc[mask, "amount"] = 10**9
    changed.loc[mask, "description"] = "CHRISTMAS BIG DISCOUNT"
    changed.loc[mask, "price"] = 9999
    assert_frame_equal(make_features(f, "2011-04-01", config()), make_features(changed, "2011-04-01", config()))
    assert_frame_equal(make_features(f, "2011-04-01", config()), make_features(f[~mask], "2011-04-01", config()))


def test_targets_boundaries(transactions_sample):
    f = clean_transactions(transactions_sample, config())[0]
    y = make_targets(f, pd.Series(["1", "2"]), "2011-04-01", 90).set_index("customer_id")
    assert y.loc["1", "churn"] == 0 and y.loc["1", "clv_90d"] == 15 and y.loc["1", "next_days"] == 0
    assert y.loc["2", "churn"] == 1 and y.loc["2", "next_days"] == 90
    with pytest.raises(ValueError, match="incomplete"):
        make_targets(f, pd.Series(["1"]), "2011-07-01", 90)


def test_sequences_ignore_future_and_pad(transactions_sample):
    f = clean_transactions(transactions_sample, config())[0]
    ids = pd.Series(["1", "2", "999"])
    x, lengths = make_sequences(f, ids, "2011-04-01", 5)
    xx, ll = make_sequences(f[f.invoice_date < pd.Timestamp("2011-04-01")], ids, "2011-04-01", 5)
    np.testing.assert_array_equal(x, xx)
    np.testing.assert_array_equal(lengths, ll)
    assert x.shape == (3, 5, 8) and lengths.tolist() == [2, 1, 1] and np.all(x[0, 2:] == 0)


def test_dedup_bulk_extremes(transactions_sample):
    f = pd.concat([transactions_sample, transactions_sample.iloc[[0]]], ignore_index=True)
    f.loc[1, "quantity"] = 500
    f.loc[3, "quantity"] = 60000
    out, a = clean_transactions(f, config())
    assert (
        a["exact_duplicate_rows"] == 1
        and a["quarantined_extremes"] == 1
        and 500 in out.quantity.values
        and out.is_return.sum() == 1
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("price", np.inf),
        ("quantity", "bad"),
        ("invoice_date", "invalid"),
        ("invoice_date", "2020-01-01"),
        ("customer_id", None),
    ],
)
def test_validation_errors(transactions_sample, key, value):
    f = transactions_sample.copy()
    f[key] = value
    with pytest.raises(ValueError):
        clean_transactions(f, config())


def test_invalid_schema(transactions_sample):
    with pytest.raises(ValueError):
        clean_transactions(transactions_sample.drop(columns="price"), config())


def test_category_rules():
    assert category(pd.Series(["HEART", "TEA CUP", "CHRISTMAS", "BAG", "UNKNOWN"])).tolist() == [
        "home",
        "kitchen",
        "seasonal",
        "accessories",
        "other",
    ]


def test_disjoint_customer_partitions(real_customers):
    groups = [set(real_customers[real_customers.partition == p].customer_id) for p in ["train", "validation", "test"]]
    assert not groups[0] & groups[1] and not groups[1] & groups[2] and not groups[0] & groups[2]
    assert sum(map(len, groups)) == len(real_customers)
    assert not {"churn", "clv_90d", "next_days", "next_category"} & set(FEATURES)
