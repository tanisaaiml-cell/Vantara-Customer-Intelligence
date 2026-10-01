"""Page execution against an actual running API, with error banners rejected."""

import os

import pytest
from streamlit.testing.v1 import AppTest


@pytest.mark.skipif(os.getenv("TEST_DASHBOARD") != "1", reason="Requires a running HTTP API")
@pytest.mark.parametrize(
    "page",
    ["Overview", "Customer segments", "Customer explorer", "Model performance", "Processed data", "Batch scoring"],
)
def test_dashboard_pages(page):
    at = AppTest.from_file("frontend/dashboard.py", default_timeout=120).run()
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception and not at.error and len(at.sidebar.radio) == 1


@pytest.mark.skipif(os.getenv("TEST_DASHBOARD") != "1", reason="Requires a running HTTP API")
def test_error_metrics_and_processed_choices():
    at = AppTest.from_file("frontend/dashboard.py", default_timeout=120).run()
    at.sidebar.radio[0].set_value("Model performance").run()
    assert any(m.label == "MSE" for m in at.metric)
    at.selectbox[0].set_value("Capped purchase wait (days)").run()
    assert not at.exception and not at.error
    assert any(m.label == "MAE" and m.value == "20.56" for m in at.metric)
    at.sidebar.radio[0].set_value("Processed data").run()
    for choice in ("Features only", "Features + target labels", "Scored snapshot"):
        at.selectbox[0].set_value(choice).run()
        assert not at.exception and not at.error
        assert len(at.warning) == (choice != "Features only")
