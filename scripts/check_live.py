"""Run API, Streamlit, page tests and a real Chromium workflow in one process tree."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import requests
from playwright.sync_api import sync_playwright


def check() -> None:
    """Capture screenshots, a silent walkthrough and measured local response times."""
    root = Path(__file__).resolve().parents[1]
    logs = [open(root / "docs/live_api.log", "w"), open(root / "docs/live_dashboard.log", "w")]
    api = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", "8000"],
        cwd=root,
        stdout=logs[0],
        stderr=subprocess.STDOUT,
    )
    ui = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "frontend/dashboard.py",
            "--server.address",
            "127.0.0.1",
            "--server.port",
            "8501",
        ],
        cwd=root,
        stdout=logs[1],
        stderr=subprocess.STDOUT,
    )
    try:
        for url in ["http://127.0.0.1:8000/health", "http://127.0.0.1:8501/_stcore/health"]:
            for _ in range(90):
                try:
                    if requests.get(url, timeout=2).ok:
                        break
                except requests.RequestException:
                    pass
                time.sleep(1)
            else:
                raise RuntimeError("Server did not start: " + url)
        with open(root / "docs/dashboard_tests.txt", "w") as output:
            subprocess.run(
                [sys.executable, "-m", "pytest", "tests/test_dashboard.py", "-q"],
                env={**os.environ, "TEST_DASHBOARD": "1"},
                cwd=root,
                stdout=output,
                stderr=subprocess.STDOUT,
                check=True,
            )
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
            context = browser.new_context(
                viewport={"width": 1440, "height": 1080},
                record_video_dir=str(root / "docs/walkthrough"),
                record_video_size={"width": 1440, "height": 1080},
                accept_downloads=True,
            )
            page = context.new_page()
            errors = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            start = time.perf_counter()
            page.goto("http://127.0.0.1:8501")
            page.get_by_text("Retention priorities", exact=True).wait_for(timeout=120000)
            initial = time.perf_counter() - start
            page.screenshot(path=str(root / "docs/dashboard_overview.png"), full_page=True)
            timings = {}
            for label, ready, name in [
                ("Customer segments", "Segment profiles", "segments"),
                ("Customer explorer", "Suggested products", "customer"),
                ("Model performance", "Held-out confusion matrix", "models"),
                ("Processed data", "Feature preview", "processed"),
                ("Batch scoring", "Download ID template", "batch"),
            ]:
                start = time.perf_counter()
                page.get_by_test_id("stSidebar").get_by_text(label, exact=True).click()
                page.get_by_text(ready, exact=True).wait_for(timeout=120000)
                timings[label] = time.perf_counter() - start
                page.wait_for_function(
                    "document.querySelector('[data-testid=stApp]').getAttribute('data-test-script-state') === 'notRunning'",
                    timeout=120000,
                )
                page.locator('[data-testid="stMain"]').evaluate("el => el.scrollTo(0, 0)")
                page.screenshot(path=str(root / f"docs/dashboard_{name}.png"), full_page=True)
                if name == "models":
                    page.get_by_text("Model error analysis", exact=True).scroll_into_view_if_needed()
                    page.screenshot(path=str(root / "docs/dashboard_errors.png"))
                    with page.expect_download() as info:
                        page.get_by_role("button", name="Download test errors CSV", exact=True).click()
                    info.value.save_as(str(root / "docs/value_test_errors.csv"))
                if name == "processed":
                    with page.expect_download() as info:
                        page.get_by_role("button", name="Download processed CSV", exact=True).click()
                    info.value.save_as(str(root / "docs/processed_download_example.csv"))
                if name == "customer":
                    with page.expect_download() as info:
                        page.get_by_role("button", name="Download customer PDF", exact=True).click()
                    info.value.save_as(str(root / "docs/customer_example.pdf"))
            page.get_by_test_id("stFileUploader").locator("input[type=file]").set_input_files(
                {"name": "ids.csv", "mimeType": "text/csv", "buffer": b"customer_id\n12347\n12348\n"}
            )
            page.get_by_role("button", name="Score customers", exact=True).click()
            page.get_by_text("2 customers scored and saved.", exact=True).wait_for(timeout=30000)
            with page.expect_download() as info:
                page.get_by_role("button", name="Download scores CSV", exact=True).click()
            info.value.save_as(str(root / "docs/batch_download_example.csv"))
            page.screenshot(path=str(root / "docs/dashboard_batch_results.png"), full_page=True)
            context.close()
            browser.close()
        session = requests.Session()
        durations = []
        for _ in range(100):
            start = time.perf_counter()
            response = session.post("http://127.0.0.1:8000/predict", json={"customer_id": "12347"})
            response.raise_for_status()
            durations.append((time.perf_counter() - start) * 1000)
        result = {
            "initial_dashboard_seconds": initial,
            "page_navigation_seconds": timings,
            "api_single_lookup_p95_ms": float(np.quantile(durations, 0.95)),
            "api_requests": len(durations),
            "browser_errors": errors,
            "environment": "Python 3.11.13, localhost, SQLite, warm API. Includes real browser rendering. PostgreSQL and Docker unverified.",
        }
        (root / "docs/performance.json").write_text(json.dumps(result, indent=2))
        print(result)
        assert not errors
    finally:
        api.terminate()
        ui.terminate()
        api.wait(timeout=20)
        ui.wait(timeout=20)
        for output in logs:
            output.close()


if __name__ == "__main__":
    check()
