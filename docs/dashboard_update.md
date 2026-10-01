# Dashboard update — 30 September 2026

This update adds original-unit model error analysis and a processed-data browser. The saved models, frozen split and original datasets are unchanged.

## What changed

- Model performance now includes MAE, MSE, RMSE, R², median absolute error, signed error and the share of squared error attributable to the worst 10 predictions.
- Select spend or capped purchase waiting time; view all held-out actual-versus-predicted points, signed errors, largest misses, and download row-level errors.
- Churn mistakes reports classification error, false positives and false negatives separately from regression metrics.
- Processed data previews the first 100 feature rows and exports all 5,249 rows as feature-only, modeling-with-targets, or scored CSVs via FastAPI.
- The preparation stage also creates feature-only and modeling CSVs. Existing Parquet files remain canonical; CSV export does not require retraining.
- `/evaluation/errors` checks the feature-table hash, partition coverage and agreement with saved metrics. Mismatched artifacts produce an explicit error.

## Measured errors from the frozen test partition

| Metric | 90-day gross spend | Capped purchase wait |
|---|---:|---:|
| Test customers | 788 | 788 |
| MAE | GBP 709.9866 | 20.5619 days |
| MSE | 19,586,670.8741 GBP² | 648.2228 days² |
| RMSE | GBP 4,425.6831 | 25.4602 days |
| R² | 0.481966 | 0.325240 |
| Median absolute error | GBP 148.8812 | 15.5340 days |
| Mean predicted-minus-actual | GBP -213.1789 | -1.0765 days |
| Worst 10 share of squared error | 97.8364% | 11.0277% |

Churn: 212 mistakes / 788 = 26.9036% error, with 165 false churn alerts and 47 missed churners, at the existing 0.36 operating threshold.

The concentration of squared error shows why spend RMSE is much larger than MAE. It does not prove a causal reason for poor generalization. R² is not percentage accuracy. Zero future-spend targets make MAPE undefined, so it is omitted. All targets and errors remain untrimmed. No selections are made using this test analysis; future modeling changes need a new untouched evaluation period.

## Applying the small update ZIP on Windows

1. Stop FastAPI and Streamlit with Ctrl+C in their terminals.
2. Extract `Vantara_Dashboard_Update.zip` into a temporary folder.
3. Copy the CONTENTS of its `customer-behavior-prediction` folder into your existing project folder (the one containing `README.md`, `api`, `frontend` and your `.venv`). Merge folders and replace matching files. Do not create another nested project folder.
4. Restart both services using the existing environment:

```powershell
# Terminal 1, from the project folder
.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

```powershell
# Terminal 2, from the same project folder
.\.venv\Scripts\python.exe -m streamlit run frontend/dashboard.py
```

5. Open http://localhost:8501 and click Refresh data. Open Model performance and Processed data in the sidebar.

No new packages, model retraining, or database refresh are needed. The small ZIP requires the previously delivered full project. The complete project ZIP is also updated for a fresh extraction.

The original report and original QA files describe the baseline delivery. See `update_test_results.txt`, `update_coverage.json`, and `dashboard_tests.txt` for this update's verification. Windows was not available for native execution; verification uses CPython 3.11.13 on Linux and SQLite.

## Numerical reproducibility

The initial rerun differed from the saved LSTM float32 outputs by at most 0.000019073486 days (about 1.65 seconds), while tabular churn/value predictions passed their existing tolerances. The neural reproducibility check now uses rtol=1e-6 and atol=1e-5 days, less than 9 seconds at the 90-day cap. Test MAE/RMSE/R² are also checked at rtol=1e-6, atol=1e-7. This accommodates floating-point kernel variation across CPUs; no model artifact, target, split or performance threshold was changed.

## Completed verification

CPython 3.11.13: 47 core/integration tests passed, including full retraining; 7 live Streamlit tests passed separately. Source coverage: 92.78%. Ruff passed. Chromium navigated all six pages, downloaded test errors and processed features, uploaded a batch, and downloaded scores and a PDF; no browser errors were reported. Screenshots of the error cards and processed-data page were visually reviewed.
