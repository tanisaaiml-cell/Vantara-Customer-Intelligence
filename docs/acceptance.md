# Acceptance status

- Churn AUC and recall: met.
- Value R²: not met by validation-selected model.
- Six classical churn and three neural models: trained and serialized.
- Five-fold training CV: implemented for supervised comparisons.
- Leakage, API, reproducibility and coverage: see test_results.txt and coverage.json.
- Dashboard and CSV/PDF downloads: see dashboard_tests.txt and live_checks.txt.
- PostgreSQL/Compose: implemented, clean-machine runtime remains unverified.
- Initial load may miss 3 seconds; local timings are in performance.json.
- Report, notebooks, explanations and diagrams: included.
- Transformer, CatBoost, supplementary demographic data: optional/deferred.
