"""Generate a reviewed technical report and exploratory notebooks from saved results."""

import json
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from src.utils.config import ROOT, artifacts
from src.utils.pdf_fonts import register_fonts


def make_notebooks() -> None:
    """Write notebooks that reuse production functions and existing evidence."""
    intro = "from pathlib import Path\nimport sys, json\nimport pandas as pd\nfrom IPython.display import display, Image\nROOT=Path.cwd().parent if Path.cwd().name=='notebooks' else Path.cwd()\nsys.path.insert(0,str(ROOT))\n"
    books = {
        "01_eda": [
            (
                "markdown",
                "# Online Retail II exploratory analysis\nInspect both source years through the cleaned ledger and pre-cutoff customer table.",
            ),
            (
                "code",
                intro
                + "ledger=pd.read_parquet(ROOT/'data/interim/transactions.parquet')\ncustomers=pd.read_parquet(ROOT/'data/processed/customers.parquet')\njson.loads((ROOT/'docs/data_audit.json').read_text())",
            ),
            ("code", "customers[['recency','frequency','monetary','return_rate']].describe()"),
            (
                "code",
                "display(Image(filename=str(ROOT/'docs/rfm_distributions.png')))\ndisplay(Image(filename=str(ROOT/'docs/country_seasonality.png')))\ndisplay(Image(filename=str(ROOT/'docs/correlations.png')))\npd.read_csv(ROOT/'docs/vif.csv')",
            ),
            (
                "markdown",
                "## Hypotheses\n1. Inactivity should predict churn: inspect SHAP/PDP associations.\n2. Recent spend should predict future spend: large wholesale outcomes may make R² unstable.\n3. Sequence models might improve timing: the trained LSTM does not beat Ridge.\n4. Customer groups differ by value/activity: compare K-Means and GMM.\n5. Returns may add information: an ablation study is still needed before claiming independent benefit.\n\nVIF flags redundant RFM features and affinity shares. Regularized baselines and tree models retain the required feature families.",
            ),
        ],
        "02_feature_engineering": [
            (
                "markdown",
                "# Point-in-time features\nPredict at 2011-09-01 using strictly earlier records. Future labels require a complete 90-day observation window.",
            ),
            (
                "code",
                intro
                + "from src.features.build import make_features,make_targets,make_sequences\nfrom src.utils.config import config\ncfg=config()\nledger=pd.read_parquet(ROOT/'data/interim/transactions.parquet')\nfeatures=make_features(ledger,cfg['cutoff'],cfg)\nfeatures.head()",
            ),
            (
                "code",
                "targets=make_targets(ledger,features.customer_id,cfg['cutoff'],cfg['horizon_days'])\ntargets.describe(include='all')",
            ),
            (
                "code",
                "before=ledger[ledger.invoice_date<pd.Timestamp(cfg['cutoff'])]\npd.testing.assert_frame_equal(features,make_features(before,cfg['cutoff'],cfg))\nsequence,lengths=make_sequences(ledger,features.customer_id.head(5),cfg['cutoff'],12)\nsequence.shape,lengths",
            ),
            (
                "markdown",
                "## Interpret carefully\nHistorical CLV is signed net spend. Predicted CLV is future gross spend over 90 days. Product groups and discount sensitivity are proxies. Read `docs/feature_dictionary.md` for exact formulas.",
            ),
        ],
        "03_model_experiments": [
            (
                "markdown",
                "# Saved model experiments\nSelection uses validation. Test metrics are final reporting, never a tuning signal.",
            ),
            (
                "code",
                intro
                + "meta=json.loads((ROOT/'models_artifacts/metadata.json').read_text())\ndisplay(pd.read_csv(ROOT/'docs/model_comparison.csv').sort_values('roc_auc',ascending=False))\nmeta['champion']",
            ),
            ("code", "pd.DataFrame([{'model':r['name'],**r['test']} for r in meta['clv_runs']])"),
            ("code", "meta['lstm'],meta['category'],meta['clustering']"),
            (
                "code",
                "display(Image(filename=str(ROOT/'docs/learning_curves.png')))\ndisplay(Image(filename=str(ROOT/'docs/cluster_selection.png')))\ndisplay(Image(filename=str(ROOT/'docs/shap_summary.png')))",
            ),
            (
                "markdown",
                "## Selection rationale\nRandom Forest wins churn validation AUC. Slightly higher XGBoost test AUC does not justify switching afterward. The same rule preserves Random Forest value despite higher Ridge test R². Value misses its target and LSTM misses baseline improvement; use a new time period for future experiments.",
            ),
        ],
    }
    for name, entries in books.items():
        cells = []
        for kind, content in entries:
            cell = {"cell_type": kind, "metadata": {}, "source": content.splitlines(keepends=True)}
            if kind == "code":
                cell.update(execution_count=None, outputs=[])
            cells.append(cell)
        (ROOT / "notebooks" / f"{name}.ipynb").write_text(
            json.dumps(
                {
                    "cells": cells,
                    "metadata": {
                        "kernelspec": {"display_name": "Python 3.11", "language": "python", "name": "python3"},
                        "language_info": {"name": "python", "version": "3.11.13"},
                    },
                    "nbformat": 4,
                    "nbformat_minor": 5,
                },
                indent=2,
            )
        )


def build() -> None:
    """Assemble ten pages of results, limitations, acceptance and mathematics."""
    register_fonts()
    meta = json.loads((artifacts() / "metadata.json").read_text())
    audit = json.loads((ROOT / "docs/data_audit.json").read_text())
    docs = ROOT / "docs"
    perf = json.loads((docs / "performance.json").read_text()) if (docs / "performance.json").exists() else {}
    coverage = (
        json.loads((docs / "coverage.json").read_text())["totals"]["percent_covered"]
        if (docs / "coverage.json").exists()
        else None
    )
    champion = next(r for r in meta["churn_runs"] if r["name"] == meta["champion"])
    value = next(r for r in meta["clv_runs"] if r["name"] == meta["clv_champion"])
    styles = getSampleStyleSheet()
    for name, font, size, leading in [
        ("TitleV", "VantaraBold", 26, 32),
        ("HeadingV", "VantaraBold", 15, 20),
        ("BodyV", "VantaraSans", 9.5, 14.5),
        ("SmallV", "VantaraSans", 7.5, 11),
    ]:
        styles.add(
            ParagraphStyle(
                name=name,
                fontName=font,
                fontSize=size,
                leading=leading,
                spaceAfter=10,
                textColor=colors.HexColor("#25304a"),
            )
        )
    story = []

    def text(content: str, style: str = "BodyV") -> None:
        story.append(Paragraph(content, styles[style]))

    def page(title: str) -> None:
        if story:
            story.append(PageBreak())
        text(title, "TitleV")

    def table(rows: list, widths: list) -> None:
        obj = Table(
            [[Paragraph(escape(str(c)), styles["SmallV"]) for c in row] for row in rows],
            colWidths=widths,
            repeatRows=1,
            hAlign="LEFT",
        )
        obj.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e7e6fb")),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d9ddea")),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING", (0, 0), (-1, -1), 8),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f6f7fb")]),
                ]
            )
        )
        story.extend([obj, Spacer(1, 12)])

    def picture(name: str, height: int = 260) -> None:
        p = docs / name
        if p.exists():
            w, h = ImageReader(str(p)).getSize()
            scale = min(499 / w, height / h)
            story.extend([Image(str(p), width=w * scale, height=h * scale, hAlign="LEFT"), Spacer(1, 12)])

    page("Customer behavior<br/>prediction platform")
    text(
        "A trained customer analytics prototype with a Streamlit dashboard, FastAPI backend and reproducible Python 3.11 pipeline."
    )
    text(
        "The churn model meets the required AUC and recall targets. The validation-selected customer-value model misses its R² target, and the LSTM does not beat its static baseline. These limitations remain visible in the application."
    )
    table(
        [
            ["Measure", "Held-out result", "Acceptance"],
            ["Churn ROC-AUC", f"{champion['test']['roc_auc']:.4f}", "0.80 target met"],
            ["Churn recall", f"{champion['test']['recall']:.2%}", "0.70 target met"],
            ["90-day value R²", f"{value['test']['r2']:.4f}", "0.60 target not met"],
            ["Source coverage", f"{coverage:.1f}%" if coverage else "See test evidence", "70% target"],
            ["Evaluation cohort", "788 test customers", "Disjoint customer IDs"],
        ],
        [160, 145, 194],
    )
    text("Delivered scope", "HeadingV")
    text(
        "The package includes both workbook sheets, clean data, feature and sequence tables, six classical churn models, ANN/LSTM/autoencoder artifacts, value and category models, two clustering approaches, SHAP/LIME explanations, dashboard, API, tests, notebooks, diagrams and Docker configuration."
    )
    text("Runtime boundary", "HeadingV")
    text(
        "Training and application execution use CPython 3.11.13 on a Linux CPU. Native API persistence is tested through SQLite. PostgreSQL schema and Compose configuration are included, but Docker is unavailable and local PostgreSQL initialization was blocked. Clean-machine container acceptance remains open. Windows launch instructions are provided but were not executed on a Windows host."
    )
    text(
        "This is a historical prediction snapshot at 1 September 2011. A 90-day gross-spend estimate is not lifetime profit. UI scores include training/validation customers; only explicitly held-out tables measure generalization."
    )
    page("Data foundation")
    table(
        [
            ["Audit item", "Observed"],
            ["Source transaction lines", f"{audit['raw_rows']:,}"],
            ["Exact duplicate rows", f"{audit['exact_duplicate_rows']:,}"],
            ["Invalid price / zero quantity", f"{audit['invalid_price_or_zero_quantity']:,}"],
            ["Extreme lines excluded", audit["quarantined_extremes"]],
            ["Clean ledger", f"{audit['clean_rows']:,}"],
            ["Missing Customer ID", f"{audit['missing_customer_rate']:.2%}"],
            ["Eligible snapshot customers", meta["customers"]],
        ],
        [280, 219],
    )
    text(
        "Returns retain signed negative amounts and produce a return-rate feature. IQR flags do not automatically remove wholesale purchases. Domain bounds remove absolute quantities above 50,000 and prices above GBP 20,000; original bytes and quarantined records are preserved. Exact identical invoice lines cannot be distinguished from legitimate repeated lines, which is a source limitation."
    )
    picture("rfm_distributions.png", 160)
    text(
        "Anonymous records remain in the clean ledger but cannot join customer models. Administrative StockCodes are excluded from product features while retained in accounting. Descriptions are normalized with a per-SKU modal lookup fitted only before cutoff. Correlation, country, seasonality and VIF assets accompany the notebooks."
    )
    text(
        "Actual source dates run from December 2009 to December 2011 and do not span a leap year. Churn prevalence is 57.31%, so the PRD minority-class assumption is not true for this snapshot."
    )
    page("Evaluation design and churn")
    text(
        "Features use timestamps strictly earlier than 2011-09-01. Labels cover [2011-09-01, 2011-11-30), a complete 90-day interval. The 70/15/15 stratified split has 3,674 training, 787 validation and 788 test customers, seed 42. No customer overlaps partitions. All learned encoders/scalers are refit within training CV folds."
    )
    rows = [["Model", "CV AUC", "Test AUC", "Recall", "Precision", "F1", "Accuracy"]]
    for run in sorted(meta["churn_runs"], key=lambda r: (r["test"]["roc_auc"], r["test"]["recall"]), reverse=True):
        m = run["test"]
        rows.append(
            [
                run["name"],
                f"{run['cv_auc_mean']:.3f}",
                *[f"{m[k]:.3f}" for k in ["roc_auc", "recall", "precision", "f1", "accuracy"]],
            ]
        )
    table(rows, [115, 60, 64, 59, 68, 58, 75])
    text(
        f"Random Forest wins validation AUC ({champion['validation']['roc_auc']:.4f}). Its {meta['threshold']:.0%} threshold maximizes validation F1 subject to recall >=0.70. Test confusion matrix: TN=171, FP=165, FN=47, TP=405. A slightly higher XGBoost test AUC does not justify retrospective model switching."
    )
    text(
        "Random Forest and XGBoost searches use five-fold training CV. XGBoost early stopping uses validation. Neural CV reserves an internal stopping split separate from the scored fold. ANN CV uses up to 15 epochs and the final fit up to 35; validation loss controls early stopping."
    )
    text(
        "The customer-holdout design follows the PRD but does not validate future-time drift. Shared pre-cutoff catalog statistics are operationally known. A genuinely later untouched period is needed for temporal deployment claims."
    )
    page("Customer value and purchase behavior")
    rows = [["Value model", "Val R²", "Test R²", "MAE GBP", "RMSE GBP"]]
    for run in meta["clv_runs"]:
        m = run["test"]
        rows.append(
            [run["name"], f"{run['validation']['r2']:.3f}", f"{m['r2']:.3f}", f"{m['mae']:,.2f}", f"{m['rmse']:,.2f}"]
        )
    table(rows, [155, 67, 67, 105, 105])
    text(
        "The regression target is untrimmed future 90-day gross spend. Random Forest wins validation but generalizes less well on rare wholesale outcomes. Ridge exceeds 0.60 test R² but was not selected using validation, so it is not promoted after test inspection. No high-error customers are removed to improve the reported score."
    )
    text(
        f"LSTM capped-wait test MAE is {meta['lstm']['test']['mae']:.2f} days, compared with {meta['lstm']['baseline_test']['mae']:.2f} for Ridge. The neural sequence model has not demonstrated improvement over static features."
    )
    picture("learning_curves.png", 160)
    text(
        f"Next proxy-category classification has test accuracy {meta['category']['test_accuracy']:.3f} and macro-F1 {meta['category']['test_macro_f1']:.3f}. Purchase probability equals one minus churn probability for the same outcome horizon. Recommendations suggest unseen popular products in the predicted description-derived category. Ranking quality is not evaluated."
    )
    page("Segmentation and explainability")
    km = max([r for r in meta["clustering"] if r["model"] == "KMeans"], key=lambda x: x["silhouette"])
    gm = next(r for r in meta["clustering"] if r["model"] == "GaussianMixture")
    text(
        f"K-Means chooses k={km['k']} from 2-6 using training silhouette, supported by an elbow plot. Silhouette={km['silhouette']:.3f}; Davies-Bouldin={km['davies_bouldin']:.3f}. The Gaussian mixture comparison has silhouette={gm['silhouette']:.3f} and Davies-Bouldin={gm['davies_bouldin']:.3f}. Profiles use historical activity/value; individual members may differ from cluster medians."
    )
    picture("cluster_selection.png", 140)
    picture("shap_summary.png", 270)
    text(
        "The API supplies SHAP and LIME for any saved customer. Three low/borderline/high-risk examples include force HTML files, waterfall PNGs and JSON explanations. Positive contributions raise predicted risk relative to the model baseline. LIME is an approximation with a reported local fit score. Attribution is not a causal effect of offering a discount."
    )
    page("Architecture and application")
    picture("architecture_diagram.png", 300)
    text(
        "Streamlit is an HTTP-only client: it never calls a model directly. The API loads immutable artifacts, reads stored customer scores and produces on-demand explanations. Full feature CSVs run new tabular inference and persist batch history; ID-only CSVs retrieve the saved snapshot."
    )
    text(
        "The relational schema contains customers, predictions, segments and invoice-level transactions, plus batch_scores. The clean line-item ledger remains in Parquet. First startup seeds the historical snapshot idempotently; intentional retraining requires an explicit database refresh. Scoring and query behavior is covered through the same SQLAlchemy schema with SQLite."
    )
    text(
        "The dashboard offers an overview, segment/country/value filters, high-risk value priorities, customer search and explanations, model metrics, CSV uploads/downloads and customer PDFs. A three-month moving-average revenue overlay is labeled as a baseline. It has no validated forecast interval."
    )
    page("Testing and acceptance")
    table(
        [
            ["Check", "Evidence"],
            ["Source test coverage", f"{coverage:.1f}%" if coverage else "See coverage.json"],
            ["Reproducibility", "Full retraining in an isolated temporary directory"],
            ["API contracts", "Health, validation, lookup, batch persistence and downloads"],
            ["Live UI", "Five HTTP-backed Streamlit page tests and Chromium upload/download journey"],
            ["Saved lookup p95", f"{perf.get('api_single_lookup_p95_ms', float('nan')):.2f} ms, 100 local requests"],
            ["Initial dashboard load", f"{perf.get('initial_dashboard_seconds', float('nan')):.2f} seconds"],
            [
                "Warm segment navigation",
                f"{perf.get('page_navigation_seconds', {}).get('Customer segments', float('nan')):.2f} seconds",
            ],
            ["Docker and PostgreSQL", "Configuration implemented; clean-machine execution unverified"],
        ],
        [175, 324],
    )
    text(
        "Tests exercise signed returns, RFM arithmetic, future-data perturbations, exact date boundaries, padded sequences, disjoint IDs, malformed CSV, unseen countries, explanations, serialized inference and database persistence. Full retraining uses identical frozen settings and checks saved probabilities/value/timing predictions; it never tunes using the test partition."
    )
    text(
        "Local performance is not a production SLA. Measurements use warm SQLite and a CPU browser session, not concurrent PostgreSQL traffic. Initial loading can exceed the 3-second target; warm segment navigation is reported separately. Explanation latency is outside the stored-score lookup benchmark."
    )
    text("Remaining work", "HeadingV")
    text(
        "The selected CLV model misses its target and the LSTM misses baseline improvement. No labeled fraud performance exists. Future work needs temporal validation, value-model robustness, calibrated probabilities and real category/discount data. Authentication, role-based access, CI/CD, streaming, A/B tests and Transformers remain deferred as allowed by the PRD."
    )
    page("Mathematical appendix 1<br/>Features and classical models")
    text("Customer summaries", "HeadingV")
    text(
        "For customer i at cutoff c, the history contains only timestamps t &lt; c. Recency R=(c-max t)/day; frequency F counts distinct positive invoices; gross monetary M sums positive quantity times price. Historical net value includes signed returns. Recent features restrict the same history to 90 days."
    )
    text(
        "Trend uses six chronological 30-day counts n_j. Centered time x_j=j-2.5 gives the least-squares slope sum(x_j n_j)/17.5. Gap variance is the sample variance of consecutive inter-order days. Engagement = 100/3 [exp(-R/90)+1-exp(-F/10)+1-exp(-M/2500)]. This composite is fixed, not fitted using future outcomes."
    )
    text("Logistic regression and Ridge", "HeadingV")
    text(
        "The churn logit z=w^T x+b produces p=1/(1+exp(-z)). Weighted cross-entropy is -mean[a y log(p)+b(1-y)log(1-p)], plus an L2 regularization penalty. Balanced weights come from training prevalence. C controls inverse regularization. Ridge minimizes squared regression error plus alpha times the sum of squared coefficients."
    )
    text(
        "Correlated RFM components and category shares can have high or infinite VIF. The pipeline exports VIF and retains these PRD feature families for tree models and regularized baselines. Their coefficients should not be interpreted as causal effects."
    )
    text("Preprocessing and ensembles", "HeadingV")
    text(
        "StandardScaler computes z_j=(x_j-mu_j)/sigma_j with statistics fit only on training folds. Median imputation and one-hot country encoding are persisted. Unknown countries are ignored. Random Forest averages randomized trees, while boosting adds weak learners along objective gradients. XGBoost stopping uses validation rather than test records."
    )
    page("Mathematical appendix 2<br/>Neural and sequential models")
    text("ANN optimization", "HeadingV")
    text(
        "The ANN combines affine layers, ReLU, batch normalization and dropout, ending in a churn logit. Backpropagation applies the chain rule to compute gradients through each layer. Adam updates weights using running first and second gradient moments; the delivered learning rate is 0.001 and weight decay 0.0001."
    )
    text(
        "BCEWithLogits combines sigmoid and cross-entropy stably. Positive-class weight is negative-count/positive-count in the training rows. Dropout is active only during training. Early stopping restores the state with minimum validation loss after six non-improving epochs. Curves for training/validation are included for all neural models."
    )
    text("Purchase-event LSTM", "HeadingV")
    text(
        "Each sequence has at most 12 product invoices. Event inputs contain log amount, log gap, age at cutoff and five proxy-category shares. Packed lengths ensure zero padding does not become a fabricated event. The LSTM maintains hidden h_t and memory c_t, with sigmoid input/forget/output gates and a tanh candidate memory update."
    )
    text(
        "The final hidden state enters dropout and a sigmoid head. Output times 90 predicts min(waiting time,90). Mean squared error trains the normalized target. Customers without a future purchase receive the 90-day cap, so this is not uncensored survival estimation."
    )
    text("Spending autoencoder", "HeadingV")
    text(
        "The network compresses d inputs through d -> 16 -> 5, then reconstructs through 5 -> 16 -> d. Per-customer reconstruction error is mean_j[(x_j-reconstructed_x_j)^2]. A validation 95th-percentile threshold requests manual review. Without fraud labels, an anomaly flag cannot establish fraud precision or recall."
    )
    page("Mathematical appendix 3<br/>Evaluation and interpretation")
    text("Metrics", "HeadingV")
    text(
        "Precision=TP/(TP+FP), recall=TP/(TP+FN), F1=2 precision recall/(precision+recall), and accuracy=(TP+TN)/N. ROC-AUC measures ranking over thresholds. Confusion-matrix rows are actual purchase/churn and columns are predicted purchase/churn. Validation fixes the deployed decision threshold before test scoring."
    )
    text(
        "MAE=mean(abs(y-prediction)); RMSE=sqrt(mean((y-prediction)^2)); R²=1-sum((y-prediction)^2)/sum((y-mean(y))^2). R² can be negative. Large wholesale purchases dominate squared error, so MAE/RMSE remain reported in original currency without target trimming."
    )
    text("Clustering and attributions", "HeadingV")
    text(
        "K-Means minimizes within-cluster squared Euclidean distances on standardized log features. Silhouette=(b-a)/max(a,b), where a is within-cluster distance and b the nearest alternative-cluster distance; higher is better. Davies-Bouldin compares within-cluster scatter to separation; lower is better. Gaussian mixtures allow overlapping elliptical groups."
    )
    text(
        "SHAP distributes model-output differences relative to a baseline across input features. TreeSHAP for the deployed Random Forest explains churn probability. LIME fits an interpretable local surrogate and exposes its local R². Neither predicts the causal effect of a retention campaign."
    )
    text("Provenance", "HeadingV")
    text(
        "Chen, D. (2012). Online Retail II [Dataset]. UCI Machine Learning Repository. DOI 10.24432/C5CG6D. https://archive.ics.uci.edu/dataset/502/online+retail+ii. CC BY 4.0. Source bytes are the supplied workbook. Requirements are the supplied Vantara PRD v1.0. Metadata records versions, configuration, seed and the feature-table SHA-256 hash."
    )
    text(
        "Implementation guidance: scikit-learn Common pitfalls and recommended practices, https://scikit-learn.org/stable/common_pitfalls.html.",
        "SmallV",
    )

    def decorate(canvas: object, doc: object) -> None:
        canvas.saveState()
        canvas.setFont("VantaraSans", 7)
        canvas.setFillColor(colors.HexColor("#7c849b"))
        canvas.drawString(48, 810, "VANTARA / CUSTOMER INTELLIGENCE / TECHNICAL REPORT")
        canvas.drawString(48, 30, "Python 3.11 | Online Retail II | Historical prototype")
        canvas.drawRightString(547, 30, str(doc.page))
        canvas.restoreState()

    SimpleDocTemplate(
        str(docs / "final_report.pdf"),
        pagesize=(595.28, 841.89),
        leftMargin=48,
        rightMargin=48,
        topMargin=58,
        bottomMargin=50,
        title="Vantara Customer Behavior Prediction Platform",
    ).build(story, onFirstPage=decorate, onLaterPages=decorate)
    make_notebooks()
    (docs / "acceptance.md").write_text(
        "# Acceptance status\n\n- Churn AUC and recall: met.\n- Value R²: not met by validation-selected model.\n- Six classical churn and three neural models: trained and serialized.\n- Five-fold training CV: implemented for supervised comparisons.\n- Leakage, API, reproducibility and coverage: see test_results.txt and coverage.json.\n- Dashboard and CSV/PDF downloads: see dashboard_tests.txt and live_checks.txt.\n- PostgreSQL/Compose: implemented, clean-machine runtime remains unverified.\n- Initial load may miss 3 seconds; local timings are in performance.json.\n- Report, notebooks, explanations and diagrams: included.\n- Transformer, CatBoost, supplementary demographic data: optional/deferred.\n"
    )


if __name__ == "__main__":
    build()
