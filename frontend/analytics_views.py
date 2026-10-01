"""HTTP-only dashboard views for held-out errors and processed data downloads."""

import os

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

API = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")


@st.cache_data(ttl=120, show_spinner=False)
def fetch(path: str, binary: bool = False):
    """Cache API responses; this UI never opens model or dataset files."""
    response = requests.get(API + path, timeout=60)
    response.raise_for_status()
    return response.content if binary else response.json()


def plot(fig):
    """Match the existing dashboard's quiet purple/teal styling."""
    fig.update_layout(
        height=360,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#59647e"),
        margin=dict(l=15, r=15, t=25, b=35),
    )
    fig.update_xaxes(gridcolor="#f0f2f8")
    fig.update_yaxes(gridcolor="#f0f2f8")
    st.plotly_chart(fig, use_container_width=True)


def render_error_analysis(meta: dict) -> None:
    """Explain original-unit regression errors and classification mistakes."""
    st.divider()
    st.subheader("Model error analysis")
    st.caption("Frozen test customers only. No retraining, model selection, or threshold tuning occurs here.")
    try:
        report = fetch("/evaluation/errors")
    except requests.RequestException:
        st.error("Error analysis is unavailable. Update and restart FastAPI; check /evaluation/errors for details.")
        return
    regression_tab, churn_tab = st.tabs(["Spend & timing errors", "Churn mistakes"])
    with regression_tab:
        target = st.selectbox("Prediction to inspect", ["90-day spend (GBP)", "Capped purchase wait (days)"])
        task = "value" if target.startswith("90") else "timing"
        unit = "£" if task == "value" else "days"
        result = report[task]
        metrics = result["summary"]
        rows = pd.DataFrame(result["rows"])
        a, b, c, d = st.columns(4)
        a.metric("MAE", f"{metrics['mae']:,.2f}", help=f"Mean absolute error in {result['unit']}. Lower is better.")
        b.metric(
            "MSE",
            f"{metrics['mse'] / 1e6:.2f}M" if metrics["mse"] >= 1e6 else f"{metrics['mse']:,.2f}",
            help=f"Mean squared error in {result['unit']} squared. Lower is better.",
        )
        c.metric(
            "RMSE", f"{metrics['rmse']:,.2f}", help=f"Square root of MSE, in {result['unit']}. Emphasizes large misses."
        )
        d.metric(
            "R²",
            f"{metrics['r2']:.3f}",
            help="1 is perfect; 0 matches predicting the test-target mean; negative is worse.",
        )
        st.caption(
            f"{metrics['count']} test customers · MAE/RMSE units: {result['unit']} · "
            f"Exact MSE: {metrics['mse']:,.2f} {result['unit']}²."
        )
        st.markdown(
            f"The average absolute miss is **{metrics['mae']:,.2f} {result['unit']}**, and the median miss is "
            f"**{metrics['median_absolute_error']:,.2f} {result['unit']}**. The 10 largest errors account for "
            f"**{metrics['worst_10_squared_error_share']:.1%}** of total squared error. "
            f"Mean signed error is **{metrics['mean_error']:+,.2f} {result['unit']}** "
            "(positive means overprediction)."
        )
        st.info(
            "These statistics show the size and concentration of the mistakes. They do not prove why a model failed. "
            "Use the plots and largest-error rows to investigate; R² is not percentage accuracy."
        )
        left, right = st.columns(2)
        with left:
            st.markdown("**Actual versus predicted**")
            fig = px.scatter(
                rows,
                x="actual",
                y="predicted",
                hover_data=["customer_id"],
                color_discrete_sequence=["#6258e8"],
                labels={"actual": f"Actual ({unit})", "predicted": f"Predicted ({unit})"},
            )
            low, high = (
                float(rows[["actual", "predicted"]].min().min()),
                float(rows[["actual", "predicted"]].max().max()),
            )
            fig.add_shape(type="line", x0=low, y0=low, x1=high, y1=high, line=dict(color="#18a999", dash="dash"))
            plot(fig)
            st.caption("The dashed line represents a perfect prediction. Every test customer is shown.")
        with right:
            st.markdown("**Signed prediction errors**")
            fig = px.histogram(
                rows,
                x="error",
                nbins=40,
                color_discrete_sequence=["#18a999"],
                labels={"error": f"Predicted − actual ({unit})"},
            )
            fig.add_vline(x=0, line_dash="dash", line_color="#6258e8")
            plot(fig)
            st.caption("Negative: underprediction. Positive: overprediction. Outcomes and errors are not trimmed.")
        st.markdown("**Largest held-out errors**")
        st.dataframe(
            rows.sort_values("absolute_error", ascending=False).head(15), hide_index=True, use_container_width=True
        )
        st.download_button("Download test errors CSV", rows.to_csv(index=False), f"{task}_test_errors.csv", "text/csv")
        if task == "value":
            st.markdown("**Value model comparison**")
            comparison = []
            for run in meta["clv_runs"]:
                for split in ("validation", "test"):
                    scores = run[split]
                    comparison.append(
                        {
                            "Model": run["name"],
                            "Partition": split,
                            "MAE (GBP)": scores["mae"],
                            "MSE (GBP²)": scores.get("mse", scores["rmse"] ** 2),
                            "RMSE (GBP)": scores["rmse"],
                            "R²": scores["r2"],
                        }
                    )
            st.dataframe(pd.DataFrame(comparison), hide_index=True, use_container_width=True)
            st.caption("Saved comparison MSE is RMSE². The deployed model remains the validation-selected model.")
            st.caption("MAPE is omitted because many customers have zero actual future spend, making it undefined.")
        else:
            baseline = meta["lstm"]["baseline_test"]
            st.info(
                f"Static Ridge baseline test MAE: {baseline['mae']:.2f} days; "
                f"LSTM test MAE: {metrics['mae']:.2f} days. Lower is better."
            )
            st.caption(
                f"Wait targets are capped at {meta['horizon_days']} days; this is not an uncensored time-to-purchase model."
            )
    with churn_tab:
        churn = report["churn"]
        a, b, c, d = st.columns(4)
        a.metric("Classification error", f"{churn['error_rate']:.1%}")
        b.metric("Incorrect predictions", f"{churn['incorrect']} / {report['count']}")
        c.metric("False churn alerts", churn["false_positives"])
        d.metric("Missed churners", churn["false_negatives"])
        st.write(
            f"At the validation-selected {meta['threshold']:.0%} threshold, "
            f"{churn['false_positives']} customers were flagged as churners but purchased, and "
            f"{churn['false_negatives']} were predicted to purchase but did not. "
            "The threshold was selected for F1 with a minimum recall requirement; catching more churners can increase false alerts."
        )
        st.caption(
            "MAE/MSE above describe regression. Churn classification uses the confusion matrix, error rate, precision, recall, F1 and ROC-AUC."
        )


def render_processed_data() -> None:
    """Make the existing feature table discoverable and easy to open in Excel."""
    st.title("Your feature-engineered dataset.")
    st.write("Explore the prepared customer snapshot and download a readable CSV.")
    try:
        info = fetch("/datasets/processed")
        a, b, c = st.columns(3)
        a.metric("Customer rows", f"{info['rows']:,}")
        b.metric("Input features", info["feature_count"])
        c.metric("Feature cutoff", info["cutoff"])
        st.code(info["source"])
        st.caption(
            "This file is relative to the project folder. Parquet is a binary data format; use pandas to read it or download CSV below."
        )
        choice = st.selectbox("Dataset to download", ["Features only", "Features + target labels", "Scored snapshot"])
        kind, name = {
            "Features only": ("features", "customer_features.csv"),
            "Features + target labels": ("modeling", "customers_with_targets.csv"),
            "Scored snapshot": ("scored", "scored_customers.csv"),
        }[choice]
        if kind != "features":
            st.warning(
                "Includes future target labels or model outputs. Do not use these columns as training predictors."
            )
        csv = fetch("/datasets/processed/download?kind=" + kind, binary=True)
        st.download_button("Download processed CSV", csv, name, "text/csv", type="primary")
        st.subheader("Feature preview")
        st.caption(
            "First 100 customers; downloads contain all customers. Customer ID is an identifier, not an input feature."
        )
        st.dataframe(pd.DataFrame(info["preview"]), hide_index=True, use_container_width=True)
        with st.expander("Inputs, targets and file locations"):
            st.write("**Input features:** " + ", ".join(info["feature_columns"]))
            st.write("**Future target columns:** " + ", ".join(info["target_columns"]))
            st.markdown(
                "- `data/processed/customer_features.csv`: ID and inputs only.\n"
                "- `data/processed/customers_with_targets.csv`: inputs and future labels.\n"
                "- `data/processed/scored_customers.parquet`: inputs, labels, predictions and split membership.\n"
                "- `data/processed/sequences.npz`: purchase sequences for the LSTM.\n"
                "- `data/interim/transactions.parquet`: cleaned transaction ledger."
            )
            st.code(
                'import pandas as pd\ndf = pd.read_parquet("data/processed/customers.parquet")\nprint(df.head())',
                language="python",
            )
    except requests.RequestException:
        st.error("Processed data is unavailable. Update and restart FastAPI, then refresh this page.")
