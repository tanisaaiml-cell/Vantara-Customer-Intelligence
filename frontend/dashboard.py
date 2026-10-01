"""Streamlit business UI. Data and inference are obtained only through FastAPI."""

import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

from analytics_views import render_error_analysis, render_processed_data

st.set_page_config(page_title="Vantara | Customer intelligence", page_icon="◈", layout="wide")
API = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
COLORS = ["#6258e8", "#18a999", "#f3b448", "#ef7480", "#6698ed", "#b984df"]
st.markdown(
    """<style>
.stApp{background:#f5f6fb;color:#18213a}.block-container{padding-top:2.2rem;max-width:1450px}
[data-testid="stSidebar"]{background:#111a32}[data-testid="stSidebar"] *{color:#e3e8fa}
[data-testid="stSidebar"] button *{color:#18213a!important}
[data-testid="stMetric"]{background:white;border:1px solid #e6e8f3;border-radius:16px;padding:22px 24px;box-shadow:0 6px 24px #19254b05}
[data-testid="stMetricLabel"]{font-size:13px;color:#68728c}[data-testid="stMetricValue"]{font-size:29px;font-weight:700;color:#19233e}
h1{font-size:2.2rem!important;letter-spacing:-1px}h2{font-size:1.3rem!important}h3{font-size:1.1rem!important}
.hero{background:linear-gradient(112deg,#18213f,#332860);border-radius:22px;padding:30px 34px;color:white;margin-bottom:25px}
.hero h1{color:white;margin:8px 0}.hero p{color:#c8cce8;margin-bottom:0;max-width:800px;font-size:15px}.eyebrow{font-size:11px;letter-spacing:2px;font-weight:700;color:#9ce5d7}
.brand{font-size:26px;font-weight:700;letter-spacing:3px;padding:10px 0 3px}.brand-sub{font-size:11px;letter-spacing:1.8px;color:#9aa7c6!important;margin-bottom:32px}
.stButton>button{border-radius:10px}
</style>""",
    unsafe_allow_html=True,
)


@st.cache_data(ttl=120, show_spinner=False)
def get(path: str) -> object:
    """Read cached JSON over HTTP; never access local model files."""
    response = requests.get(API + path, timeout=60)
    response.raise_for_status()
    return response.json()


def chart_style(fig: go.Figure, height: int = 350) -> go.Figure:
    """Apply a consistent readable chart theme."""
    fig.update_layout(
        height=height,
        margin=dict(l=15, r=15, t=30, b=25),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Arial", color="#59647e"),
        colorway=COLORS,
        legend_title_text="",
    )
    fig.update_xaxes(gridcolor="#f0f2f8")
    fig.update_yaxes(gridcolor="#f0f2f8")
    return fig


def hero(title: str, subtitle: str) -> None:
    """Show a page title and practical description."""
    st.markdown(
        f'<div class="hero"><div class="eyebrow">CUSTOMER INTELLIGENCE / VANTARA</div><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


with st.sidebar:
    st.markdown(
        '<div class="brand">◈ VANTARA</div><div class="brand-sub">RETAIL INTELLIGENCE</div>', unsafe_allow_html=True
    )
    page = st.radio(
        "WORKSPACE",
        ["Overview", "Customer segments", "Customer explorer", "Model performance", "Processed data", "Batch scoring"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("HISTORICAL SNAPSHOT")
    st.markdown("**01 September 2011**")
    st.caption("Predictions cover the following 90 days. Values are in GBP.")
    if st.button("Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.caption("Built for informed retention decisions.")
try:
    meta = get("/metadata")
    customers = pd.DataFrame(get("/customers"))
except (requests.RequestException, ValueError):
    st.error("Cannot reach the analytics API. Start FastAPI on port 8000, then refresh.")
    st.code("python -m uvicorn api.main:app --port 8000")
    st.stop()

if page == "Overview":
    hero(
        "A clearer view of your customers.",
        "Understand purchase patterns, prioritize retention, and explore the signals behind every prediction.",
    )
    a, b, c, d = st.columns(4)
    a.metric("Customers in snapshot", f"{len(customers):,}")
    b.metric("High churn risk", f"{(customers.churn_probability >= meta['threshold']).sum():,}")
    c.metric("Expected 90-day spend", f"£{customers.predicted_clv_90d.sum() / 1e6:.2f}M")
    d.metric("Customer segments", customers.segment.nunique())
    st.write("")
    left, right = st.columns([1.7, 1])
    with left:
        st.subheader("Revenue over time")
        trend = get("/trends")
        actual = pd.DataFrame(trend["actual"])
        forecast = pd.DataFrame(trend["forecast"])
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=actual.month,
                y=actual.revenue,
                name="Observed net revenue",
                line=dict(color=COLORS[0], width=3),
                fill="tozeroy",
                fillcolor="rgba(98,88,232,.07)",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[actual.month.iloc[-1], *forecast.month],
                y=[actual.revenue.iloc[-1], *forecast.revenue],
                name="Baseline forecast",
                line=dict(color=COLORS[1], width=3, dash="dot"),
            )
        )
        st.plotly_chart(chart_style(fig), use_container_width=True)
        st.caption(trend["method"])
    with right:
        st.subheader("Churn risk distribution")
        fig = px.histogram(customers, x="churn_probability", nbins=25, color_discrete_sequence=[COLORS[0]])
        fig.update_xaxes(title="No-purchase probability", tickformat=".0%")
        fig.update_yaxes(title="Customers")
        st.plotly_chart(chart_style(fig), use_container_width=True)
        st.caption(f"Validation-selected threshold: {meta['threshold']:.0%}.")
    st.subheader("Retention priorities")
    st.caption("Ranked by churn probability × expected 90-day spend.")
    st.dataframe(
        customers[["customer_id", "country", "segment", "churn_probability", "predicted_clv_90d", "priority"]].head(12),
        hide_index=True,
        use_container_width=True,
        column_config={
            "churn_probability": st.column_config.ProgressColumn("Churn risk", min_value=0, max_value=1, format="%.2f"),
            "predicted_clv_90d": st.column_config.NumberColumn("90-day spend", format="£%.0f"),
        },
    )
elif page == "Customer segments":
    hero(
        "Different customers. Better decisions.",
        "Explore behavior-based groups and focus on the customers who matter to your next campaign.",
    )
    a, b, c = st.columns(3)
    country = a.multiselect("Country", sorted(customers.country.unique()))
    segment = b.multiselect("Segment", sorted(customers.segment.unique()))
    tier = c.multiselect("Value tier", ["High", "Medium", "Low"])
    selected = customers.copy()
    for key, values in [("country", country), ("segment", segment), ("value_tier", tier)]:
        if values:
            selected = selected[selected[key].isin(values)]
    if selected.empty:
        st.info("No customers match these filters.")
        st.stop()
    a, b, c = st.columns(3)
    a.metric("Selected customers", f"{len(selected):,}")
    b.metric("Average churn risk", f"{selected.churn_probability.mean():.1%}")
    c.metric("Historical gross spend", f"£{selected.monetary.sum():,.0f}")
    left, right = st.columns([1.5, 1])
    with left:
        st.subheader("Behavior map")
        fig = px.scatter(
            selected,
            x="recency",
            y="monetary",
            color="segment",
            size="frequency",
            size_max=25,
            hover_data=["customer_id"],
            log_y=True,
            color_discrete_sequence=COLORS,
            labels={"recency": "Days since purchase", "monetary": "Historical gross spend (£)"},
        )
        fig = chart_style(fig, 430)
        fig.update_layout(legend=dict(orientation="h", y=-0.3), margin=dict(b=90))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.subheader("Segment mix")
        counts = selected.segment.value_counts().rename_axis("segment").reset_index(name="customers")
        fig = chart_style(
            px.pie(counts, names="segment", values="customers", hole=0.7, color_discrete_sequence=COLORS), 430
        )
        fig.update_layout(legend=dict(orientation="h", y=-0.2), margin=dict(b=90))
        st.plotly_chart(fig, use_container_width=True)
    st.subheader("Segment profiles")
    st.dataframe(
        selected.groupby("segment")
        .agg(
            customers=("customer_id", "size"),
            median_recency=("recency", "median"),
            median_orders=("frequency", "median"),
            median_spend=("monetary", "median"),
            average_risk=("churn_probability", "mean"),
        )
        .round(2),
        use_container_width=True,
    )
    st.download_button(
        "Download filtered customers", selected.to_csv(index=False).encode(), "customer_segments.csv", "text/csv"
    )
elif page == "Customer explorer":
    hero(
        "Every score has a story.", "Inspect purchase behavior and the features that influence a customer’s churn risk."
    )
    cid = st.selectbox("Search customer ID", customers.customer_id.astype(str).sort_values())
    row = get("/customers/" + cid)
    a, b, c, d = st.columns(4)
    a.metric("Churn risk", f"{row['churn_probability']:.1%}")
    b.metric("Expected 90-day spend", f"£{row['predicted_clv_90d']:,.0f}")
    c.metric("Purchase probability", f"{row['purchase_probability']:.1%}")
    d.metric("Capped wait estimate", f"{row['predicted_next_days']:.0f} days")
    st.caption(
        f"{row['country']} · {row['segment']} · {row['frequency']:.0f} orders · {row['recency']:.0f} days since purchase"
    )
    if row["anomaly"]:
        st.warning("Unusual spending pattern: review manually. This is not a verified fraud label.")
    with st.spinner("Calculating explanations…"):
        explanation = get("/customers/" + cid + "/explain")
    st.info(explanation["explanation"])
    left, right = st.columns(2)
    for col, key, title in [(left, "shap", "SHAP contribution"), (right, "lime", "LIME local approximation")]:
        with col:
            st.subheader(title)
            values = pd.DataFrame(explanation[key]).sort_values("contribution")
            values["feature"] = values.feature.str.replace("numeric__", "", regex=False).str.replace(
                "country__", "", regex=False
            )
            values["direction"] = values.contribution.apply(lambda x: "Higher risk" if x > 0 else "Lower risk")
            fig = px.bar(
                values,
                y="feature",
                x="contribution",
                orientation="h",
                color="direction",
                color_discrete_map={"Higher risk": COLORS[0], "Lower risk": COLORS[1]},
            )
            fig = chart_style(fig, 420)
            fig.update_yaxes(title=None, tickfont=dict(size=11))
            fig.update_xaxes(tickangle=0)
            fig.update_layout(margin=dict(l=10, r=10, b=75, t=15), legend=dict(orientation="h", y=-0.22))
            st.plotly_chart(fig, use_container_width=True)
    st.caption(explanation["units"] + f" LIME local fit R²: {explanation['lime_local_fit']:.3f}.")
    recommendations = get("/recommendations/" + cid)
    st.subheader("Suggested products")
    st.caption(recommendations["method"])
    st.dataframe(pd.DataFrame(recommendations["items"]), hide_index=True, use_container_width=True)
    report = requests.get(API + "/report/" + cid, timeout=30)
    if report.ok:
        st.download_button("Download customer PDF", report.content, f"customer_{cid}.pdf", "application/pdf")
elif page == "Model performance":
    hero(
        "Evidence behind the predictions.",
        "Compare validation and held-out results and inspect the model’s strongest signals.",
    )
    runs = meta["churn_runs"]
    champion = next(r for r in runs if r["name"] == meta["champion"])
    clv = next(r for r in meta["clv_runs"] if r["name"] == meta["clv_champion"])
    a, b, c, d = st.columns(4)
    a.metric("Test ROC-AUC", f"{champion['test']['roc_auc']:.3f}")
    b.metric("Test churn recall", f"{champion['test']['recall']:.1%}")
    c.metric("Test CLV R²", f"{clv['test']['r2']:.3f}")
    d.metric("Held-out customers", meta["split_sizes"]["test"])
    if clv["test"]["r2"] < 0.6:
        st.warning(
            "The selected CLV model misses the R² ≥ 0.60 target. LSTM timing also underperforms the static Ridge baseline. See the report."
        )
    st.caption(
        f"Churn: {meta['champion']} · Value: {meta['clv_champion']}. Selection uses validation; test results are for final reporting."
    )
    table = pd.DataFrame(
        [
            {
                "Model": r["name"],
                "CV AUC": r["cv_auc_mean"],
                "Validation AUC": r["validation"]["roc_auc"],
                **{k: v for k, v in r["test"].items() if k != "confusion_matrix"},
            }
            for r in runs
        ]
    ).sort_values("roc_auc", ascending=False)
    st.dataframe(
        table.style.format({c: "{:.3f}" for c in table.columns if c != "Model"}),
        hide_index=True,
        use_container_width=True,
    )
    left, right = st.columns([1.4, 1])
    with left:
        st.subheader("Global feature importance")
        importance = pd.DataFrame(get("/global-importance")).head(12).sort_values("importance")
        st.plotly_chart(
            chart_style(
                px.bar(importance, y="feature", x="importance", orientation="h", color_discrete_sequence=[COLORS[0]]),
                420,
            ),
            use_container_width=True,
        )
    with right:
        st.subheader("Held-out confusion matrix")
        st.plotly_chart(
            chart_style(
                px.imshow(
                    champion["test"]["confusion_matrix"],
                    x=["Predicted purchase", "Predicted churn"],
                    y=["Actual purchase", "Actual churn"],
                    text_auto=True,
                    color_continuous_scale="Purples",
                ),
                420,
            ),
            use_container_width=True,
        )
    st.caption(
        "One row per customer, 70/15/15 stratified split at a historical cutoff. This is not a forward-time deployment backtest."
    )
    render_error_analysis(meta)
elif page == "Processed data":
    render_processed_data()
else:
    hero(
        "Score a customer list in seconds.",
        "Upload snapshot customer IDs or a CSV containing the documented customer features.",
    )
    st.download_button(
        "Download ID template", customers[["customer_id"]].head(5).to_csv(index=False), "customer_ids.csv", "text/csv"
    )
    upload = st.file_uploader("Customer CSV", type=["csv"])
    st.caption("Up to 5,000 rows / 5 MB. ID-only files retrieve saved scores; feature files run trained models.")
    if upload is not None and st.button("Score customers", type="primary"):
        with st.spinner("Scoring through the API…"):
            try:
                response = requests.post(
                    API + "/predict/batch", files={"file": (upload.name, upload.getvalue(), "text/csv")}, timeout=120
                )
                if not response.ok:
                    st.error(response.json().get("detail", "Scoring failed"))
                else:
                    st.session_state["batch_result"] = response.json()["results"]
            except requests.RequestException as exc:
                st.error(str(exc))
    if "batch_result" in st.session_state:
        result = pd.DataFrame(st.session_state["batch_result"])
        st.success(f"{len(result)} customers scored and saved.")
        st.dataframe(result, hide_index=True, use_container_width=True)
        st.download_button("Download scores CSV", result.to_csv(index=False), "customer_predictions.csv", "text/csv")
