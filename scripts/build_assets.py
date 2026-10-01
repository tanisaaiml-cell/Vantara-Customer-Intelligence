"""Produce EDA, explanations, catalog assets and exact architecture diagrams."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from matplotlib.patches import FancyBboxPatch
from sklearn.inspection import PartialDependenceDisplay
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src.features.build import FEATURES, NUMERIC, history
from src.models.service import ModelService
from src.utils.config import ROOT, artifacts, config, log


def diagram(filename: str, title: str, nodes: list, edges: list) -> None:
    """Draw readable boxes with arrows connected at their edges."""
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor("#f5f6fb")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.text(0.4, 6.55, title, fontsize=21, fontweight="bold", color="#18213a")
    for x, y, label in nodes:
        ax.add_patch(
            FancyBboxPatch(
                (x, y), 3.1, 1.05, boxstyle="round,pad=.12", linewidth=1, edgecolor="#d5d8e9", facecolor="white"
            )
        )
        ax.text(x + 1.55, y + 0.52, label, ha="center", va="center", fontsize=11, color="#18213a")
    for a, b in edges:
        x, y, _ = nodes[a]
        xx, yy, _ = nodes[b]
        if abs(y - yy) < 0.01:
            right = xx > x
            start = (x + 3.22 if right else x - 0.12, y + 0.525)
            end = (xx - 0.12 if right else xx + 3.22, yy + 0.525)
        else:
            down = yy < y
            start = (x + 1.55, y - 0.12 if down else y + 1.17)
            end = (xx + 1.55, yy + 1.17 if down else yy - 0.12)
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", color="#6258e8", lw=2))
        if filename == "er_diagram.png":
            ax.text(
                (start[0] + end[0]) / 2,
                (start[1] + end[1]) / 2 + 0.16,
                "1 : 1" if (a, b) == (0, 1) else "1 : N",
                fontsize=10,
                ha="center",
                color="#6258e8",
            )
    fig.savefig(ROOT / "docs" / filename, dpi=160, bbox_inches="tight")
    plt.close(fig)


def build() -> None:
    """Use only observable catalog data and the saved trained artifacts."""
    cfg = config()
    out = artifacts()
    docs = ROOT / "docs"
    model = ModelService()
    data = pd.read_parquet(ROOT / "data/processed/scored_customers.parquet")
    transactions = pd.read_parquet(ROOT / "data/interim/transactions.parquet")
    past = history(transactions, cfg["cutoff"])
    products = past[~past.is_return & past.is_product]
    catalog = (
        products.groupby("stock_code")
        .agg(
            description=("description", "first"),
            category=("category", "first"),
            orders=("invoice", "nunique"),
            typical_price=("price", "median"),
        )
        .reset_index()
        .sort_values("orders", ascending=False)
    )
    catalog.to_parquet(out / "recommendations.parquet", index=False)
    (out / "purchased_products.json").write_text(
        json.dumps(products.groupby("customer_id").stock_code.agg(lambda s: sorted(set(s))).to_dict())
    )
    sample = data.sample(min(160, len(data)), random_state=cfg["seed"])
    values, z = model.explainer.values(sample)
    names = [n.replace("numeric__", "").replace("country__", "") for n in model.explainer.names]
    importance = pd.DataFrame({"feature": names, "importance": np.abs(values).mean(axis=0)}).sort_values(
        "importance", ascending=False
    )
    (out / "global_importance.json").write_text(importance.to_json(orient="records"))
    shap.summary_plot(values, z, feature_names=names, show=False, max_display=14)
    plt.tight_layout()
    plt.savefig(docs / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    order = data.churn_probability.sort_values().index
    representatives = [order[0], (data.churn_probability - model.metadata["threshold"]).abs().idxmin(), order[-1]]
    for label, index in zip(["low", "borderline", "high"], representatives):
        row = data.loc[[index]]
        v, z = model.explainer.values(row)
        result = model.explainer.individual(row)
        (docs / f"explanation_{label}.json").write_text(
            json.dumps({"customer_id": str(row.customer_id.iloc[0]), **result}, indent=2)
        )
        base = np.asarray(model.explainer.shap.expected_value)
        base = float(base[-1]) if base.ndim else float(base)
        shap.save_html(
            str(docs / f"shap_force_{label}.html"),
            shap.force_plot(base, v[0], z[0], feature_names=model.explainer.names, matplotlib=False),
        )
        shap.plots.waterfall(
            shap.Explanation(values=v[0], base_values=base, data=z[0], feature_names=model.explainer.names),
            show=False,
            max_display=10,
        )
        plt.savefig(docs / f"shap_{label}.png", dpi=140, bbox_inches="tight")
        plt.close()
    PartialDependenceDisplay.from_estimator(
        model.churn, data[FEATURES].sample(150, random_state=42), ["recency", "frequency"], grid_resolution=20
    )
    plt.tight_layout()
    plt.savefig(docs / "partial_dependence.png", dpi=150)
    plt.close()
    data[NUMERIC + ["churn", "clv_90d"]].describe().to_csv(docs / "feature_summary.csv")
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(data[NUMERIC + ["churn", "clv_90d"]].corr(), ax=ax, cmap="coolwarm", center=0)
    fig.tight_layout()
    fig.savefig(docs / "correlations.png", dpi=130)
    plt.close(fig)
    train = data.iloc[json.loads((out / "split.json").read_text())["train"]]
    scaled = (train[NUMERIC] - train[NUMERIC].mean()) / train[NUMERIC].std().replace(0, 1)
    matrix = np.column_stack([np.ones(len(scaled)), scaled])
    pd.DataFrame(
        [{"feature": col, "vif": float(variance_inflation_factor(matrix, i + 1))} for i, col in enumerate(NUMERIC)]
    ).to_csv(docs / "vif.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, col in zip(axes, ["recency", "frequency", "monetary"]):
        ax.hist(data[col] if col == "recency" else np.log1p(data[col]), bins=35, color="#6258e8")
        ax.set_title(col if col == "recency" else "log(1 + " + col + ")")
        ax.set_ylabel("Customers")
    fig.tight_layout()
    fig.savefig(docs / "rfm_distributions.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    past.groupby("country").amount.sum().sort_values().tail(10).plot.barh(
        ax=axes[0], color="#6258e8", title="Top countries by net spend"
    )
    past.groupby(past.invoice_date.dt.to_period("M")).amount.sum().plot(
        ax=axes[1], color="#18a999", title="Monthly net revenue"
    )
    fig.tight_layout()
    fig.savefig(docs / "country_seasonality.png", dpi=150)
    plt.close(fig)
    curves = json.loads((out / "learning_curves.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7))
    for ax, (name, curve) in zip(axes, curves.items()):
        c = pd.DataFrame(curve)
        ax.plot(c.epoch, c.train_loss, label="Training")
        ax.plot(c.epoch, c.validation_loss, label="Validation")
        ax.set_title(name)
        ax.set_xlabel("Epoch")
        ax.legend()
    fig.tight_layout()
    fig.savefig(docs / "learning_curves.png", dpi=150)
    plt.close(fig)
    cm = pd.DataFrame(model.metadata["clustering"])
    km = cm[cm.model == "KMeans"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].plot(km.k, km.inertia, "o-", color="#6258e8")
    axes[0].set_title("K-Means elbow")
    axes[1].plot(km.k, km.silhouette, "o-", color="#18a999")
    axes[1].set_title("Silhouette by k")
    fig.tight_layout()
    fig.savefig(docs / "cluster_selection.png", dpi=150)
    plt.close(fig)
    diagram(
        "architecture_diagram.png",
        "Vantara customer intelligence architecture",
        [
            (0.5, 4.7, "Workbook\nBoth source sheets"),
            (4.45, 4.7, "Data pipeline\nValidation + features"),
            (8.4, 4.7, "Training + evaluation\nML / DL / segments"),
            (8.4, 2.8, "Batch scoring\nVersioned model artifacts"),
            (4.45, 2.8, "PostgreSQL\nCustomers + predictions"),
            (0.5, 2.8, "FastAPI\nScores + explanations"),
            (0.5, 0.85, "Streamlit\nBusiness dashboard"),
        ],
        [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)],
    )
    diagram(
        "er_diagram.png",
        "PostgreSQL entity relationships",
        [
            (0.7, 4.6, "customers\nPK customer_id; country; features"),
            (4.45, 4.6, "predictions\nPK/FK customer_id; FK segment_id"),
            (8.2, 4.6, "segments\nPK segment_id; name"),
            (0.7, 2.2, "transactions\nPK row_id; FK customer_id"),
            (4.45, 2.2, "batch_scores\nPK id; created_at; payload"),
        ],
        [(0, 1), (2, 1), (0, 3)],
    )
    diagram(
        "workflow_diagram.png",
        "Training and serving workflow",
        [
            (0.5, 4.7, "Prepare\nClean + point-in-time features"),
            (4.45, 4.7, "Split customers\n70% train / 15% val / 15% test"),
            (8.4, 4.7, "Training CV\nFold-local transformations"),
            (8.4, 2.8, "Validation\nModel + threshold selection"),
            (4.45, 2.8, "Frozen test evaluation\nMetrics + limitations"),
            (0.5, 2.8, "Saved artifacts\nScores + explanations"),
            (0.5, 0.85, "API + dashboard\nPersist and inspect scores"),
        ],
        [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)],
    )
    log("assets_complete")


if __name__ == "__main__":
    build()
