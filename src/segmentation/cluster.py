"""Training-only cluster selection and readable customer profiles."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

COLUMNS = ["recency", "frequency", "monetary", "return_rate", "seasonal_concentration"]


def fit_segments(
    train: pd.DataFrame, all_customers: pd.DataFrame, cfg: dict[str, Any]
) -> tuple[dict, pd.DataFrame, list]:
    """Choose K-Means k using silhouette; compare Gaussian mixtures at the same k."""
    scaler = StandardScaler().fit(np.log1p(train[COLUMNS]))
    x = scaler.transform(np.log1p(train[COLUMNS]))
    results = []
    models = {}
    for k in cfg["cluster_k"]:
        model = KMeans(n_clusters=k, n_init=20, random_state=cfg["seed"]).fit(x)
        models[k] = model
        results.append(
            {
                "model": "KMeans",
                "k": k,
                "inertia": float(model.inertia_),
                "silhouette": float(
                    silhouette_score(x, model.labels_, sample_size=min(2000, len(x)), random_state=cfg["seed"])
                ),
                "davies_bouldin": float(davies_bouldin_score(x, model.labels_)),
            }
        )
    best = max(results, key=lambda r: r["silhouette"])
    km = models[best["k"]]
    gm = GaussianMixture(n_components=best["k"], random_state=cfg["seed"], n_init=3).fit(x)
    labels = gm.predict(x)
    results.append(
        {
            "model": "GaussianMixture",
            "k": best["k"],
            "inertia": None,
            "silhouette": float(silhouette_score(x, labels, sample_size=min(2000, len(x)), random_state=cfg["seed"])),
            "davies_bouldin": float(davies_bouldin_score(x, labels)),
        }
    )
    profiles = train.assign(cluster=km.labels_).groupby("cluster")[COLUMNS].median()
    names = {}
    for cluster, row in profiles.iterrows():
        activity = "Lapsed" if row.recency > 90 else "Active"
        value = "high value" if row.monetary >= train.monetary.median() else "low value"
        names[int(cluster)] = f"{activity}, {value} · {cluster + 1}"
    output = all_customers[["customer_id", *COLUMNS]].copy()
    z = scaler.transform(np.log1p(all_customers[COLUMNS]))
    output["cluster"] = km.predict(z)
    output["segment"] = output.cluster.map(names)
    output["gmm_cluster"] = gm.predict(z)
    return {"scaler": scaler, "kmeans": km, "gmm": gm, "names": names}, output, results
