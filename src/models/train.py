"""Train all models, select on validation and evaluate the frozen test partition."""

import hashlib
import importlib.metadata
import json
import platform
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.features.build import FEATURES, NUMERIC
from src.models.deep import ANN, Autoencoder, PurchaseLSTM, infer, seed_everything, train_network
from src.models.evaluate import choose_threshold, classification, regression
from src.segmentation.cluster import fit_segments
from src.utils.config import ROOT, artifacts, config, log


def preprocessing(scaled: bool = False) -> ColumnTransformer:
    """Build fold-local imputation, optional scaling and country encoding."""
    numeric = Pipeline(
        [("fill", SimpleImputer(strategy="median")), ("scale", StandardScaler() if scaled else "passthrough")]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, NUMERIC),
            ("country", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["country"]),
        ]
    )


def train() -> None:
    """Train, compare and serialize all required classical and neural models."""
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    cfg = config()
    seed_everything(cfg["seed"])
    out = artifacts()
    out.mkdir(parents=True, exist_ok=True)
    data = pd.read_parquet(ROOT / "data/processed/customers.parquet")
    x = data[FEATURES]
    y = data.churn.to_numpy()
    indices = np.arange(len(data))
    train_i, rest = train_test_split(
        indices, test_size=cfg["test_fraction"] + cfg["validation_fraction"], stratify=y, random_state=cfg["seed"]
    )
    val_i, test_i = train_test_split(
        rest,
        test_size=cfg["test_fraction"] / (cfg["test_fraction"] + cfg["validation_fraction"]),
        stratify=y[rest],
        random_state=cfg["seed"],
    )
    split = {"train": train_i.tolist(), "validation": val_i.tolist(), "test": test_i.tolist()}
    (out / "split.json").write_text(json.dumps(split))
    folds = list(
        StratifiedKFold(n_splits=cfg["cv_folds"], shuffle=True, random_state=cfg["seed"]).split(
            x.iloc[train_i], y[train_i]
        )
    )
    m = cfg["models"]
    seed = cfg["seed"]
    candidates = {
        "LogisticRegression": (LogisticRegression(**m["logistic"]), True),
        "DecisionTree": (DecisionTreeClassifier(**m["tree"], random_state=seed), False),
        "RandomForest": (RandomForestClassifier(**m["forest"], random_state=seed), False),
        "XGBoost": (XGBClassifier(**m["xgb"], random_state=seed), False),
        "LightGBM": (LGBMClassifier(**m["lgb"], random_state=seed), False),
        "KNN": (KNeighborsClassifier(**m["knn"]), True),
    }
    models = {}
    runs = []
    for name, (estimator, scaled) in candidates.items():
        start = time.perf_counter()
        log("fit_model", model=name)
        pipe = Pipeline([("preprocess", preprocessing(scaled)), ("model", estimator)])
        if name in ["RandomForest", "XGBoost"]:
            search = GridSearchCV(
                pipe, cfg["rf_grid" if name == "RandomForest" else "xgb_grid"], cv=folds, scoring="roc_auc", n_jobs=1
            )
            search.fit(x.iloc[train_i], y[train_i])
            pipe = search.best_estimator_
            scores = [
                float(search.cv_results_[f"split{i}_test_score"][search.best_index_]) for i in range(cfg["cv_folds"])
            ]
            pd.DataFrame(search.cv_results_).to_csv(out / f"{name}_search.csv", index=False)
            if name == "XGBoost":
                tr = pipe["preprocess"].transform(x.iloc[train_i])
                va = pipe["preprocess"].transform(x.iloc[val_i])
                pipe["model"].set_params(early_stopping_rounds=25)
                pipe["model"].fit(tr, y[train_i], eval_set=[(va, y[val_i])], verbose=False)
        else:
            scores = cross_val_score(pipe, x.iloc[train_i], y[train_i], cv=folds, scoring="roc_auc", n_jobs=1).tolist()
            pipe.fit(x.iloc[train_i], y[train_i])
        vp = pipe.predict_proba(x.iloc[val_i])[:, 1]
        threshold = choose_threshold(y[val_i], vp, cfg["recall_target"])
        models[name] = pipe
        joblib.dump(pipe, out / f"{name}.joblib", compress=3)
        runs.append(
            {
                "name": name,
                "task": "churn",
                "cv_scores": scores,
                "cv_auc_mean": float(np.mean(scores)),
                "validation": classification(y[val_i], vp, threshold),
                "threshold": threshold,
                "seconds": time.perf_counter() - start,
                "parameters": str(pipe["model"].get_params()),
            }
        )
    prep = preprocessing(True).fit(x.iloc[train_i])
    z = prep.transform(x).astype("float32")
    joblib.dump(prep, out / "neural_preprocessing.joblib")
    start = time.perf_counter()
    ann_scores = []
    for fold, (a, b) in enumerate(folds):
        aa, bb = train_i[a], train_i[b]
        fit_i, stop_i = train_test_split(aa, test_size=0.15, stratify=y[aa], random_state=seed)
        pp = preprocessing(True).fit(x.iloc[fit_i])
        zz = pp.transform(x).astype("float32")
        seed_everything(seed + fold)
        network = ANN(zz.shape[1], cfg["dl"]["hidden"], cfg["dl"]["dropout"])
        train_network(network, zz[fit_i], y[fit_i], zz[stop_i], y[stop_i], cfg, "ann", epochs=cfg["dl"]["cv_epochs"])
        ann_scores.append(float(roc_auc_score(y[bb], infer(network, zz[bb], sigmoid=True))))
    seed_everything(seed)
    ann = ANN(z.shape[1], cfg["dl"]["hidden"], cfg["dl"]["dropout"])
    curves = {"ANN": train_network(ann, z[train_i], y[train_i], z[val_i], y[val_i], cfg, "ann")}
    vp = infer(ann, z[val_i], sigmoid=True)
    threshold = choose_threshold(y[val_i], vp, cfg["recall_target"])
    runs.append(
        {
            "name": "ANN",
            "task": "churn",
            "cv_scores": ann_scores,
            "cv_auc_mean": float(np.mean(ann_scores)),
            "validation": classification(y[val_i], vp, threshold),
            "threshold": threshold,
            "seconds": time.perf_counter() - start,
            "parameters": cfg["dl"],
        }
    )
    torch.save(ann.state_dict(), out / "ann.pt")
    champion = max([r for r in runs if r["name"] != "ANN"], key=lambda r: r["validation"]["roc_auc"])
    joblib.dump(models[champion["name"]], out / "churn.joblib", compress=3)
    log("champion_selected", name=champion["name"], validation=champion["validation"])
    clv_models = {}
    clv_runs = []
    for name, estimator in [
        ("Ridge", Ridge(**m["ridge"])),
        ("RandomForestRegressor", RandomForestRegressor(**m["forest_value"], random_state=seed)),
        ("LightGBMRegressor", LGBMRegressor(**m["lgb_value"], random_state=seed)),
    ]:
        start = time.perf_counter()
        pipe = Pipeline([("preprocess", preprocessing(name == "Ridge")), ("model", estimator)])
        scores = cross_val_score(pipe, x.iloc[train_i], data.clv_90d.iloc[train_i], cv=folds, scoring="r2").tolist()
        pipe.fit(x.iloc[train_i], data.clv_90d.iloc[train_i])
        clv_models[name] = pipe
        clv_runs.append(
            {
                "name": name,
                "cv_r2": scores,
                "validation": regression(data.clv_90d.iloc[val_i], np.maximum(0, pipe.predict(x.iloc[val_i]))),
                "seconds": time.perf_counter() - start,
                "parameters": str(estimator.get_params()),
            }
        )
        joblib.dump(pipe, out / f"{name}.joblib", compress=3)
    clv_best = max(clv_runs, key=lambda r: r["validation"]["r2"])["name"]
    joblib.dump(clv_models[clv_best], out / "clv.joblib", compress=3)
    start = time.perf_counter()
    seq = np.load(ROOT / "data/processed/sequences.npz")
    sx = seq["values"]
    sl = seq["lengths"]
    timing = data.next_days.to_numpy() / cfg["horizon_days"]
    seq_scores = []
    for fold, (a, b) in enumerate(folds):
        aa, bb = train_i[a], train_i[b]
        fit_i, stop_i = train_test_split(aa, test_size=0.15, stratify=y[aa], random_state=seed)
        seed_everything(seed + fold)
        net = PurchaseLSTM()
        train_network(
            net,
            sx[fit_i],
            timing[fit_i],
            sx[stop_i],
            timing[stop_i],
            cfg,
            "lstm",
            sl[fit_i],
            sl[stop_i],
            cfg["dl"]["cv_epochs"],
        )
        seq_scores.append(regression(data.next_days.iloc[bb], infer(net, sx[bb], sl[bb]) * cfg["horizon_days"]))
    seed_everything(seed)
    lstm = PurchaseLSTM()
    curves["LSTM"] = train_network(
        lstm, sx[train_i], timing[train_i], sx[val_i], timing[val_i], cfg, "lstm", sl[train_i], sl[val_i]
    )
    torch.save(lstm.state_dict(), out / "lstm.pt")
    sequence_seconds = time.perf_counter() - start
    timing_baseline = Pipeline([("preprocess", preprocessing(True)), ("model", Ridge(**m["ridge"]))]).fit(
        x.iloc[train_i], data.next_days.iloc[train_i]
    )
    joblib.dump(timing_baseline, out / "timing_baseline.joblib")
    baseline_cv = cross_val_score(
        timing_baseline, x.iloc[train_i], data.next_days.iloc[train_i], cv=folds, scoring="neg_mean_absolute_error"
    ).tolist()
    start = time.perf_counter()
    ae = Autoencoder(z.shape[1])
    curves["Autoencoder"] = train_network(ae, z[train_i], z[train_i], z[val_i], z[val_i], cfg, "autoencoder")
    error = np.mean((infer(ae, z) - z) ** 2, axis=1)
    anomaly_threshold = float(np.quantile(error[val_i], cfg["anomaly_quantile"]))
    torch.save(ae.state_dict(), out / "autoencoder.pt")
    ae_seconds = time.perf_counter() - start
    start = time.perf_counter()
    clusters, segments, cluster_metrics = fit_segments(x.iloc[train_i], data, cfg)
    joblib.dump(clusters, out / "segments.joblib")
    cluster_seconds = time.perf_counter() - start
    start = time.perf_counter()
    cat = Pipeline(
        [("preprocess", preprocessing()), ("model", RandomForestClassifier(**m["category"], random_state=seed))]
    )
    cat_cv = cross_val_score(
        cat, x.iloc[train_i], data.next_category.iloc[train_i], cv=folds, scoring="f1_macro"
    ).tolist()
    cat.fit(x.iloc[train_i], data.next_category.iloc[train_i])
    joblib.dump(cat, out / "category.joblib", compress=3)
    category_seconds = time.perf_counter() - start
    # Only now evaluate the held-out test partition, with models and thresholds frozen.
    for run in runs:
        p = (
            infer(ann, z[test_i], sigmoid=True)
            if run["name"] == "ANN"
            else models[run["name"]].predict_proba(x.iloc[test_i])[:, 1]
        )
        run["test"] = classification(y[test_i], p, run["threshold"])
    for run in clv_runs:
        run["test"] = regression(
            data.clv_90d.iloc[test_i], np.maximum(0, clv_models[run["name"]].predict(x.iloc[test_i]))
        )
    lstm_metrics = {
        "cv": seq_scores,
        "seconds": sequence_seconds,
        "test": regression(data.next_days.iloc[test_i], infer(lstm, sx[test_i], sl[test_i]) * cfg["horizon_days"]),
        "baseline_cv_mae": [-s for s in baseline_cv],
        "baseline_test": regression(
            data.next_days.iloc[test_i], np.clip(timing_baseline.predict(x.iloc[test_i]), 0, cfg["horizon_days"])
        ),
    }
    category_metrics = {
        "cv_macro_f1": cat_cv,
        "seconds": category_seconds,
        "parameters": str(cat["model"].get_params()),
        "test_accuracy": float(accuracy_score(data.next_category.iloc[test_i], cat.predict(x.iloc[test_i]))),
        "test_macro_f1": float(f1_score(data.next_category.iloc[test_i], cat.predict(x.iloc[test_i]), average="macro")),
    }
    metadata = {
        "python": platform.python_version(),
        "seed": seed,
        "training_configuration": cfg,
        "package_versions": {
            name: importlib.metadata.version(name)
            for name in ["numpy", "pandas", "scikit-learn", "torch", "xgboost", "lightgbm"]
        },
        "feature_table_sha256": hashlib.sha256((ROOT / "data/processed/customers.parquet").read_bytes()).hexdigest(),
        "cutoff": cfg["cutoff"],
        "horizon_days": cfg["horizon_days"],
        "customers": len(data),
        "split_sizes": {k: len(v) for k, v in split.items()},
        "churn_rate": float(y.mean()),
        "champion": champion["name"],
        "threshold": champion["threshold"],
        "churn_runs": runs,
        "clv_champion": clv_best,
        "clv_runs": clv_runs,
        "lstm": lstm_metrics,
        "category": category_metrics,
        "clustering": cluster_metrics,
        "clustering_training_seconds": cluster_seconds,
        "autoencoder_training_seconds": ae_seconds,
        "anomaly_threshold": anomaly_threshold,
        "anomaly_test_flag_rate": float((error[test_i] > anomaly_threshold).mean()),
        "feature_columns": FEATURES,
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (out / "learning_curves.json").write_text(json.dumps(curves, indent=2))
    scored = data.copy()
    scored["churn_probability"] = models[champion["name"]].predict_proba(x)[:, 1]
    scored["predicted_clv_90d"] = np.maximum(0, clv_models[clv_best].predict(x))
    scored["purchase_probability"] = 1 - scored.churn_probability
    scored["predicted_next_days"] = infer(lstm, sx, sl) * cfg["horizon_days"]
    scored["predicted_category"] = cat.predict(x)
    scored["anomaly_score"] = error
    scored["anomaly"] = error > anomaly_threshold
    scored["segment"] = segments.segment
    scored["cluster"] = segments.cluster
    scored["value_tier"] = np.where(
        scored.monetary >= x.iloc[train_i].monetary.quantile(0.8),
        "High",
        np.where(scored.monetary >= x.iloc[train_i].monetary.quantile(0.5), "Medium", "Low"),
    )
    scored["partition"] = "train"
    scored.loc[val_i, "partition"] = "validation"
    scored.loc[test_i, "partition"] = "test"
    scored["priority"] = scored.churn_probability * scored.predicted_clv_90d
    scored.to_parquet(ROOT / "data/processed/scored_customers.parquet", index=False)
    data.iloc[train_i[:100]][FEATURES].to_parquet(out / "explanation_background.parquet", index=False)
    pd.DataFrame(
        [
            {
                "model": r["name"],
                "cv_auc": r["cv_auc_mean"],
                "validation_auc": r["validation"]["roc_auc"],
                **{k: v for k, v in r["test"].items() if k != "confusion_matrix"},
            }
            for r in runs
        ]
    ).to_csv(ROOT / "docs/model_comparison.csv", index=False)
    log("training_complete", champion=champion["name"], test=champion["test"], clv=clv_runs, lstm=lstm_metrics)


if __name__ == "__main__":
    train()
