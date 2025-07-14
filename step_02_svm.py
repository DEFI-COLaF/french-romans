import argparse
import json
import os
import pickle
import gzip
from itertools import product
from typing import Dict, Any, List
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import tqdm
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    f1_score,
)
from scipy.stats import wilcoxon

from ruzicka.BDIVerifier import BDIVerifier

from tools.prepare import extract_all_authors_decade, QueryCandidatesImpostors
from tools.constants import rng
from tools import compress

# ───────────────────────────────────────────── global parameters ─────────
MIN_CANDIDATES = 1
MAX_NUMBER_OF_SAMPLES = 50
C_GRID = [0.01, 0.1, 1, 10, 100]
NB_PROCS = int(os.getenv("NBPROC", "4"))
TRANSFORM_SAMPLE = 100_000

FAST_ENV_KEY = "SVM_FAST"
SEARCH_ENV_KEY = "SVM_SEARCH"
ENSEMBLE_ENV_KEY = "SVM_ENSEMBLE"

# ───────────────────────────────────────────── helper --------------------

def _sample(df: pd.DataFrame, frac: float, *, replace: bool = True) -> pd.DataFrame:
    """Bootstrap‑sample *frac* of df (≥1 row)."""
    if df.empty:
        return df
    n = max(1, int(np.ceil(frac * len(df))))
    return df.sample(n=n, replace=replace, random_state=rng)

# ───────────────────────────────────────── baseline BDI worker -----------

def run_bdi(pickled_experiment: bytes) -> Dict[str, Any]:
    experiment: QueryCandidatesImpostors = pickle.loads(pickled_experiment)

    # train --------------------------------------------------------------
    X_train = pd.concat([
        experiment.candidate[experiment.features],
        experiment.impostors[experiment.features],
    ])
    y_raw = pd.concat([
        experiment.candidate["var_author"],
        experiment.impostors["var_author"],
    ])
    y, _ = y_raw.factorize()
    scaler = StandardScaler(with_mean=False).fit(X_train)
    verifier = BDIVerifier(metric="minmax", nb_bootstrap_iter=1000, rnd_prop=0.35, random_state=rng)
    verifier.fit(scaler.transform(X_train), y)

    # query & impostor predictions --------------------------------------
    q_scaled = scaler.transform(experiment.query[experiment.features])
    imp_df = _sample(experiment.impostors[experiment.features], len(q_scaled))
    imp_scaled = scaler.transform(imp_df)

    proba_q = verifier.predict_proba(q_scaled, [y[0]] * len(q_scaled))
    proba_imp = verifier.predict_proba(imp_scaled, [y[0]] * len(imp_scaled)) if len(imp_scaled) else np.array([])

    # metrics ------------------------------------------------------------
    if len(proba_imp):
        y_true = np.concatenate([np.ones(len(proba_q)), np.zeros(len(proba_imp))])
        y_scores = np.concatenate([proba_q, proba_imp])
        roc_auc = float(roc_auc_score(y_true, y_scores))
        avg_prec = float(average_precision_score(y_true, y_scores))
        acc = float(accuracy_score(y_true, y_scores >= 0.5))
        f1 = float(f1_score(y_true, y_scores >= 0.5))
    else:
        roc_auc = avg_prec = acc = f1 = np.nan

    labels = [f"{fn}#{w}" for w, fn in zip(experiment.query["var_window"], experiment.query.index)]
    return {
        "arrays": np.round(verifier._dist_arrays, 4).tolist(),
        "date": int(experiment.year),
        "probas": np.round(proba_q, 3).tolist(),
        "labels": labels,
        "gap": experiment.gap,
        "author": experiment.author,
        "metrics": {
            "roc_auc": None if np.isnan(roc_auc) else round(roc_auc, 3),
            "avg_prec": None if np.isnan(avg_prec) else round(avg_prec, 3),
            "accuracy": None if np.isnan(acc) else round(acc, 3),
            "f1": None if np.isnan(f1) else round(f1, 3),
        },
    }

# ───────────────────────────────────────── SVM worker (ensemble‑aware) ---

def run_svm(pickled_payload: bytes) -> Dict[str, Any]:
    fast_mode = os.getenv(FAST_ENV_KEY) == "1"
    search_mode = os.getenv(SEARCH_ENV_KEY, "grid")
    ensemble_n = int(os.getenv(ENSEMBLE_ENV_KEY, "1"))

    experiment, transformer_blob = pickle.loads(pickled_payload)
    transformer: Pipeline = pickle.loads(transformer_blob)

    # query + impostor constant across replicas -------------------------
    q_df = _sample(experiment.query[experiment.features], 1.0, replace=False)
    q_t = transformer.transform(q_df)

    imp_df = _sample(experiment.impostors[experiment.features], len(q_df))
    imp_t = transformer.transform(imp_df)

    proba_q_sum = np.zeros(len(q_df))
    proba_imp_sum = np.zeros(len(imp_df)) if len(imp_df) else np.array([])
    margin_sum = np.zeros(len(q_df))

    # bagging loop -------------------------------------------------------
    for rep in range(ensemble_n):
        cand_boot = _sample(experiment.candidate, 0.7)
        imp_boot = _sample(experiment.impostors, 0.7)

        X_train = pd.concat([
            cand_boot[experiment.features],
            imp_boot[experiment.features],
        ])
        y_raw = pd.concat([
            cand_boot["var_author"],
            imp_boot["var_author"],
        ])
        y = (y_raw == experiment.author).astype(int).values
        X_train_t = transformer.transform(X_train)

        if fast_mode:
            base = SGDClassifier(loss="hinge", alpha=1e-4, class_weight="balanced", random_state=rep)
            clf = CalibratedClassifierCV(base, method="sigmoid", cv=3).fit(X_train_t, y)
        else:
            svc = SVC(kernel="linear", class_weight="balanced", probability=True, random_state=rep)
            pipe = Pipeline([("svc", svc)])
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=rep)
            search_cls = HalvingGridSearchCV if search_mode == "halving" else GridSearchCV
            search = search_cls(pipe, {"svc__C": C_GRID}, scoring="roc_auc", cv=cv, n_jobs=1, refit=True)
            search.fit(X_train_t, y)
            clf = search.best_estimator_

        proba_q_sum += clf.predict_proba(q_t)[:, 1]
        margin_sum += clf.decision_function(q_t)
        if len(imp_df):
            proba_imp_sum += clf.predict_proba(imp_t)[:, 1]

    proba_q_avg = proba_q_sum / ensemble_n
    margin_avg = margin_sum / ensemble_n
    proba_imp_avg = proba_imp_sum / ensemble_n if len(imp_df) else np.array([])

    # metrics ------------------------------------------------------------
    if len(imp_df):
        y_true = np.concatenate([np.ones(len(q_df)), np.zeros(len(imp_df))])
        y_scores = np.concatenate([proba_q_avg, proba_imp_avg])
        roc_auc = float(roc_auc_score(y_true, y_scores))
        avg_prec = float(average_precision_score(y_true, y_scores))
        acc = float(accuracy_score(y_true, y_scores >= 0.5))
        f1 = float(f1_score(y_true, y_scores >= 0.5))
    else:
        roc_auc = avg_prec = acc = f1 = np.nan

    labels = [f"{fn}#{w}" for w, fn in zip(experiment.query["var_window"], experiment.query.index)]

    return {
        "margin": np.round(margin_avg, 3).tolist(),
        "probas": np.round(proba_q_avg, 3).tolist(),
        "ensemble_n": ensemble_n,
        "metrics": {
            "roc_auc": None if np.isnan(roc_auc) else round(roc_auc, 3),
            "avg_prec": None if np.isnan(avg_prec) else round(avg_prec, 3),
            "accuracy": None if np.isnan(acc) else round(acc, 3),
            "f1": None if np.isnan(f1) else round(f1, 3),
        },
        "labels": labels,
        "date": int(experiment.year),
        "gap": experiment.gap,
        "author": experiment.author,
    }

# ───────────────────────────────────────────── launcher ------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BDI vs SVM benchmark (ensemble edition)")
    parser.add_argument("--engine", choices=["bdi", "svm"], default="svm")
    parser.add_argument("--fast", action="store_true", help="Use SGD + calibration (skip search)")
    parser.add_argument("--search", choices=["grid", "halving"], default="grid", help="Hyper‑parameter search type")
    parser.add_argument("--ensemble", type=int, default=1, help="Number of bagging replicas (default 1)")
    args = parser.parse
