##############################
# step_02_svm.py
# ----------------------------------------------------------------------
# What this script does
# ---------------------
# • BDI baseline (unchanged logic, built on Thibault's version ; now with basic metrics)
# • Linear‑SVM verifier with options:
#     --fast      : SGD + Platt, no C‑search (quick sanity check)
#     --search    : grid | halving  for libSVM C tuning
#     --ensemble N: bag N bootstrap replicas, average scores
# • Strict train/test split for impostor windows → no leakage
# • One global log1p+scaler transformer fitted per (gap, direction)
# • Results in JSON.GZ with per‑task metrics + run summary + Wilcoxon vs BDI
##############################

# --- import block --------------------------------
import json
import os
import pickle
from itertools import product
from typing import Dict, Any, List
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import numpy as np
import tqdm
from sklearn.preprocessing import StandardScaler
from ruzicka.BDIVerifier import BDIVerifier

# --- imports for the SVM pipeline ------------------------------------
import argparse
import gzip
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score
from scipy.stats import wilcoxon

from tools.prepare import extract_all_authors_decade, QueryCandidatesImpostors
from tools.constants import rng
from tools import compress

# ------------------------------------------------------------------------
# Global parameters
# ------------------------------------------------------------------------
MIN_CANDIDATES = 1
MAX_NUMBER_OF_SAMPLES = 50
C_GRID = [0.01, 0.1, 1, 10, 100]
NB_PROCS = int(os.getenv("NBPROC", "4"))
TRANSFORM_SAMPLE = 100_000
TEST_FRAC_IMP = 0.3          # portion of impostor windows kept for test only

# flags propagated to workers via env vars
FAST_ENV_KEY = "SVM_FAST"
SEARCH_ENV_KEY = "SVM_SEARCH"
ENSEMBLE_ENV_KEY = "SVM_ENSEMBLE"

# ------------------------------------------------------------------------
# Helper utils
# ------------------------------------------------------------------------

def _bootstrap(df: pd.DataFrame, frac: float) -> pd.DataFrame:
    """Bootstrap‑sample roughly *frac* rows (never returns 0 rows)."""
    if df.empty:
        return df
    n = max(1, int(np.ceil(frac * len(df))))
    return df.sample(n=n, replace=True, random_state=rng)


def _train_test_split_imp(df: pd.DataFrame, test_frac: float = TEST_FRAC_IMP):
    """Deterministic split of impostor pool → train / test (no overlap)."""
    if df.empty:
        return df, df
    test_df = df.sample(frac=test_frac, replace=False, random_state=rng)
    return df.drop(test_df.index), test_df

# ------------------------------------------------------------------------
# BDI worker (distance baseline) — unchanged except for metrics
# ------------------------------------------------------------------------

def run_bdi(pickled_experiment: bytes) -> Dict[str, Any]:
    experiment: QueryCandidatesImpostors = pickle.loads(pickled_experiment)

    # split impostors once
    imp_train, imp_test = _train_test_split_imp(experiment.impostors)

    # fit verifier on candidate + train impostors
    X_train = pd.concat([experiment.candidate[experiment.features], imp_train[experiment.features]])
    y_raw = pd.concat([experiment.candidate["var_author"], imp_train["var_author"]])
    y, _ = y_raw.factorize()

    scaler = StandardScaler(with_mean=False).fit(X_train)
    verifier = BDIVerifier(metric="minmax", nb_bootstrap_iter=1000, rnd_prop=0.35, random_state=rng)
    verifier.fit(scaler.transform(X_train), y)

    # probabilities for query + unseen impostors
    q_scaled = scaler.transform(experiment.query[experiment.features])
    imp_scaled = scaler.transform(imp_test[experiment.features]) if len(imp_test) else np.array([])
    proba_q = verifier.predict_proba(q_scaled, [y[0]] * len(q_scaled))
    proba_imp = verifier.predict_proba(imp_scaled, [y[0]] * len(imp_scaled)) if len(imp_scaled) else np.array([])

    # simple metrics (skip if no impostor sample)
    if len(proba_imp):
        y_true = np.concatenate([np.ones(len(proba_q)), np.zeros(len(proba_imp))])
        y_scores = np.concatenate([proba_q, proba_imp])
        roc_auc = roc_auc_score(y_true, y_scores)
        avg_prec = average_precision_score(y_true, y_scores)
        acc = accuracy_score(y_true, y_scores >= 0.5)
        f1 = f1_score(y_true, y_scores >= 0.5)
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

# ------------------------------------------------------------------------
# Launcher
# ------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BDI vs SVM benchmark (ensemble edition)")
    parser.add_argument("--engine", choices=["bdi", "svm"], default="svm")
    parser.add_argument("--fast", action="store_true", help="Use SGD + calibration (skip search)")
    parser.add_argument("--search", choices=["grid", "halving"], default="grid", help="Hyper-parameter search type")
    parser.add_argument("--ensemble", type=int, default=1, help="Number of bagging replicas (default 1)")
    args = parser.parse_args()

    # propagate flags to worker environment
    os.environ[FAST_ENV_KEY] = "1" if args.fast else "0"
    os.environ[SEARCH_ENV_KEY] = args.search
    os.environ[ENSEMBLE_ENV_KEY] = str(max(1, args.ensemble))

    # load artefacts produced in step_00
    with open("features.json") as f:
        features = json.load(f)

    authors = pd.read_pickle("authors.pickle")
    impostors = pd.read_pickle("impostors.pickle") if os.path.exists("impostors.pickle") else pd.read_pickle("authors.pickle")

    feature_cols = [c for c in authors.columns if c in features]
    meta_cols = [c for c in authors.columns if c not in features]

    worker = run_bdi if args.engine == "bdi" else run_svm
    suffix = "bdi" if args.engine == "bdi" else "svm"

    progress = tqdm.tqdm()

    for gap, ascending in product([1, 5, 10, 15], [True, False]):
        # fit global transformer on a sample of all author windows
        fit_df = authors[feature_cols]
        if len(fit_df) > TRANSFORM_SAMPLE:
            fit_df = fit_df.sample(n=TRANSFORM_SAMPLE, random_state=rng)
        transformer = Pipeline([
            ("log", FunctionTransformer(np.log1p, validate=False)),
            ("scale", StandardScaler(with_mean=False)),
        ]).fit(fit_df)
        transformer_blob = pickle.dumps(transformer)

        results: List[Dict[str, Any]] = []
        with ProcessPoolExecutor(max_workers=NB_PROCS) as executor:
            futures = []
            for exp in extract_all_authors_decade(
                df=authors[meta_cols + feature_cols],
                features=features,
                gap=gap,
                ascending=ascending,
                min_candidates=MIN_CANDIDATES,
                general_impostors=impostors[meta_cols + feature_cols],
                as_pickle=True,
            ):
                payload = pickle.dumps((exp, transformer_blob)) if suffix == "svm" else exp
                futures.append(executor.submit(worker, payload))

            for fut in as_completed(futures):
                results.append(fut.result())
                progress.update(1)

        # summarise metrics for svm
        if suffix == "svm":
            roc_vals = [r["metrics"]["roc_auc"] for r in results if r.get("metrics")]
            if roc_vals:
                summary = {m: round(float(np.nanmean([r["metrics"][m] for r in results if r.get("metrics")])), 3)
                           for m in ("roc_auc", "avg_prec", "accuracy", "f1")}
                # optional Wilcoxon vs BDI
                bdi_file = f"results-bdi-{gap}-{ascending}.json.gz"
                if os.path.isfile(bdi_file):
                    with gzip.open(bdi_file, "rt") as fh:
                        bdi_results = json.load(fh)
                    bdi_roc = [r["metrics"]["roc_auc"] for r in bdi_results if isinstance(r, dict) and r.get("metrics")]
                    if len(bdi_roc) == len(roc_vals):
                        _, p = wilcoxon(roc_vals, bdi_roc, alternative="greater")
                        summary["wilcoxon_p"] = round(float(p), 4)
                results.append({"__summary__": summary})

        fname = f"results-{suffix}{'-fast' if args.fast and suffix=='svm' else ''}-ens{args.ensemble}-{gap}-{ascending}.json"
        compress.dump(results, fname)

    progress.close()
