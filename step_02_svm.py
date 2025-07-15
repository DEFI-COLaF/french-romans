# ---------------- mandatory import block -------------------------------
import json
import os
import pickle
from itertools import product
from typing import Dict, Any, List, Optional

from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import numpy as np
import tqdm
from sklearn.preprocessing import StandardScaler
from ruzicka.BDIVerifier import BDIVerifier

# ---------------- extra deps for SVM path ------------------------------
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
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, balanced_accuracy_score, make_scorer
from scipy.stats import wilcoxon

from tools.prepare import extract_all_authors_decade, QueryCandidatesImpostors
from tools.constants import rng
from tools import compress

# ---------------- global parameters ------------------------------------
MIN_CANDIDATES = 2
MAX_NUMBER_OF_SAMPLES = 50
C_GRID = [0.01, 0.1, 1, 10, 100]
NB_PROCS = int(os.getenv("NBPROC", "4"))
TRANSFORM_SAMPLE = 100_000
TEST_FRAC_IMP = 0.3

FAST_ENV_KEY = "SVM_FAST"
SEARCH_ENV_KEY = "SVM_SEARCH"
ENSEMBLE_ENV_KEY = "SVM_ENSEMBLE"

# ---------------- helper utils -----------------------------------------

def _bootstrap(df: pd.DataFrame, frac: float, seed: int = 42) -> pd.DataFrame:
    if df.empty:
        return df
    n = max(1, int(np.ceil(frac * len(df))))
    return df.sample(n=n, replace=True, random_state=np.random.default_rng(seed))

def _train_test_split_imp(df: pd.DataFrame, test_frac: float = TEST_FRAC_IMP):
    if df.empty:
        return df, df

    # un seul imposteur → tout en test
    if len(df) == 1:
        return df.iloc[:0], df

    test_df = df.sample(frac=test_frac, replace=False, random_state=rng)

    # si le tirage aléatoire donne un test vide, on force 1 imposteur en test
    if test_df.empty:
        test_df = df.sample(1, random_state=rng)

    return df.drop(test_df.index), test_df



# ---------------- BDI worker -------------------------------------------

def run_bdi(pickled_experiment: bytes) -> Dict[str, Any]:
    experiment: QueryCandidatesImpostors = pickle.loads(pickled_experiment)
    imp_train, imp_test = _train_test_split_imp(experiment.impostors)

    X_train = pd.concat([experiment.candidate[experiment.features], imp_train[experiment.features]])
    y_raw = pd.concat([experiment.candidate["var_author"], imp_train["var_author"]])
    y, _ = y_raw.factorize()

    scaler = StandardScaler(with_mean=False).fit(X_train)
    verifier = BDIVerifier(metric="minmax", nb_bootstrap_iter=1000, rnd_prop=0.35, random_state=rng)
    verifier.fit(scaler.transform(X_train), y)

    q_scaled = scaler.transform(experiment.query[experiment.features])
    imp_scaled = scaler.transform(imp_test[experiment.features]) if len(imp_test) else np.array([])

    proba_q = verifier.predict_proba(q_scaled, [y[0]] * len(q_scaled))
    proba_imp = verifier.predict_proba(imp_scaled, [y[0]] * len(imp_scaled)) if len(imp_scaled) else np.array([])

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
        "dist_arrays": np.round(verifier._dist_arrays, 4).tolist(),
        "probas": np.round(proba_q, 3).tolist(),
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

# ---------------- SVM worker -------------------------------------------

BALACC_SCORER = make_scorer(balanced_accuracy_score)

def run_svm(
    pickled_payload: bytes,
    fast: bool = False,
    search_mode: str = "halving",
    n_ensemble: int = 20,
) -> Optional[Dict[str, Any]]:
    """Entraîne/évalue un ensemble de SVM avec pare-feu classe unique."""

    experiment = pickle.loads(pickled_payload)

    # -------- split imposteurs --------------------------------------
    imp_train, imp_test = _train_test_split_imp(experiment.impostors)

    # -------- transformer ------------------------------------------
    fit_df = imp_train[experiment.features]
    if len(fit_df) > TRANSFORM_SAMPLE:
        fit_df = fit_df.sample(TRANSFORM_SAMPLE, random_state=rng)

    transformer = Pipeline(
        [
            ("log",   FunctionTransformer(np.log1p, validate=False)),
            ("scale", StandardScaler(with_mean=False)),
        ]
    )
    transformer.fit(fit_df)

    q_df       = experiment.query[experiment.features]
    q_t        = transformer.transform(q_df)
    imp_test_t = (
        transformer.transform(imp_test[experiment.features]) if len(imp_test) else np.array([])
    )

    # -------- ensemble ---------------------------------------------
    proba_q_sum   = np.zeros(len(q_df))
    proba_imp_sum = np.zeros(len(imp_test)) if len(imp_test) else np.array([])
    margin_sum    = np.zeros(len(q_df))
    n_valid       = 0

    for rep in range(n_ensemble):
        cand_boot = _bootstrap(experiment.candidate, 0.7, seed=rep)
        imp_boot  = _bootstrap(imp_train,         0.7, seed=rep)

        X_train = pd.concat(
            [cand_boot[experiment.features], imp_boot[experiment.features]]
        )
        y_raw = pd.concat(
            [cand_boot["var_author"], imp_boot["var_author"]]
        )
        y = (y_raw == experiment.author).astype(int).values

        # -- pare-feu classe unique --
        if len(np.unique(y)) < 2:
            continue

        X_train_t = transformer.transform(X_train)

        if fast:
            base = SGDClassifier(
                loss="hinge", alpha=1e-4, class_weight="balanced", random_state=rep
            )
            clf = CalibratedClassifierCV(base, method="sigmoid", cv=3).fit(X_train_t, y)
        else:
            svc = SVC(
                kernel="linear", class_weight="balanced", probability=True, random_state=rep
            )
            
            n_pos = int((y == 1).sum())
            n_neg = len(y) - n_pos

            # si trop peu de positifs ou de négatifs, on saute cette réplication
            if min(n_pos, n_neg) < 2:
                continue                             

            # choisir un nombre de folds compatible
            n_splits = min(3, n_pos, n_neg)            # 2 ou 3 folds selon la dispo
            SearchCls = HalvingGridSearchCV if search_mode == "halving" else GridSearchCV
            search = SearchCls(
                Pipeline([("svc", svc)]),
                {"svc__C": C_GRID},
                scoring=BALACC_SCORER,
                cv=StratifiedKFold(n_splits, shuffle=True, random_state=rep),
                n_jobs=1,
                refit=True,
                error_score=np.nan,
            )
            search.fit(X_train_t, y)
            clf = search.best_estimator_

        # -- accumulate predictions --
        proba_q_sum   += clf.predict_proba(q_t)[:, 1]
        if hasattr(clf, "decision_function"):
            margin_sum += clf.decision_function(q_t)
        if len(imp_test_t):
            proba_imp_sum += clf.predict_proba(imp_test_t)[:, 1]

        n_valid += 1

    # -------- aucun modèle valide ----------------------------------
    if n_valid == 0:
        return None

    # -------- moyennes ---------------------------------------------
    proba_q_avg   = proba_q_sum  / n_valid
    margin_avg    = margin_sum   / n_valid
    proba_imp_avg = proba_imp_sum / n_valid if len(imp_test) else np.array([])

    # -------- métriques --------------------------------------------
    if len(imp_test):
        y_true   = np.concatenate([np.ones(len(q_df)), np.zeros(len(imp_test))])
        y_scores = np.concatenate([proba_q_avg, proba_imp_avg])
        roc_auc  = roc_auc_score(y_true, y_scores)
        avg_prec = average_precision_score(y_true, y_scores)
        acc      = accuracy_score(y_true, y_scores >= 0.5)
        bal_acc  = balanced_accuracy_score(y_true, y_scores >= 0.5)
        f1       = f1_score(y_true, y_scores >= 0.5)
    else:
        roc_auc = avg_prec = bal_acc = acc = f1 = np.nan

    labels = [
        f"{fn}#{w}" for w, fn in zip(experiment.query["var_window"], experiment.query.index)
    ]

    return {
        "margin": np.round(margin_avg, 3).tolist(),
        "probas": np.round(proba_q_avg, 3).tolist(),
        "ensemble_n": n_valid,
        "metrics": {
            #"roc_auc":  None if np.isnan(roc_auc)  else round(roc_auc, 3),
            #"avg_prec": None if np.isnan(avg_prec) else round(avg_prec, 3),
            "accuracy": None if np.isnan(acc)      else round(acc, 3),
            "balanced_acc": None if np.isnan(bal_acc) else round(bal_acc, 3),
            "f1":       None if np.isnan(f1)       else round(f1, 3),
        },
        "labels": labels,
        "date":   int(experiment.year),
        "gap":    experiment.gap,
        "author": experiment.author,
    }

# ---------------- launcher ---------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser("BDI vs SVM benchmark")
    parser.add_argument("--engine", choices=["bdi", "svm"], default="svm")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--search", choices=["grid", "halving"], default="grid")
    parser.add_argument("--ensemble", type=int, default=1)
    args = parser.parse_args()

    os.environ[FAST_ENV_KEY] = "1" if args.fast else "0"
    os.environ[SEARCH_ENV_KEY] = args.search
    os.environ[ENSEMBLE_ENV_KEY] = str(max(1, args.ensemble))

    # load artefacts from step_00
    with open("features.json") as f:
        features = json.load(f)
    authors = pd.read_pickle("authors.pickle")
    impostors = pd.read_pickle("impostors.pickle")
    authors_features = [feat for feat in authors.columns]
    impostors_features = [feat for feat in impostors.columns]
    metadata_cols = [col for col in authors.columns if col not in authors_features]

    worker = run_bdi if args.engine == "bdi" else run_svm
    suffix = "bdi" if args.engine == "bdi" else "svm"

    bar = tqdm.tqdm()

    for gap, ascending in product([1, 5, 10, 15], [True, False]):
        # fit transformer once per (gap, direction)
        results: List[Dict[str, Any]] = []
        with ProcessPoolExecutor(max_workers=NB_PROCS) as ex:
            futs = [
                ex.submit(worker, exp)
                for exp in extract_all_authors_decade(
                    df=authors[metadata_cols + authors_features],
                    features=features,
                    gap=gap,
                    ascending=ascending,
                    min_candidates=MIN_CANDIDATES,
                    general_impostors=impostors[metadata_cols + impostors_features],
                    as_pickle=True,
                )
            ]

            for fut in as_completed(futs):
                res = fut.result()
                if res is not None:     
                    results.append(res)
        
                bar.update(1)

        # summary block for SVM
        # if suffix == "svm":
        #     roc_vals = [r["metrics"]["roc_auc"] for r in results if r.get("metrics") and r["metrics"]["roc_auc"] is not None]
        #     if roc_vals:
        #         summary = {m: round(float(np.nanmean([r["metrics"][m] for r in results if r.get("metrics")])), 3)
        #                    for m in ("roc_auc", "avg_prec", "accuracy", "f1")}
        #         bdi_file = f"results-bdi-{gap}-{ascending}.json.gz"
        #         if os.path.isfile(bdi_file):
        #             with gzip.open(bdi_file, "rt") as fh:
        #                 bdi_res = json.load(fh)
        #             bdi_roc = [r["metrics"]["roc_auc"] for r in bdi_res if isinstance(r, dict) and r.get("metrics")]  # type: ignore
        #             if len(bdi_roc) == len(roc_vals):
        #                 _, p = wilcoxon(roc_vals, bdi_roc, alternative="greater")
        #                 summary["wilcoxon_p"] = round(float(p), 4)
        #         results.append({"__summary__": summary})

        fname = f"results-{suffix}{'-fast' if args.fast and suffix=='svm' else ''}-ens{args.ensemble}-{gap}-{ascending}.json"
        compress.dump(results, fname)

    bar.close()
