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

from tools.prepare import extract_all_authors_decade, QueryCandidatesImpostors
from tools.constants import rng
from tools import compress

# Set MIN_Candidates (used either in training set or as candidates for BPI)
MIN_CANDIDATES = 1
MAX_NUMBER_OF_SAMPLES = 50
NB_PROCS = int(os.getenv("NBPROC", 4))

# Define your subprocessed routine here
def run_bdi(pickled_experiment: bytes) -> Dict[str, Any]:
    import pickle  # Needed inside subprocess
    experiment: QueryCandidatesImpostors = pickle.loads(pickled_experiment)

    X = pd.concat([experiment.candidate[experiment.features], experiment.impostors[experiment.features]])
    Y_raw = pd.concat([experiment.candidate["var_author"], experiment.impostors["var_author"]])
    Y, label_uniques = Y_raw.factorize()

    # We scale things as we should in BDI
    scaler: StandardScaler = StandardScaler(with_mean=False).fit(X)
    X = scaler.transform(X)

    # WE set up the BDI Verifier
    bdi_mm = BDIVerifier(metric="minmax", nb_bootstrap_iter=1000, rnd_prop=.35, random_state=rng)
    bdi_mm.fit(X, Y)

    # We get our query scaled
    query = scaler.transform(experiment.query[experiment.features].sample(
        n=min([MAX_NUMBER_OF_SAMPLES, experiment.query.shape[0]]),
        random_state=rng,
        replace=True)
    )
    # Set labels as Filename:Window
    labels = [f'{fn}#{w}' for w, fn in zip(experiment.query["var_window"].tolist(), experiment.query.index.tolist())]

    # We query against candidate, hence the second list is a list of the author
    proba = bdi_mm.predict_proba(query, [Y[0]] * (query.shape[0]))

    return {
        "arrays": np.round(bdi_mm._dist_arrays, decimals=4).tolist(),
        "date": int(experiment.year),
        "probas": np.round(proba, decimals=3).tolist(),
        "labels": labels,
        "gap": experiment.gap,
        "author": experiment.author
    }


if __name__ == "__main__":
    # Load what was compiled in step_00
    with open("features.json") as f:
        features = json.load(f)
    authors = pd.read_pickle("authors.pickle")
    impostors = pd.read_pickle("impostors.pickle")
    authors_features = [feat for feat in authors.columns]
    impostors_features = [feat for feat in impostors.columns]
    metadata_cols = [col for col in authors.columns if col not in authors_features]

    # We add a TQDM progress bar
    bar = tqdm.tqdm()

    # Now, we extract experiment situation from the author dataset
    for gap in ["random", 1, 5, 10, 15, -1, -5, -10, -15]:
        results = []
        # if os.path.exists(f"results-bdi-{gap}-{ascending}.json"):
        #     continue
        with ProcessPoolExecutor(max_workers=NB_PROCS) as executor:
            futures = [
                executor.submit(run_bdi, experiment)
                for experiment in extract_all_authors_decade(
                    df=authors[metadata_cols+authors_features],
                    features=features,
                    gap=gap,
                    min_candidates=1,
                    general_impostors=impostors[metadata_cols+impostors_features],
                    as_pickle=True
                )]
            for future in as_completed(futures):
                results.append(future.result())
                bar.update(1)
        compress.dump(results, f"results-bdi-G{gap}.json")

