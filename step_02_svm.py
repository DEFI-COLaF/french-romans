import json
from itertools import product
import pandas as pd
from tools.prepare import extract_all_authors_decade

# Set MIN_Candidates (used either in training set or as candidates for BPI)
MIN_CANDIDATES = 1

# Load what was compiled in step_00
with open("features.json") as f:
    features = json.load(f)
authors = pd.read_pickle("authors.pickle")
impostors = pd.read_pickle("authors.pickle")
authors_features = [feat for feat in authors.columns]
impostors_features = [feat for feat in impostors.columns]
metadata_cols = [col for col in authors.columns if col not in authors_features]

# Now, we extract experiment situation from the author dataset
for (gap, ascending) in product([0, 5, 10, 15], [True, False]):
    for experiment in extract_all_authors_decade(
            df=authors[metadata_cols+authors_features],
            features=features,
            gap=gap,
            ascending=ascending,
            min_candidates=1,
            general_impostors=impostors[metadata_cols +impostors_features]
        ):
        # WE put everything together to all the same column order
        def train(same_author, not_same):
            return
        def test(model, valid_test_data):
            return

        train(experiment.candidate, experiment.impostors)
        test("", experiment.candidate)
        # Note, you could also sample from experiment.impostors, to test precision

        # Use the following for your logs:
        #   experiment.author, experiment.year, experiment.gap
        print("WOOHOOO")