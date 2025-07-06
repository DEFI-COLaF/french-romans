# Example code
from tools.extract import load_authors, load_impostors, texts_to_dataframe

authors, authors_features = texts_to_dataframe(load_authors(True), window=5000)
impostors, impostors_features = texts_to_dataframe(load_impostors(True), window=5000)
metadata_cols = [col for col in authors.columns if col not in authors_features]

# We now have two dataframe whose size has been largely reduced !
# Now we loed function words
from functionwords.loader import load
fnwords = load("fr_21c")
authors_features = [feat for feat in authors_features if feat in fnwords.all]
impostors_features = [feat for feat in impostors_features if feat in fnwords.all]
features = impostors_features + [feat for feat in authors_features if feat not in authors_features]

# Now, we extract experiment situation from the author dataset
from tools.prepare import get_relative_frequencies, extract_all_authors_decade
from itertools import product
import pandas as pd
# Set MIN_Candidates (used either in training set or as candidates for BPI)
MIN_CANDIDATES = 1


for (gap, ascending) in product([0, 5, 10, 15], [True, False]):
    for experiment in extract_all_authors_decade(
            df=authors[metadata_cols+authors_features],
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