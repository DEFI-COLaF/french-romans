from tools.extract import load_authors, load_impostors, texts_to_dataframe
from tools.temp_fnwords import ALL
import json


def fnwords_extract(word: str) -> bool:
    return word in ALL

if __name__ == "__main__":
    authors, authors_features = texts_to_dataframe(
        load_authors(True),
        window=5000,
        extract_features=fnwords_extract
    )
    authors.to_pickle("authors.pickle")
    impostors, impostors_features = texts_to_dataframe(
        load_impostors(True),
        window=5000,
        extract_features=fnwords_extract
    )
    impostors.to_pickle("impostors.pickle")
    features = impostors_features + [feat for feat in authors_features if feat not in authors_features]
    with open("features.json", "w") as f:
        json.dump(features, f)
