import glob
import os.path as op
import dataclasses
from typing import List, Dict, Optional, Tuple
import unicodedata
import pandas as pd
import regex as re
import collections


_datapath = op.abspath(op.join(op.dirname(__file__), "..", "data"))



def split(text: str) -> List[str]:
    """Split a text into tokens """
    return re.sub(r"[\p{P}\s]+", " ", text).split()


@dataclasses.dataclass
class Metadata:
    author: str
    date: int
    author: str
    date: int
    title: str
    title: str
    window: str


@dataclasses.dataclass
class Text:
    author: str
    date: int
    title: str
    text: str
    impostor: bool = False


    def tokens(self) -> Dict[str, int]:
        return collections.Counter(split(self.text))

    def window_tokens(self, window: int = 2000) -> List[Dict[str, int]]:
        """ Retrieve various windows of size $window every $window/2
        """
        tokens = split(self.text)
        start = 0
        step = window // 2
        counters = []
        while start < len(tokens):
            end = min(start + window, len(tokens))
            if end-start == window:
                counters.append(collections.Counter(tokens[start:end]))
            start += step
        return counters



def normalize(string: str, case: bool = False) -> str:
    """ Normalizes whitespace, except for newlines
    """
    text = re.sub(r"[^\S\r\n]+", " ", unicodedata.normalize("NFKC", string))
    if case:
        return text.lower()
    return text


def load_authors(case: bool = False) -> List[Text]:
    data = []
    for file in glob.glob(f"{_datapath}/authors/*/*.txt"):
        au, da, ti = op.basename(file).replace(".txt", "").split("_")
        with open(file, encoding="utf8", errors='ignore') as f:
            data.append(
                Text(au, int(da), ti, normalize(f.read(), case=case))
            )
    return data


def load_impostors(case: bool = False) -> List[Text]:
    data = []
    for file in glob.glob(f"{_datapath}/impostors/*.txt"):
        au, da, *ti = op.basename(file).replace(".txt", "").split("_")
        with open(file, encoding="utf8", errors='ignore') as f:
            data.append(
                Text(au, int(da), "_".join(ti), normalize(f.read(), case=case), impostor=True)
            )
    return data


def filter_sparse_columns(
    df: pd.DataFrame,
    min_freq: Optional[int] = None,
    min_row_occurrence: Optional[int] = None
) -> pd.DataFrame:
    """ Filters out columns based on frequency thresholds.

    Args:
        df: Input DataFrame.
        min_freq: Minimum number of non-zero values required for a column to be kept.
        min_row_occurrence: Minimum number of rows where the column has a value > 0.

    Returns:
        Filtered DataFrame.
    """
    result = df.copy()

    if min_freq is not None:
        result = result.loc[:, (result != 0).sum() >= min_freq]

    if min_row_occurrence is not None:
        result = result.loc[:, (result > 0).sum() >= min_row_occurrence]

    return result


def texts_to_dataframe(
        texts: List[Text],
        min_freq: int = 1,
        min_row_occurence: int = 1,
        window: Optional[int] = None
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Converts a list of text metadata dictionaries into a pandas DataFrame of feature vectors.

    Each input dictionary must contain:
        - "author": the author of the text
        - "name": a unique name/identifier for the text
        - "text": the raw text string to extract features from

    The function performs the following:
        1. Extracts features from each text using a trigram-based method (length=3) for features that require it.
        2. Combines all features across texts to determine a global feature set.
        3. Builds a DataFrame with rows for each text and columns for each feature (zero-filled when missing).
        4. Adds metadata columns for "author" and sets "name" as the index.

    Returns:
        A tuple (df, features), where:
        - df is a pandas DataFrame containing feature vectors and metadata.
        - features is a list of feature names in the order used in the DataFrame.
    """

    # Step 1: Extract features from each text (as Counters or dict-like)
    if window:
        raw_feats: List[Dict[str, int]] = []
        labels = []
        for text in texts:
            for idx, counter in enumerate(text.window_tokens(window=window)):
                raw_feats.append(counter)
                labels.append(Metadata(text.author, text.date, text.title, window=f"[{idx*window}:{(idx+1)*window}]"))
    else:
        raw_feats: List[Dict[str, int]] = [text.tokens() for text in texts]
        labels = [Metadata(text.author, text.date, text.title, "full") for text in texts]


    # Step 2: Aggregate all features into a global frequency Counter
    all_feats: Dict[str, int] = raw_feats[0].copy()
    for local_feat in raw_feats[1:]:
        all_feats += local_feat

    # Step 3: Create a consistent feature order based on frequency
    features = [feature for feature, _ in all_feats.most_common(len(all_feats))]

    # Step 4: Convert list of Counters into a DataFrame, filling missing values with zero
    all_feats_df = pd.DataFrame(raw_feats).fillna(0)

    # Step 5: Reorder columns to match the global feature order
    all_feats_df = all_feats_df[features]

    # Step 6: Filter unusable values
    filtered = filter_sparse_columns(
        all_feats_df, min_freq=min_freq, min_row_occurrence=min_row_occurence
    ).astype(float)
    features = [col for col in filtered.columns if col != "name" and col != "author"]

    # Step 7: Add metadata columns for author and name
    all_feats_df = pd.concat([
        pd.DataFrame(
            [
                [label.author, label.date, label.title]
                for label in labels
            ], columns=["var_author", "var_date", "var_title"]),
        filtered
    ], axis=1)

    # Step 7: Use 'name' as the index
    all_feats_df.set_index("var_title", inplace=True)


    return all_feats_df, features

if __name__ == "__main__":
    # print([sum(val.values()) for val in load_authors()[0].window_tokens()])
    print(texts_to_dataframe(load_authors(True))[0])
# print(load_impostors()[0])