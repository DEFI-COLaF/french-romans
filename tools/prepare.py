import pandas as pd
from typing import Optional, Iterator, List, Union
from dataclasses import dataclass
import pickle


@dataclass
class QueryCandidatesImpostors:
    author: str # Author we are working against
    gap: int # Gap used to select candidates
    year: int # Year that was used to select query
    query: pd.DataFrame # A dataframe of N values to query
    candidate: pd.DataFrame  # A dataframe of N values to use as candidates
    impostors: pd.DataFrame  # A dataframe of other authors to use as impostors
    features: List[str] = None


def extract_author_decade(df: pd.DataFrame, author: str, ascending: bool = True, gap: int = 5,
                   min_candidates: int = 1
                   ) -> Iterator[QueryCandidatesImpostors]:
    """ Given the dataframe of authors we want to query, we iterate over each available series of texts written at
    current_work_year + gap (or -gap in descending mode), yielding a QueryCandidatesImpostors"""
    impostors = df[df.var_author != author].copy()
    subset = df[df.var_author == author].copy()

    # Iterate over each year
    for year in sorted(subset.var_date.unique(), reverse=ascending == False):
        # Find if there is a subset by checking that there are texts of the author on year + gap
        #   but also that there is enough candidates to train (min_candidates)
        # Formula changes if we are doing reverse (late versus early)
        if ascending:
            condition = (subset.var_date >= (year + gap))
        else:
            condition = (subset.var_date <= (year - gap))
            # Mark gap as negative
            gap = -abs(gap)
        print(ascending, gap)
        if condition.any() and len(set(subset[condition].index.tolist())) >= min_candidates and (subset.var_date == year).any():
            yield QueryCandidatesImpostors(
                author=author,
                gap=gap,
                year=year,
                query=subset[subset.var_date == year].copy(),
                candidate=subset[condition],
                impostors=impostors
            )


def extract_all_authors_decade(
        df: pd.DataFrame, general_impostors: pd.DataFrame, features: List[str],
        ascending: bool = True, gap: int = 5, min_candidates: int = 1,
        as_pickle: bool = False
) -> Iterator[Union[QueryCandidatesImpostors, bytes]]:
    """ Given the dataframe of authors we want to query, we iterate over each available series of texts written at
    current_work_year + gap (or -gap in descending mode), yielding a QueryCandidatesImpostors

    Unlike extract_author_decade, this yields on every author
    """
    for author in df.var_author.unique():
        for experiment in extract_author_decade(
            df, author, ascending, gap, min_candidates
        ):
            all_subsets = pd.concat(
                [experiment.impostors, general_impostors, experiment.candidate, experiment.query]
            ).fillna(0)
            features = [feat for feat in all_subsets.columns if not feat.startswith("var_")]
            all_subsets = get_relative_frequencies(all_subsets, features=features)

            nb_impostors = general_impostors.shape[0] + experiment.impostors.shape[0]

            candidate = all_subsets.iloc[nb_impostors:nb_impostors + experiment.candidate.shape[0], :]
            impostors = all_subsets.iloc[:nb_impostors, :]
            query = all_subsets.iloc[candidate.shape[0]+nb_impostors:, :]
            qci =  QueryCandidatesImpostors(
                author=experiment.author,
                year=experiment.year,
                gap=experiment.gap,
                features=features,
                query=query,
                impostors=impostors,
                candidate=candidate
            )
            if as_pickle:
                yield pickle.dumps(qci)
            else:
                yield qci


def get_relative_frequencies(dataframe: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """ Converts raw feature counts in the given DataFrame to relative frequencies (per row).

    Each feature value is divided by the sum of all feature values in the same row.
    This normalization is useful when comparing documents of different lengths.

    Parameters:
        dataframe (pd.DataFrame): The input DataFrame with feature columns and other metadata.
        features (List[str]): List of column names to normalize.

    Returns:
        pd.DataFrame: The modified DataFrame with features scaled to relative frequencies.
    """
    # Avoid modifying original DataFrame by making a copy
    df = dataframe.copy()
    df[features] = df[features].astype("float64")

    # Avoid division by zero by replacing zero row sums with NaN
    row_sums = df[features].sum(axis=1).astype("float64")

    # Normalize each feature by the row sum (document length)
    df.loc[:, features] = df[features].astype(float).div(row_sums, axis=0).fillna(.0)

    return df