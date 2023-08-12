import numpy as np
import pandas as pd
from typing import Any, Dict, Tuple


def indicate_missing_values(input_data: pd.DataFrame) -> pd.DataFrame:
    """
    Replaces empty strings with NaN in a dataframe.

    Args:
        input_data (ps.DataFrame): The dataframe to be processed.

    Returns:
        A dataframe after replacing empty strings with NaN.
    """
    return input_data.replace("", np.nan)


def drop_all_nan_features(input_data: pd.DataFrame) -> pd.DataFrame:
    """
    Drops columns that only contain NaN values.

    Args:
        input_data (pd.DataFrame): The dataframe to be processed.

    Returns:
        A dataframe after dropping NaN columns
    """
    return input_data.dropna(axis=1, how='all')


def percentage_of_missing_values(input_data: pd.DataFrame) -> Dict:
    """
    Calculates the percentage of missing values in each column of a given dataframe.

    Args:
        input_data (pd.DataFrame): The dataframe to calculate the percentage of missing values on.

    Returns:
        A dictionary of column names as keys and the percentage of missing values as values.
    """
    columns_with_missing_values = input_data.columns[input_data.isna().any()]
    return (input_data[columns_with_missing_values].isna().mean().sort_values(ascending=False) * 100).to_dict()


def drop_constant_features(input_data: pd.DataFrame) -> pd.DataFrame:
    """
    Drops columns that contain only one value.

    Args:
        input_data (pd.DataFrame): The dataframe to be processed.

    Returns:
        A dataframe after dropping constant columns
    """
    constant_columns = input_data.columns[input_data.nunique() == 1]
    return input_data.drop(columns=constant_columns)


def drop_duplicate_features(input_data: pd.DataFrame) -> pd.DataFrame:
    """
    Drops columns that are exactly the same and keeps only one of them.

    Args:
        input_data (pd.DataFrame): The dataframe to be processed.

    Returns:
        A dataframe after dropping duplicated columns
    """
    return input_data.T.drop_duplicates().T


def drop_mostly_missing_columns(input_data: pd.DataFrame, thresh=0.6) -> pd.DataFrame:
    """
    Drops columns in which NaN values exceeds a certain threshold.

    Args:
        input_data: (pd.DataFrame): the data to be processed.
        thresh (float): The threshold to use.

    Returns:
        A dataframe after dropping the specified columns.
    """
    threshold = int(thresh * len(input_data))
    return input_data.dropna(axis=1, thresh=threshold)
