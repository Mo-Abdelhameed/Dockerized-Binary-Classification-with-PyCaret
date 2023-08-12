import os
from typing import List
from schema.data_schema import BinaryClassificationSchema
from preprocessing.preprocess import *
from config import paths


def create_pipeline() -> List[Any]:
    """
        Creates pipeline of preprocessing steps

        Args:
            schema (BinaryClassificationSchema): BinaryClassificationSchema object carrying data about the schema
        Returns:
            A list of tuples containing the functions to be executed in the pipeline on a certain column
        """
    pipeline = [(drop_constant_features, None),
                (drop_all_nan_features, None),
                (drop_duplicate_features, None),
                (drop_mostly_missing_columns, None),
                (indicate_missing_values, None),
                ]
    return pipeline


def run_pipeline(data: pd.DataFrame, pipeline: List) -> pd.DataFrame:
    """
    Transforms the data by passing it through every step of the given pipeline.

    Args:
        data (pd.DataFrame): The data to be processed
        pipeline (List): A list of functions to be performed on the data.

    Returns:
        The transformed data
    """

    for stage, column in pipeline:
        data = stage(data)
    return data

