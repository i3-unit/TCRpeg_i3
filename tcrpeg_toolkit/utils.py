import os
import re
import inspect
import logging
import warnings

import pandas as pd
import numpy as np

# Configure logging to display messages with timestamp and log level
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configure logging to display messages only
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')


def load_data(input_data, message=True):
    """
    Loads the input data from a specified source, which can be a CSV file, a Pandas DataFrame, or a numpy array.
    Parameters:
    input_data (str, pd.DataFrame, np.ndarray): The input data to be loaded. It can be a file path to a CSV or .npy file, 
                                                a Pandas DataFrame, or a numpy array.
    message (bool): If True, logs messages indicating the progress of data loading. Default is True.
    Returns:
    pd.DataFrame or np.ndarray: The loaded data as a Pandas DataFrame or numpy array.
    Raises:
    FileNotFoundError: If the specified file does not exist.
    """
    if message:
        logging.info("Loading data...")

    if input_data is None:
        logging.warning("No input data provided.") if message else None
        return None

    if isinstance(input_data, pd.DataFrame):
        if message:
            logging.info("Data loaded from Pandas DataFrame.")
        return input_data

    elif isinstance(input_data, np.ndarray):
        if message:
            logging.info("Data loaded from numpy array.")
        return input_data

    elif input_data.endswith('.npy'):
        try:
            data = np.load(input_data, allow_pickle=True)
            logging.info(f"Data loaded from {input_data}")
            return data
        except FileNotFoundError:
            logging.error(f"File not found: {input_data}")
            raise FileNotFoundError(f"File not found: {input_data}")

    else:
        try:
            file_extension = os.path.splitext(input_data)[1].lower()
            if file_extension == '.csv':
                data = pd.read_csv(input_data)
            elif file_extension in ['.tsv', '.txt']:
                data = pd.read_csv(input_data, sep='\t')
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
            logging.info(f"Data loaded from {input_data}")
            return data
        except FileNotFoundError:
            logging.error(f"File not found: {input_data}")
            raise FileNotFoundError(f"File not found: {input_data}")
        except ValueError as e:
            logging.error(str(e))
            raise


def filter_kwargs_for_function(func, kwargs):
    valid_keys = set(inspect.signature(func).parameters)
    return {key: value for key, value in kwargs.items() if key in valid_keys}


def apply_grouping_and_filtering(df, groupby=None, filter_values=None, filters=None):
    """
    Apply grouping and filtering to a DataFrame with optional combined or separate logic.

    Args:
        df (pd.DataFrame): The input DataFrame.
        groupby (list, optional): List of columns to group by.
        filter_values (dict, optional): Dictionary of filters (col -> values).
        filters (dict, optional): Combined argument where keys are columns and values:
                                  - Value to filter by.
                                  - None to indicate grouping.
    Returns:
        dict: Grouped and filtered DataFrames.
    """

    # Transform groupby to list
    groupby = groupby if isinstance(groupby, list) else [
        groupby] if groupby else None

    # Clean and lower columns
    df.columns = df.columns.str.strip().str.lower()

    # Clean all data from leading and trailing whitespaces
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

    # Clean inputs
    filter_values = {col.strip().lower(): val for col,
                     val in filter_values.items()} if filter_values else {}
    filters = {col.strip().lower(): val for col,
               val in filters.items()} if filters else {}
    groupby = [col.strip().lower() for col in groupby] if groupby else []

    # Parse combined `filters` if provided
    if filters:
        logging.info("Applying combined filters...")
        logging.info(f"Filters: {filters}")
        groupby.extend([col for col, val in filters.items() if val is None])
        filter_values.update(
            {col: val for col, val in filters.items() if val is not None})

    # Apply filtering
    if filter_values:
        logging.info("Applying filter values...")
        logging.info(f"Filter values: {filter_values}")
        mask = pd.Series([True] * len(df))
        for col, val in filter_values.items():
            mask &= df[col] == val
        df = df[mask]

    # Apply grouping
    if groupby:
        logging.info("Applying grouping...")
        logging.info(f"Grouping by: {groupby}")
        grouped = df.groupby(groupby)
        return {group: group_df for group, group_df in grouped}
    else:
        return {'all': df}

# def filter_kwargs_for_function(func, kwargs):
#     sig = inspect.signature(func)
#     parameters = sig.parameters
#     accepts_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values())

#     if accepts_var_keyword:
#         # Function accepts **kwargs, pass all through
#         return kwargs
#     else:
#         # Filter kwargs to only those that the function can accept
#         return {k: v for k, v in kwargs.items() if k in parameters}

# def expected_arguments(func):
#     sig = inspect.signature(func)
#     return [p.name for p in sig.parameters.values() if p.default == inspect.Parameter.empty]
