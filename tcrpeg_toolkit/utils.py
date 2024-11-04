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