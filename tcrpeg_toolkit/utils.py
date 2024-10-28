import os
import re
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
            raise FileNotFoundErr