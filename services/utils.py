import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta

def convert_to_numpy_array(data):
    """
    Convert input data to a numpy array if it's not already.
    Supports pandas Series, xarray DataArray, and lists.
    """
    if isinstance(data, (pd.Series, xr.DataArray)):
        return data.values
    elif not isinstance(data, np.ndarray):
        return np.array(data)
    return data

def convert_to_list_of_floats(data):
    """
    Ensure that the data is a list of floats.
    If data is already a list of floats, return it as is.
    """
    if isinstance(data, list) and all(isinstance(x, float) for x in data):
        return data
    return data.astype(float).tolist()

def convert_date_to_iso_strings(date):
    """
    Convert date inputs to a list of ISO 8601 strings ('YYYY-MM-DD').
    Handles numpy datetime64, Python datetime, and MATLAB datenum formats.
    """
    if isinstance(date, (pd.Series, xr.DataArray)):
        date = date.values
    elif isinstance(date, list):
        date = np.array(date)
    
    if np.issubdtype(date.dtype, np.datetime64) or isinstance(date[0], datetime):
        return [d.strftime('%Y-%m-%d') if isinstance(d, datetime) else str(d.astype('M8[D]')) for d in date]
    elif isinstance(date[0], (int, float)):  # Assuming MATLAB datenum
        matlab_origin = datetime(1, 1, 1) + timedelta(days=-366)
        return [(matlab_origin + timedelta(days=d)).strftime('%Y-%m-%d') for d in date]
    return date.tolist()  # Already in the correct format

def preprocess_inputs(lat, lon, date):
    """
    Preprocess the lat, lon, and date inputs to ensure they are in the correct format.
    Supports numpy arrays, pandas Series, xarray DataArray, Python datetime, and MATLAB datenum.
    
    Returns:
    - lat: list of floats
    - lon: list of floats
    - date: list of ISO 8601 strings ('YYYY-MM-DD')
    """
    lat = convert_to_list_of_floats(convert_to_numpy_array(lat))
    lon = convert_to_list_of_floats(convert_to_numpy_array(lon))
    date = convert_date_to_iso_strings(date)
    
    return lat, lon, date
