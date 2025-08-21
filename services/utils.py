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
    Handles nested lists/arrays that contain single values.
    """
    if isinstance(data, list) and all(isinstance(x, float) for x in data):
        return data
    
    # Convert to numpy array first
    data_array = convert_to_numpy_array(data)
    
    # If the array contains nested arrays/lists, flatten them
    if data_array.ndim > 1:
        data_array = data_array.flatten()
    
    # Convert to list of floats
    return data_array.astype(float).tolist()

def convert_date_to_iso_strings(date):
    """
    Convert date inputs to a list of ISO 8601 strings ('YYYY-MM-DD').
    Handles numpy datetime64, Python datetime, and MATLAB datenum formats.
    """
    if isinstance(date, (pd.Series, xr.DataArray)):
        date = date.values
    elif isinstance(date, list):
        date = np.array(date)
    
    # Handle MATLAB datenum (floating point numbers)
    if np.issubdtype(date.dtype, np.floating) or (len(date) > 0 and isinstance(date[0], (int, float))):
        # MATLAB datenum: days since 0000-01-01 (with leap year corrections)
        # Use a safer approach by starting from year 1
        iso_dates = []
        for d in date:
            try:
                # MATLAB datenum starts from year 0, but Python datetime doesn't support year 0
                # So we need to adjust by adding 366 days to account for the difference
                days_since_year1 = float(d) - 366
                # Start from year 1, month 1, day 1
                base_date = datetime(1, 1, 1)
                target_date = base_date + timedelta(days=days_since_year1)
                iso_dates.append(target_date.strftime('%Y-%m-%d'))
            except Exception as e:
                # Fallback: try to convert to string
                iso_dates.append(str(d))
        return iso_dates
    
    # Handle numpy datetime64
    elif np.issubdtype(date.dtype, np.datetime64):
        return [str(d.astype('M8[D]')) for d in date]
    
    # Handle Python datetime objects
    elif len(date) > 0 and isinstance(date[0], datetime):
        return [d.strftime('%Y-%m-%d') for d in date]
    
    # If already strings, return as is
    elif len(date) > 0 and isinstance(date[0], str):
        return date.tolist()
    
    # Fallback: try to convert to string
    else:
        return [str(d) for d in date]

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
