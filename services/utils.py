# services/utils.py
from __future__ import annotations
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta

def convert_to_numpy_array(data):
    if isinstance(data, (pd.Series, xr.DataArray)):
        return data.values
    return np.asarray(data)

def convert_to_list_of_floats(data):
    if isinstance(data, list) and all(isinstance(x, float) for x in data):
        return data
    arr = convert_to_numpy_array(data)
    if arr.ndim > 1:
        arr = arr.ravel()
    return arr.astype(float).tolist()

def convert_date_to_iso_strings(date):
    if isinstance(date, (pd.Series, xr.DataArray)):
        date = date.values
    if isinstance(date, list):
        date = np.array(date, dtype=object)

    arr = np.asarray(date)
    if arr.dtype.kind in {"f", "i"}:
        # MATLAB datenum → ISO
        out = []
        for d in arr:
            try:
                days_since_year1 = float(d) - 366.0
                base = datetime(1, 1, 1)
                out.append((base + timedelta(days=days_since_year1)).strftime('%Y-%m-%d'))
            except Exception:
                out.append(str(d))
        return out
    if np.issubdtype(arr.dtype, np.datetime64):
        return [str(d.astype('M8[D]')) for d in arr]
    if arr.size and isinstance(arr.flat[0], datetime):
        return [d.strftime('%Y-%m-%d') for d in arr]
    if arr.size and isinstance(arr.flat[0], str):
        return arr.tolist()
    return [str(d) for d in arr]

def preprocess_inputs(lat, lon, date):
    lat = convert_to_list_of_floats(convert_to_numpy_array(lat))
    lon = convert_to_list_of_floats(convert_to_numpy_array(lon))
    date = convert_date_to_iso_strings(date)
    return lat, lon, date
