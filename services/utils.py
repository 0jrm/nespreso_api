# services/utils.py
from __future__ import annotations
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
from typing import Mapping

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


def apply_netcdf_global_attributes(ds: xr.Dataset, extra_attrs: Mapping[str, str] | None = None) -> xr.Dataset:
    """
    Ensure NeSPReSO global attributes are present on an xarray Dataset.

    Does not remove existing attributes, only updates/sets the keys below:
      - coordinate_system: geographic
      - institution: COAPS, FSU
      - author: Jose Roberto Miranda
      - contact: jrm22n@fsu.edu
      - DOI: https://doi.org/10.1016/j.ocemod.2025.102550

    Any extra attributes provided will also be applied (overriding defaults).

    Returns the same Dataset instance for chaining.
    """
    defaults = {
        "coordinate_system": "geographic",
        "institution": "COAPS, FSU",
        "author": "Jose Roberto Miranda",
        "contact": "jrm22n@fsu.edu",
        "DOI": "https://doi.org/10.1016/j.ocemod.2025.102550",
    }
    if extra_attrs:
        defaults.update({str(k): str(v) for k, v in dict(extra_attrs).items()})
    try:
        ds.attrs.update(defaults)
    except Exception:
        # Fallback: set individually to avoid failure if attrs is read-only-like
        for k, v in defaults.items():
            try:
                ds.attrs[k] = v
            except Exception:
                pass
    return ds
