import torch
import numpy as np
from datetime import datetime
from scipy.interpolate import RegularGridInterpolator
from io_utils.coaps_io_data import get_aviso_by_date, get_sst_ghrsst_by_date, get_sss_by_date
from functools import lru_cache
from tenacity import retry, stop_after_attempt, wait_fixed, RetryError

# Helper: MATLAB datenum to np.datetime64
# MATLAB datenum 1.0 is 0000-01-01, Python datetime starts at 0001-01-01
# We'll use np.datetime64 for all time handling

def matlab_datenum_to_datetime64(matlab_datenum):
    # MATLAB datenum 1.0 is 0000-01-01, but Python's datetime64 starts at 0001-01-01
    # There are 366 days in year 0 in MATLAB
    days = matlab_datenum - 366
    return np.datetime64('0001-01-01') + np.timedelta64(int(days), 'D')

# LRU-cached wrappers for remote tile access
@lru_cache(maxsize=128)
def cached_get_sss_by_date(sss_folder, c_date, bbox):
    return get_sss_by_date(sss_folder, c_date, bbox)

@lru_cache(maxsize=128)
def cached_get_aviso_by_date(aviso_folder, c_date, bbox):
    return get_aviso_by_date(aviso_folder, c_date, bbox)

@lru_cache(maxsize=128)
def cached_get_sst_ghrsst_by_date(sst_folder, c_date, bbox):
    return get_sst_ghrsst_by_date(sst_folder, c_date, bbox)

@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
def load_satellite_data(times, lat, lon):
    """
    Load SSS, SST, and AVISO data for the given times, latitudes, and longitudes.
    Args:
        times: list/array of np.datetime64 or datetime
        lat, lon: arrays of coordinates
    Returns:
        sss_data, sst_data, aviso_data: arrays of satellite data
    """
    aviso_folder = "/unity/f1/ozavala/DATA/GOFFISH/AVISO/GoM/"
    sst_folder = "/unity/f1/ozavala/DATA/GOFFISH/SST/OISST"
    sss_folder = "/Net/work/ozavala/DATA/GOFFISH/SSS/SMAP_Global/"
    min_lat, max_lat, min_lon, max_lon = 18.0, 31.0, -98.0, -81.0
    ex_lon, ex_lat = -88.0, 23.0
    bbox = (min_lat, max_lat, min_lon, max_lon)
    unique_dates = sorted(list(set(times)))
    sss_data = np.nan * np.ones(len(times))
    sst_data = np.nan * np.ones(len(times))
    aviso_data = np.nan * np.ones(len(times))
    for c_date in unique_dates:
        date_idx = np.array([date_obj == c_date for date_obj in times])
        coordinates = np.array([lat[date_idx], lon[date_idx]]).T
        try:
            sss_datapoint, lats, lons = cached_get_sss_by_date(sss_folder, c_date, bbox)
            interpolator = RegularGridInterpolator((lats, lons), sss_datapoint.sss_smap_40km.values, bounds_error=False, fill_value=None)
            sss_data[date_idx] = interpolator(coordinates)
            if (sss_data[date_idx] < 0).any() or (sss_data[date_idx] > 45).any():
                sss_data[date_idx] = np.nan
        except Exception:
            pass
        try:
            aviso_adt, aviso_lats, aviso_lons = cached_get_aviso_by_date(aviso_folder, c_date, bbox)
            lons_grid, lats_grid = np.meshgrid(aviso_lons, aviso_lats)
            inclusion_mask = (lats_grid >= min_lat) & (lats_grid <= max_lat) & (lons_grid >= min_lon) & (lons_grid <= max_lon)
            exclusion_mask = (lats_grid < ex_lat) & (lons_grid > ex_lon)
            combined_mask = inclusion_mask & ~exclusion_mask
            daily_avg = np.nanmean(aviso_adt.adt.values[combined_mask])
            interpolator_ssh = RegularGridInterpolator((aviso_lats, aviso_lons), aviso_adt.adt.values, bounds_error=False, fill_value=None)
            aviso_data[date_idx] = interpolator_ssh(coordinates) - daily_avg
        except Exception:
            pass
        try:
            sst_date, sst_lats, sst_lons = cached_get_sst_ghrsst_by_date(sst_folder, c_date, bbox)
            interpolator_sst = RegularGridInterpolator((sst_lats, sst_lons), sst_date.analysed_sst.values[0], bounds_error=False, fill_value=None)
            sst_data[date_idx] = interpolator_sst(coordinates)
            if (sst_data[date_idx] < 0).any() or (sst_data[date_idx] > 350).any():
                sst_data[date_idx] = np.nan
        except Exception:
            pass
    return sss_data, sst_data, aviso_data

def prepare_inputs(time, lat, lon, sss, sst, ssh, input_params):
    """
    Transforms the individual data arrays into the format expected by the model.
    Args:
        time (array): Time data (datenum or float days)
        lat (array): Latitude data
        lon (array): Longitude data
        sss (array): Sea Surface Salinity data
        sst (array): Sea Surface Temperature data
        ssh (array): Sea Surface Height data
        input_params (dict): Dictionary indicating which features to include.
    Returns:
        torch.Tensor: Tensor of transformed input data.
    """
    num_samples = len(time)
    inputs = []
    for i in range(num_samples):
        sample_inputs = []
        if input_params.get("timecos", False):
            sample_inputs.append(np.cos(2 * np.pi * (time[i] % 365) / 365))
        if input_params.get("timesin", False):
            sample_inputs.append(np.sin(2 * np.pi * (time[i] % 365) / 365))
        if input_params.get("latcos", False):
            sample_inputs.append(np.cos(2 * np.pi * (lat[i] / 180)))
        if input_params.get("latsin", False):
            sample_inputs.append(np.sin(2 * np.pi * (lat[i] / 180)))
        if input_params.get("loncos", False):
            sample_inputs.append(np.cos(2 * np.pi * (lon[i] / 360)))
        if input_params.get("lonsin", False):
            sample_inputs.append(np.sin(2 * np.pi * (lon[i] / 360)))
        if input_params.get("sat", False):
            if input_params.get("sss", False):
                sample_inputs.append(sss[i])
            if input_params.get("sst", False):
                sample_inputs.append(sst[i] - 273.15)
            if input_params.get("ssh", False):
                sample_inputs.append(ssh[i])
        inputs.append(torch.tensor(sample_inputs, dtype=torch.float32))
    inputs_tensor = torch.stack(inputs)
    return inputs_tensor

# Output validation utility
def validate_accessor_output(tensor, expected_shape=None):
    assert isinstance(tensor, torch.Tensor), "Output is not a torch.Tensor"
    if expected_shape:
        assert tensor.shape == expected_shape, f"Shape {tensor.shape} != expected {expected_shape}"
    assert not torch.isnan(tensor).any(), "Output contains NaNs"
    return True 