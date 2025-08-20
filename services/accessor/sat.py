# TODO: on load_satellite_data, instead of always downloading the data, simply check if the data is already available, and return only the data that is available
# downloading the data should be done separately

import torch
import numpy as np
import xarray as xr
from datetime import date, datetime, timedelta
import calendar
from scipy.interpolate import RegularGridInterpolator
from functools import lru_cache
from tenacity import retry, stop_after_attempt, wait_fixed, RetryError
import os
import earthaccess
import copernicusmarine
import sys
from pathlib import Path
import tempfile, shutil, time, gc
from os.path import join

def get_day_of_year_from_month_and_day(month, day_of_month, year=datetime.now().year):
    """
    Gets a list of integers with the days of the month, starting from 0 and from the day of the year
    :param month:
    :param year:
    :return:
    """
    first_jan = date(year, 1, 1)
    day_of_year = date(year, month, day_of_month).toordinal() - first_jan.toordinal() + 1
    return day_of_year

# %% AVISO by date
def get_aviso_by_date(aviso_folder, c_date, bbox=None):
    '''
    Reads AVISO data for a specified date, and optionally crops to a specified bounding box.
    If the standard file naming convention fails, an alternative file naming convention is used.

    Parameters:
        aviso_folder (str): Directory containing AVISO files.
        c_date (datetime.date): Date for which to retrieve data.
        bbox (tuple of float, optional): Bounding box as (min_lat, max_lat, min_lon, max_lon).

    Returns:
        Tuple containing the AVISO dataset, latitudes, and longitudes.
    '''
    # Standard file naming format
    alternative_folder = '/home/jmiranda/Data/SSH/SEALEVEL_GLO_PHY_L4_NRT_008_046/'
    standard_format = join(aviso_folder, f"{c_date.year}-{c_date.month:02d}.nc")
    alternative_format = f"nrt_global_allsat_phy_l4_{c_date.strftime('%Y%m%d')}"
    
    # Attempt to load dataset using standard naming format
    try:
        aviso_data = xr.open_dataset(standard_format)
    except FileNotFoundError:
        # Alternative file naming format when standard file is not found
        files = os.listdir(alternative_folder)
        matching_files = [file for file in files if alternative_format in file]
        
        if not matching_files:
            raise FileNotFoundError(f"No files found for date {c_date} in {alternative_format}")
        
        try:
            aviso_data = xr.open_dataset(join(alternative_folder, matching_files[0]))
        except:
            raise RuntimeError(f"Could not load AVISO data for {c_date}, {matching_files[0]}")

    # Crop to bounding box if specified
    if bbox is not None:
        target_time = np.datetime64(c_date)
        # Calculate the absolute time differences
        time_diff = np.abs(aviso_data["time"] - target_time)
        # Get the index of the closest time
        closest_index = np.argmin(time_diff.values)
        
        # Select data for the closest time and within the specified bounding box
        aviso_data = aviso_data.sel(time=aviso_data["time"][closest_index],
                                    latitude=slice(bbox[0], bbox[1]),
                                    longitude=slice(bbox[2], bbox[3]))

    lats = aviso_data.latitude
    lons = aviso_data.longitude

    return aviso_data, lats, lons

# ========================= SST ======================================
# %% SST GHRSST by date
def get_sst_ghrsst_by_date(sst_folder, c_date, bbox=None):
    '''
    Reads SST single day for a given date. You can also specify a bounding box and the data will be cropped to that region.
    '''
    c_date_str = c_date.strftime("%Y%m%d")
    sst_file_name = join(sst_folder, str(c_date.year), f"{c_date_str}090000-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1_subset.nc")
    sst_data = xr.open_dataset(sst_file_name)
    if bbox is not None:
        sst_data = sst_data.sel( lat=slice(bbox[0],bbox[1]),
                                lon=slice(bbox[2],bbox[3]))

    lats = sst_data.lat
    lons = sst_data.lon

    return sst_data, lats, lons

# %% SST OSTIA by year
def get_sst_ostia_by_year(sst_folder, year, bbox=None):
    sst_file_name = join(sst_folder, f"OSTIA_SST_{year}.nc")
    sst_data = xr.open_dataset(sst_file_name)
    if bbox is not None:
        sst_data = sst_data.sel(lat=slice(bbox[0], bbox[1]), lon=slice(bbox[2], bbox[3]))

    lats = sst_data.latitude
    lons = sst_data.longitude

    return sst_data, lats, lons


# %% SSS by date
def get_sss_by_date(sss_folder, c_date, bbox=None):
    '''
    Reads salinity single day for a given date. You can also specify a bounding box and the data will be cropped to that region.
    '''
    c_date_str = c_date.strftime("%Y%m%d")

    day_of_year = get_day_of_year_from_month_and_day(c_date.month, c_date.day, year=datetime.now().year)

    sss_file_name = join(sss_folder, str(c_date.year), f"RSS_smap_SSS_L3_8day_running_{c_date.year}_{day_of_year:03d}_FNL_v05.0.nc") #old version 5.0
    try:
        sss_data = xr.open_dataset(sss_file_name)
    except Exception:
        sss_file_name = join(sss_folder, str(c_date.year), f"RSS_smap_SSS_L3_8day_running_{c_date.year}_{day_of_year:03d}_FNL_v06.0.nc") # new version 6.0
        try:
            sss_data = xr.open_dataset(sss_file_name)
        except Exception:
            raise Exception(f"Failed to load SSS data for date {c_date} with both v05.0 and v06.0 versions")
        
    if bbox is not None:
        sss_data = sss_data.sel( lat=slice(bbox[0],bbox[1]),
                                lon=slice((bbox[2] + 360)%360,(bbox[3] + 360)%360))

    lats = sss_data.lat
    lons = np.where(sss_data.lon > 180, sss_data.lon - 360, sss_data.lon)

    return sss_data, lats, lons

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

# ------------------------------------------------------------------
def _assert_writable(dir_):
    if not os.path.isdir(dir_):
        raise FileNotFoundError(f"Directory does not exist: {dir_}")
    if not os.access(dir_, os.W_OK):
        raise PermissionError(f"No write permission: {dir_}")

def _atomic_move(src, dst):
    """Move src→dst safely across file-systems."""
    try:
        os.replace(src, dst)          # python ≥3.3 – atomic if same FS
    except OSError:
        import shutil                 # cross-device fallback
        shutil.move(src, dst)
        
## Data availability check functions

def check_data_availability(
    dates: list, sss_root: str, sst_root: str, aviso_root: str
) -> list:
    """
    Check which dates have all required satellite data available.

    Args:
        dates (list): List of date strings in 'YYYY-MM-DD' format.
        sss_folder (str): Path to SSS data folder.
        sst_folder (str): Path to SST data folder.
        aviso_folder (str): Path to AVISO data folder.

    Returns:
        list: List of dates (as in input) for which all data is available.
    """

    def sss_file_exists(date_) -> bool:
        year_dir = os.path.join(sss_root, f"{date_.year:04d}")
        doy      = date_.timetuple().tm_yday
        fname    = f"RSS_smap_SSS_L3_8day_running_{date_.year}_{doy:03d}_FNL_v06.0.nc"
        final    = os.path.join(year_dir, fname)
        return os.path.isfile(final)

    def sst_file_exists(date_) -> bool:
        year_dir  = os.path.join(sst_root, f"{date_.year:04d}")
        fname     = f"{date_.strftime('%Y%m%d')}090000-" \
                    "JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1_subset.nc"
        final     = os.path.join(year_dir, fname)
        return os.path.isfile(final)

    def aviso_file_exists(date_) -> bool:
        month_tag = f"{date_.year}-{date_.month:02d}"
        final = Path(aviso_root) / f"{month_tag}.nc"
        return os.path.isfile(final)

    available_dates = [
        d for d in dates
        if sss_file_exists(d) and sst_file_exists(d) and aviso_file_exists(d)
    ]
    return available_dates

## Data download functions
# ------------------------------------------------------------------
#  1.  GHRSST / MUR  (daily) ---------------------------------------
# ------------------------------------------------------------------
def download_sst(sst_root, date_):
    year_dir  = os.path.join(sst_root, f"{date_.year:04d}")
    fname     = f"{date_.strftime('%Y%m%d')}090000-" \
                "JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1_subset.nc"
    final     = os.path.join(year_dir, fname)

    if os.path.exists(final):
        return final

    if not os.path.exists(year_dir):
        os.makedirs(year_dir)
    
    _assert_writable(year_dir)

    t0 = date_.strftime("%Y-%m-%dT09:00:00Z")
    results = earthaccess.search_data(
        short_name="MUR-JPL-L4-GLOB-v4.1",          # <- fixed name
        temporal=( (date_ - timedelta(hours=24)).isoformat()+"Z",
                (date_).isoformat()+"Z" )
    )

    if not results:
        print(f"SST not available for {t0}")
        return None

    with tempfile.TemporaryDirectory(dir=year_dir) as tmp:
        tmpfile = earthaccess.download(results[0], local_path=tmp)[0]
        os.replace(tmpfile, final)          # or _atomic_move(...)

    return final

# ------------------------------------------------------------------
#  2.  SMAP SSS  (8-day running mean) ------------------------------
# ------------------------------------------------------------------
def download_sss(sss_root, date_):
    year_dir = os.path.join(sss_root, f"{date_.year:04d}")
    doy      = date_.timetuple().tm_yday
    fname    = f"RSS_smap_SSS_L3_8day_running_{date_.year}_{doy:03d}_FNL_v06.0.nc"
    final    = os.path.join(year_dir, fname)

    if os.path.exists(final):
        return final
    
    if not os.path.exists(year_dir):
        os.makedirs(year_dir)
    
    _assert_writable(year_dir)

    t0 = date_.strftime("%Y-%m-%dT12:00:00Z")
    results = earthaccess.search_data(
        short_name="SMAP_RSS_L3_SSS_SMI_8DAY-RUNNINGMEAN_V6",
        temporal=( (date_ - timedelta(days=8)).isoformat()+"Z",
                (date_).isoformat()+"Z" )
    )
    
    if not results:
        print(f"SSS not available for {t0}")
        return None
    
    with tempfile.TemporaryDirectory(dir=year_dir) as tmp:
        tmpfile = earthaccess.download(results[0], local_path=tmp)[0]
        os.replace(tmpfile, final)          # or _atomic_move(...)

    return final

# ------------------------------------------------------------------
#  3.  AVISO / DUACS  (monthly) ------------------------------------
# ------------------------------------------------------------------
def download_aviso(aviso_root: str, date_: datetime) -> Path:
    """
    For the month containing *date_*:
    • Fetch every daily DUACS file (0.125° NRT, P1D) via Copernicus Marine.
    • Concatenate lazily, rechunk, and write a single <YYYY>-<MM>.nc
      in *aviso_root*.
    • The function is idempotent: if the monthly file already exists, it
      is returned immediately.

    Robustness / hygiene
    --------------------
    1. Reads daily granules with the *netcdf4* engine (read-only, releases
       file handles promptly).  
    2. Writes the monthly aggregate with *h5netcdf* (pure-python, thread-
       friendly).  
    3. Uses a manually-managed temporary directory; cleanup is forced in
       a finally-block, after closing datasets, running GC, and giving the
       OS a short grace period.  
    4. Final file move is atomic across filesystems.
    """
    month_tag = f"{date_.year}-{date_.month:02d}"
    final = Path(aviso_root) / f"{month_tag}.nc"
    if final.exists():
        return final

    _assert_writable(Path(aviso_root))           # your helper

    # ---------- create temp workspace ----------
    tmpdir = tempfile.mkdtemp(dir=aviso_root)    # manual cleanup
    try:
        pattern = f"*{date_.year}{date_.month:02d}??*.nc"  # YYYYMMDD
        resp = copernicusmarine.get(
            dataset_id=(
                "cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D"
            ),
            filter=pattern,
            output_directory=tmpdir,
            overwrite=False
        )
        daily_files = sorted(Path(f.file_path) for f in resp.files)
        if not daily_files:
            print(f"SSH not available for {pattern}")
            return None
        # ---------- read lazily, concat, decode ----------
        ds = xr.open_mfdataset(
            daily_files,
            combine="nested", concat_dim="time",
            parallel=False,           # simpler, fewer handles
            decode_times=False,
            engine="netcdf4"          # read-only backend
        )
        ds = xr.decode_cf(ds)

        # ---------- rechunk & write ----------
        ds = ds.chunk({"time": -1, "latitude": 171, "longitude": 173})
        encoding = {v: {"zlib": True, "complevel": 0} for v in ds.data_vars}

        tmp_month = Path(tmpdir) / f"{month_tag}.nc"
        ds.to_netcdf(tmp_month, engine="h5netcdf", encoding=encoding)
        ds.close()
        del ds                       # drop last reference

        # ---------- ensure handles are gone ----------
        gc.collect()
        time.sleep(0.1)              # OS breathing room

        # ---------- atomic publish ----------
        _atomic_move(tmp_month, final)

    finally:
        # best-effort cleanup
        shutil.rmtree(tmpdir, ignore_errors=True)

    return final
    
# @retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
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

    unique_dates = check_data_availability(unique_dates, sss_folder, sst_folder, aviso_folder)
    # print(unique_dates, len(unique_dates))
    # if empty, return empty
    if len(unique_dates) == 0:
        return unique_dates, unique_dates, unique_dates
    
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

if __name__ == "__main__":
    # Simple test
    #test load_satellite_data for 2024-10-25
    times = np.array([datetime(2020, 10, 25)])
    lat = np.array([25.0])
    lon = np.array([-83.0])
    sss, sst, ssh = load_satellite_data(times, lat, lon)
    print(sss)
    print(sst)
    # print(ssh)

    # # Download all DUACS files for 2024-2025
    # aviso_folder = "/unity/f1/ozavala/DATA/GOFFISH/AVISO/GoM/"
    # # gets all first day of each month
    # dates = [datetime(year, month, 1) for year in range(2024, 2026) for month in range(1, 13)]
    # for c_date in dates:
    #     print(f"date: {c_date}")
    #     download_aviso(aviso_folder, c_date)
    # print("Done!")

    # ## Download all SMAP SSS files for 2025
    # sss_folder = "/Net/work/ozavala/DATA/GOFFISH/SSS/SMAP_Global/"
    # # # gets all days from 2025
    # dates = [
    #     datetime(year, month, day)
    #     for year in range(2025, 2026)
    #     for month in range(1, 13)
    #     for day in range(1, calendar.monthrange(year, month)[1] + 1)
    # ]
    # for c_date in dates:
    #     print(f"date: {c_date}")
    #     download_sss(sss_folder, c_date)
    # print("Done!")
    
    # ## Download all SST files for 2025
    # sst_folder = "/unity/f1/ozavala/DATA/GOFFISH/SST/OISST"
    # # # gets all days from 2025
    # dates = [
    #     datetime(year, month, day)
    #     for year in range(2025, 2026)
    #     for month in range(1, 13)
    #     for day in range(1, calendar.monthrange(year, month)[1] + 1)
    # ]
    # for c_date in dates:
    #     print(f"date: {c_date}")
    #     download_sst(sst_folder, c_date)
    # print("Done!")