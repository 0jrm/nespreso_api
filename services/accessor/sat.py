# services/accessor/sat.py
from __future__ import annotations
import glob
import numpy as np
import xarray as xr
from datetime import date, datetime, timedelta
from scipy.interpolate import RegularGridInterpolator
from pathlib import Path
import os, time, gc
from os.path import join
import torch
from services.config import CFG
import time
import gc
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.interpolate import RegularGridInterpolator
from collections import defaultdict
import signal

# -----------------------------
# Helpers
# -----------------------------

def timeout_handler(signum, frame):
    """Signal handler for timeout"""
    raise TimeoutError("Operation timed out")

def interpolate_with_fallback(lats, lons, data_arr, target_coords, data_name="data"):
    """
    Interpolate data with fallback to nearest-neighbor values when interpolation returns NaN.
    
    Args:
        lats: Latitude coordinates array
        lons: Longitude coordinates array  
        data_arr: Data array to interpolate
        target_coords: Target coordinates for interpolation (N, 2) array
        data_name: Name of the data for debugging
        
    Returns:
        Interpolated values with NaN values replaced by nearest-neighbor fallbacks
    """
    try:
        # Perform regular interpolation
        f = RegularGridInterpolator((lats, lons), data_arr,
                                    bounds_error=False, fill_value=np.nan)
        interpolated_vals = f(target_coords)
        
        # Check for NaN values that need fallback
        nan_mask = np.isnan(interpolated_vals)
        if np.any(nan_mask):
            nan_count = np.sum(nan_mask)
            total_count = len(interpolated_vals)
            print(f"DEBUG[{data_name}]: {nan_count}/{total_count} interpolated values are NaN, applying fallback")
            
            # For each NaN value, find the nearest valid grid point
            for i in np.where(nan_mask)[0]:
                target_lat, target_lon = target_coords[i]
                
                # Find nearest grid point
                lat_idx = np.argmin(np.abs(lats - target_lat))
                lon_idx = np.argmin(np.abs(lons - target_lon))
                
                # Get nearest neighbor value
                nearest_val = data_arr[lat_idx, lon_idx]
                
                if np.isfinite(nearest_val):
                    interpolated_vals[i] = nearest_val
                    print(f"DEBUG[{data_name}]: Point {i} (lat={target_lat:.3f}, lon={target_lon:.3f}) - fallback to nearest neighbor: {nearest_val:.3f}")
                else:
                    print(f"DEBUG[{data_name}]: Point {i} (lat={target_lat:.3f}, lon={target_lon:.3f}) - nearest neighbor also NaN, keeping NaN")
            
            # Report final statistics
            final_nan_count = np.sum(np.isnan(interpolated_vals))
            print(f"DEBUG[{data_name}]: After fallback: {final_nan_count}/{total_count} values are still NaN")
        
        return interpolated_vals
        
    except Exception as e:
        print(f"DEBUG[{data_name}]: Interpolation failed: {e}")
        # Return NaN array as fallback
        return np.full(len(target_coords), np.nan)

def timeout_decorator(seconds=300):  # 5 minutes default
    """Decorator to add timeout to functions"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Set the signal handler
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                # Restore the old handler and cancel the alarm
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        return wrapper
    return decorator

def get_day_of_year_from_month_and_day(month, day_of_month, year=datetime.now().year) -> int:
    first_jan = date(year, 1, 1)
    return date(year, month, day_of_month).toordinal() - first_jan.toordinal() + 1

def _round_bbox(bbox, ndigits=2):
    """Make bbox hashable & coalesce-close requests for caching."""
    if bbox is None:
        return None
    return tuple(round(float(x), ndigits) for x in bbox)

# -----------------------------
# Readers (open → extract arrays → close)
# -----------------------------

@timeout_decorator(300)  # 5 minutes timeout
def get_aviso_by_date(aviso_folder: str, c_date: datetime, bbox=None):
    """
    Return (lat, lon, adt_2d) for nearest available time to c_date.
    """
    monthly = join(aviso_folder, f"{c_date.year}-{c_date.month:02d}.nc")
    if os.path.isfile(monthly):
        try:
            print(f"DEBUG[aviso]: Loading monthly file: {monthly}")
            ds = xr.open_dataset(monthly, drop_variables=[v for v in []])  # minimal open
        except Exception as e:
            print(f"DEBUG[aviso]: Failed to open monthly file {monthly}: {e}")
            ds = None
    else:
        ds = None
        alt_root = CFG.AVISO_ALT_ROOT
        if not alt_root:
            raise FileNotFoundError(f"AVISO file not found: {monthly}")
        alt_pattern = f"nrt_global_allsat_phy_l4_{c_date.strftime('%Y%m%d')}"
        try:
            files = [f for f in os.listdir(alt_root) if alt_pattern in f]
        except FileNotFoundError:
            files = []
        if not files:
            raise FileNotFoundError(f"No AVISO files for {c_date:%Y-%m-%d} under {alt_root}")
        try:
            print(f"DEBUG[aviso]: Loading alt file: {join(alt_root, files[0])}")
            ds = xr.open_dataset(join(alt_root, files[0]), drop_variables=[v for v in []])
        except Exception as e:
            print(f"DEBUG[aviso]: Failed to open alt file: {e}")
            ds = None

    if ds is None:
        raise FileNotFoundError(f"AVISO data unavailable for {c_date:%Y-%m-%d}")

    try:
        target_time = np.datetime64(c_date)
        # nearest time index (dataset can be daily or multi-time monthly)
        tcoord = ds["time"].values
        idx = int(np.argmin(np.abs(tcoord - target_time)).item()) if tcoord.ndim == 1 else 0
        sub = ds.isel(time=idx)

        if bbox is not None:
            sub = sub.sel(latitude=slice(bbox[0], bbox[1]),
                          longitude=slice(bbox[2], bbox[3]))

        lats = sub.latitude.values
        lons = sub.longitude.values
        
        # Add timeout protection for data access
        try:
            adt = np.asarray(sub.adt.values, dtype=np.float32)  # 2D (lat, lon)
            print(f"DEBUG[aviso]: Successfully loaded ADT data, shape: {adt.shape}")
        except Exception as e:
            print(f"DEBUG[aviso]: Failed to access ADT values: {e}")
            # Try alternative approach
            try:
                adt = sub.adt.to_numpy().astype(np.float32)
                print(f"DEBUG[aviso]: Successfully loaded ADT data via to_numpy(), shape: {adt.shape}")
            except Exception as e2:
                print(f"DEBUG[aviso]: Alternative ADT loading also failed: {e2}")
                raise e2
        
        return lats, lons, adt
    finally:
        try: ds.close()
        except Exception: pass

@timeout_decorator(300)  # 5 minutes timeout
def get_sst_ghrsst_by_date(sst_folder: str, c_date: datetime, bbox=None):
    """
    Return (lat, lon, analysed_sst_2d[K]) for c_date (MUR daily).
    """
    ctag = c_date.strftime("%Y%m%d")
    pattern = join(sst_folder, str(c_date.year),
                   f"{ctag}090000-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1*.nc")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No SST file found matching pattern: {pattern}")
    
    # Prefer files with "_subset" suffix, fall back to others
    subset_files = [f for f in files if "_subset" in f]
    if subset_files:
        fname = subset_files[0]
    else:
        fname = files[0]
    
    print(f"DEBUG[sst]: Using SST file: {fname}")
    
    ds = xr.open_dataset(fname, decode_timedelta=False,
                         drop_variables=[v for v in []])
    try:
        sub = ds
        if bbox is not None:
            sub = sub.sel(lat=slice(bbox[0], bbox[1]), lon=slice(bbox[2], bbox[3]))
        lats = sub.lat.values
        lons = sub.lon.values
        # Some files have time dim of length 1 → squeeze
        arr = sub["analysed_sst"].values
        if arr.ndim == 3:
            arr = arr[0]
        sst = np.asarray(arr, dtype=np.float32)  # Kelvin
        return lats, lons, sst
    finally:
        try: ds.close()
        except Exception: pass

@timeout_decorator(300)  # 5 minutes timeout
def get_sss_by_date(sss_folder: str, c_date: datetime, bbox=None):
    """
    Return (lat, lon[-180..180], sss_smap_2d).
    """
    doy = get_day_of_year_from_month_and_day(c_date.month, c_date.day, year=c_date.year)
    candidates = [
        join(sss_folder, str(c_date.year), f"RSS_smap_SSS_L3_8day_running_{c_date.year}_{doy:03d}_FNL_v06.0.nc"),
        join(sss_folder, str(c_date.year), f"RSS_smap_SSS_L3_8day_running_{c_date.year}_{doy:03d}_FNL_v05.0.nc"),
    ]
    last_err = None
    ds = None
    
    for path in candidates:
        try:
            print(f"DEBUG[sss]: Trying SSS file: {path}")
            ds = xr.open_dataset(path, drop_variables=[v for v in []])
            print(f"DEBUG[sss]: Successfully opened SSS file: {path}")
            break
        except Exception as e:
            print(f"DEBUG[sss]: Failed to open SSS file {path}: {e}")
            last_err = e
            ds = None
    
    if ds is None:
        # Try wildcard pattern as fallback
        pattern = join(sss_folder, str(c_date.year), f"RSS_smap_SSS_L3_8day_running_{c_date.year}_{doy:03d}_FNL_v*.nc")
        wildcard_files = glob.glob(pattern)
        if wildcard_files:
            try:
                print(f"DEBUG[sss]: Trying wildcard pattern: {pattern}")
                print(f"DEBUG[sss]: Found files: {wildcard_files}")
                ds = xr.open_dataset(wildcard_files[0], drop_variables=[v for v in []])
                print(f"DEBUG[sss]: Successfully opened SSS file via wildcard: {wildcard_files[0]}")
            except Exception as e:
                print(f"DEBUG[sss]: Failed to open SSS file via wildcard: {e}")
                last_err = e
                ds = None
    
    if ds is None:
        raise FileNotFoundError(f"SSS not found for {c_date:%Y-%m-%d}: {last_err}")

    try:
        sub = ds
        if bbox is not None:
            sub = sub.sel(
                lat=slice(bbox[0], bbox[1]),
                lon=slice((bbox[2] + 360) % 360, (bbox[3] + 360) % 360),
            )
        lats = sub.lat.values
        lons = sub.lon.values
        # Normalize to [-180,180] for consistency with inputs
        lons = np.where(lons > 180, lons - 360, lons).astype(np.float32)
        sss = np.asarray(sub["sss_smap"].values, dtype=np.float32)  # 2D
        print(f"DEBUG[sss]: SSS data loaded successfully, shape: {sss.shape}, range: {np.nanmin(sss):.3f} to {np.nanmax(sss):.3f}")
        return lats.astype(np.float32), lons, sss
    finally:
        try: ds.close()
        except Exception: pass

def check_data_availability(dates: list[datetime], sss_root: str, sst_root: str, aviso_root: str) -> list[datetime]:
    def sss_exists(d): 
        ydir = os.path.join(sss_root, f"{d.year:04d}")
        doy  = d.timetuple().tm_yday
        return os.path.isfile(os.path.join(ydir, f"RSS_smap_SSS_L3_8day_running_{d.year}_{doy:03d}_FNL_v06.0.nc"))
    def sst_exists(d):
        ydir = os.path.join(sst_root, f"{d.year:04d}")
        pattern = os.path.join(ydir, f"{d.strftime('%Y%m%d')}090000-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1*.nc")
        files = glob.glob(pattern)
        return len(files) > 0
    def aviso_exists(d):
        return os.path.isfile(os.path.join(aviso_root, f"{d.year}-{d.month:02d}.nc"))
    return [d for d in dates if sss_exists(d) and sst_exists(d) and aviso_exists(d)]

def _lru_cache_ds(maxsize=128):
    """LRU cache for xarray Datasets, with .close() on eviction."""
    def decorator(func):
        cache = lru_cache(maxsize=maxsize)(func)
        def wrapper(*args, **kwargs):
            return cache(*args, **kwargs)
        wrapper.cache_clear = cache.cache_clear
        return wrapper
    return decorator

# -----------------------------
# Array-level caches (NO xarray in cache)
# -----------------------------

@lru_cache(maxsize=256)
def cached_sss_arrays(sss_folder: str, y: int, m: int, d: int, bbox_rounded):
    c_date = datetime(y, m, d)
    return get_sss_by_date(sss_folder, c_date, bbox_rounded)

@lru_cache(maxsize=256)
def cached_sst_arrays(sst_folder: str, y: int, m: int, d: int, bbox_rounded):
    c_date = datetime(y, m, d)
    return get_sst_ghrsst_by_date(sst_folder, c_date, bbox_rounded)

@lru_cache(maxsize=256)
def cached_aviso_arrays(aviso_folder: str, y: int, m: int, d: int, bbox_rounded):
    c_date = datetime(y, m, d)
    return get_aviso_by_date(aviso_folder, c_date, bbox_rounded)


# -----------------------------
# Main accessor
# -----------------------------

def load_satellite_data(times, lat, lon):
    """
    Fast path:
      - If len(times) == len(lat) == len(lon) == N > 1 → one-to-one mode.
        Group by date; per date, open each field once, build 1 interpolator, eval all coords.
      - Else → legacy cross-product (T x N) retained.

    Returns SSS, SST, SSH as:
      - (N,) in one-to-one mode
      - (T, N) in legacy mode
    """
    t0 = time.time()
    times = list(times)
    lat = np.asarray(lat, dtype=np.float32)
    lon = np.asarray(lon, dtype=np.float32)

    num_times, num_locations = len(times), len(lat)
    print(f"DEBUG[sat]: load_satellite_data: T={num_times}, N={num_locations}")

    # ------------- One-to-one mode -------------
    if num_times == num_locations and num_times > 1:
        N = num_times
        sss = np.full(N, np.nan, dtype=np.float32)
        sst = np.full(N, np.nan, dtype=np.float32)
        ssh = np.full(N, np.nan, dtype=np.float32)

        # Group indices by (year, month, day) to reuse the same opens
        groups = defaultdict(list)
        for i, t in enumerate(times):
            dt = datetime(t.year, t.month, t.day)
            groups[(dt.year, dt.month, dt.day)].append(i)

        # Process each date once
        for (y, m, d), idxs in groups.items():
            pts_lat = lat[idxs]
            pts_lon = lon[idxs]

            # Build a per-date bbox covering all points (with padding)
            min_lat = float(np.min(pts_lat)) - CFG.BBOX_PADDING_DEG
            max_lat = float(np.max(pts_lat)) + CFG.BBOX_PADDING_DEG
            min_lon = float(np.min(pts_lon)) - CFG.BBOX_PADDING_DEG
            max_lon = float(np.max(pts_lon)) + CFG.BBOX_PADDING_DEG
            bbox = (min_lat, max_lat, min_lon, max_lon)
            bbox_r = _round_bbox(bbox)
            
            print(f"DEBUG[sat]: Processing date {y}-{m:02d}-{d:02d} with {len(idxs)} points")
            print(f"DEBUG[sat]: Bounding box: lat[{min_lat:.3f}, {max_lat:.3f}], lon[{min_lon:.3f}, {max_lon:.3f}]")
            print(f"DEBUG[sat]: Points lat range: {np.min(pts_lat):.3f} to {np.max(pts_lat):.3f}")
            print(f"DEBUG[sat]: Points lon range: {np.min(pts_lon):.3f} to {np.max(pts_lon):.3f}")

            # Fetch arrays (from cache or disk)
            try:
                sss_lats, sss_lons, sss_arr = cached_sss_arrays(CFG.SSS_ROOT, y, m, d, bbox_r)
                print(f"DEBUG[sat]: SSS data loaded for {y}-{m:02d}-{d:02d}, shape: {sss_arr.shape if sss_arr is not None else 'None'}")
                if sss_arr is not None:
                    print(f"DEBUG[sat]: SSS data within bbox - valid pixels: {np.sum(np.isfinite(sss_arr))}/{sss_arr.size}")
                    print(f"DEBUG[sat]: SSS data within bbox - range: {np.nanmin(sss_arr):.3f} to {np.nanmax(sss_arr):.3f}")
                    # Get the filename from the cache key or function call
                    print(f"DEBUG[sat]: SSS data source: CFG.SSS_ROOT={CFG.SSS_ROOT}, year={y}, month={m:02d}, day={d:02d}")
            except TimeoutError as e:
                print(f"DEBUG[sat]: SSS data loading timed out for {y}-{m:02d}-{d:02d}: {e}")
                sss_lats = sss_lons = sss_arr = None
            except Exception as e:
                print(f"DEBUG[sat]: SSS data failed for {y}-{m:02d}-{d:02d}: {e}")
                sss_lats = sss_lons = sss_arr = None
            try:
                sst_lats, sst_lons, sst_arr = cached_sst_arrays(CFG.SST_ROOT, y, m, d, bbox_r)
                print(f"DEBUG[sat]: SST data loaded for {y}-{m:02d}-{d:02d}, shape: {sst_arr.shape if sst_arr is not None else 'None'}")
                if sst_arr is not None:
                    print(f"DEBUG[sat]: SST data within bbox - valid pixels: {np.sum(np.isfinite(sst_arr))}/{sst_arr.size}")
                    print(f"DEBUG[sat]: SST data within bbox - range: {np.nanmin(sst_arr):.3f} to {np.nanmax(sst_arr):.3f}")
                    # Get the filename from the cache key or function call
                    print(f"DEBUG[sat]: SST data source: CFG.SST_ROOT={CFG.SST_ROOT}, year={y}, month={m:02d}, day={d:02d}")
            except TimeoutError as e:
                print(f"DEBUG[sat]: SST data loading timed out for {y}-{m:02d}-{d:02d}: {e}")
                sst_lats = sst_lons = sst_arr = None
            except Exception as e:
                print(f"DEBUG[sat]: SST data failed for {y}-{m:02d}-{d:02d}: {e}")
                sst_lats = sst_lons = sst_arr = None
            try:
                ssh_lats, ssh_lons, ssh_arr = cached_aviso_arrays(CFG.AVISO_ROOT, y, m, d, bbox_r)
                print(f"DEBUG[sat]: SSH data loaded for {y}-{m:02d}-{d:02d}, shape: {ssh_arr.shape if ssh_arr is not None else 'None'}")
                if ssh_arr is not None:
                    print(f"DEBUG[sat]: SSH data within bbox - valid pixels: {np.sum(np.isfinite(ssh_arr))}/{ssh_arr.size}")
                    print(f"DEBUG[sat]: SSH data within bbox - range: {np.nanmin(ssh_arr):.3f} to {np.nanmax(ssh_arr):.3f}")
                    # Get the filename from the cache key or function call
                    print(f"DEBUG[sat]: SSH data source: CFG.AVISO_ROOT={CFG.AVISO_ROOT}, year={y}, month={m:02d}, day={d:02d}")
            except TimeoutError as e:
                print(f"DEBUG[sat]: SSH data loading timed out for {y}-{m:02d}-{d:02d}: {e}")
                ssh_lats = ssh_lons = ssh_arr = None
            except Exception as e:
                print(f"DEBUG[sat]: SSH data failed for {y}-{m:02d}-{d:02d}: {e}")
                ssh_lats = ssh_lons = ssh_arr = None

            # Vectorized coords for this date
            coords = np.column_stack((pts_lat, pts_lon)).astype(np.float32)
            
            # Check if coordinates are within data bounds for each satellite product
            if sss_arr is not None:
                lat_in_bounds = (pts_lat >= np.min(sss_lats)) & (pts_lat <= np.max(sss_lats))
                lon_in_bounds = (pts_lon >= np.min(sss_lons)) & (pts_lon <= np.max(sss_lons))
                coords_in_bounds = lat_in_bounds & lon_in_bounds
                print(f"DEBUG[sat]: SSS bounds check - {np.sum(coords_in_bounds)}/{len(idxs)} coordinates within data bounds")
                if not np.any(coords_in_bounds):
                    print(f"DEBUG[sat]: WARNING - No coordinates within SSS data bounds for date {y}-{m:02d}-{d:02d}")
                    print(f"DEBUG[sat]: Target lat range: {np.min(pts_lat):.3f} to {np.max(pts_lat):.3f}")
                    print(f"DEBUG[sat]: Target lon range: {np.min(pts_lon):.3f} to {np.max(pts_lon):.3f}")
                    print(f"DEBUG[sat]: SSS data lat range: {np.min(sss_lats):.3f} to {np.max(sss_lats):.3f}")
                    print(f"DEBUG[sat]: SSS data lon range: {np.min(sss_lons):.3f} to {np.max(sss_lons):.3f}")
            
            if sst_arr is not None:
                lat_in_bounds = (pts_lat >= np.min(sst_lats)) & (pts_lat <= np.max(sst_lats))
                lon_in_bounds = (pts_lon >= np.min(sst_lons)) & (pts_lon <= np.max(sst_lons))
                coords_in_bounds = lat_in_bounds & lon_in_bounds
                print(f"DEBUG[sat]: SST bounds check - {np.sum(coords_in_bounds)}/{len(idxs)} coordinates within data bounds")
                if not np.any(coords_in_bounds):
                    print(f"DEBUG[sat]: WARNING - No coordinates within SST data bounds for date {y}-{m:02d}-{d:02d}")
            
            if ssh_arr is not None:
                lat_in_bounds = (pts_lat >= np.min(ssh_lats)) & (pts_lat <= np.max(ssh_lats))
                lon_in_bounds = (pts_lon >= np.min(ssh_lons)) & (pts_lon <= np.max(ssh_lons))
                coords_in_bounds = lat_in_bounds & lon_in_bounds
                print(f"DEBUG[sat]: SSH bounds check - {np.sum(coords_in_bounds)}/{len(idxs)} coordinates within data bounds")
                if not np.any(coords_in_bounds):
                    print(f"DEBUG[sat]: WARNING - No coordinates within SSH data bounds for date {y}-{m:02d}-{d:02d}")

            # Interpolate SSS
            if sss_arr is not None:
                try:
                    print(f"DEBUG[sss]: Interpolating SSS for {len(idxs)} points")
                    print(f"DEBUG[sss]: SSS array shape: {sss_arr.shape}, range: {np.nanmin(sss_arr):.3f} to {np.nanmax(sss_arr):.3f}")
                    print(f"DEBUG[sss]: SSS lats range: {np.min(sss_lats):.3f} to {np.max(sss_lats):.3f}")
                    print(f"DEBUG[sss]: SSS lons range: {np.min(sss_lons):.3f} to {np.max(sss_lons):.3f}")
                    print(f"DEBUG[sss]: Target coords range - lat: {np.min(pts_lat):.3f} to {np.max(pts_lat):.3f}")
                    print(f"DEBUG[sss]: Target coords range - lon: {np.min(pts_lon):.3f} to {np.max(pts_lon):.3f}")
                    
                    # Use fallback interpolation
                    interpolated_vals = interpolate_with_fallback(sss_lats, sss_lons, sss_arr, coords, "sss")
                    print(f"DEBUG[sss]: Interpolation completed, result shape: {interpolated_vals.shape}")
                    
                    # Handle all-NaN case gracefully
                    if np.all(np.isnan(interpolated_vals)):
                        print(f"DEBUG[sss]: WARNING - All interpolated values are NaN for date {y}-{m:02d}-{d:02d}")
                        print(f"DEBUG[sss]: This suggests the interpolation failed completely - check if target coordinates are within data bounds")
                        print(f"DEBUG[sss]: Target coords: {coords}")
                        print(f"DEBUG[sss]: Data bounds: lat[{np.min(sss_lats):.3f}, {np.max(sss_lats):.3f}], lon[{np.min(sss_lons):.3f}, {np.max(sss_lons):.3f}]")
                    else:
                        print(f"DEBUG[sss]: Interpolated values range: {np.nanmin(interpolated_vals):.3f} to {np.nanmax(interpolated_vals):.3f}")
                        print(f"DEBUG[sss]: NaN count in interpolated values: {np.sum(np.isnan(interpolated_vals))}")
                    
                    sss[idxs] = interpolated_vals.astype(np.float32)
                    print(f"DEBUG[sss]: SSS interpolation successful for {len(idxs)} points")
                except Exception as e:
                    print(f"DEBUG[sss]: SSS interpolation failed for {y}-{m:02d}-{d:02d}: {e}")
                    # Keep NaN values for failed interpolation
                    pass

            # Interpolate SST (Kelvin sanity check later)
            if sst_arr is not None:
                try:
                    # Use fallback interpolation
                    interpolated_vals = interpolate_with_fallback(sst_lats, sst_lons, sst_arr, coords, "sst")
                    
                    # Handle all-NaN case gracefully
                    if np.all(np.isnan(interpolated_vals)):
                        print(f"DEBUG[sst]: WARNING - All interpolated values are NaN for date {y}-{m:02d}-{d:02d}")
                        print(f"DEBUG[sst]: This suggests the interpolation failed completely - check if target coordinates are within data bounds")
                        print(f"DEBUG[sst]: Target coords: {coords}")
                        print(f"DEBUG[sst]: Data bounds: lat[{np.min(sst_lats):.3f}, {np.max(sst_lats):.3f}], lon[{np.min(sst_lons):.3f}, {np.max(sst_lons):.3f}]")
                    else:
                        print(f"DEBUG[sst]: Interpolated values range: {np.nanmin(interpolated_vals):.3f} to {np.nanmax(interpolated_vals):.3f}")
                    
                    vals = interpolated_vals.astype(np.float32)
                    bad = (vals < 0) | (vals > 350)
                    if np.any(bad): vals[bad] = np.nan
                    sst[idxs] = vals
                    print(f"DEBUG[sst]: SST interpolation successful for {len(idxs)} points")
                except Exception as e:
                    print(f"DEBUG[sst]: SST interpolation failed for {y}-{m:02d}-{d:02d}: {e}")
                    # Keep NaN values for failed interpolation
                    pass

            # Interpolate SSH with regional mean removal
            if ssh_arr is not None:
                try:
                    # Compute a local mean (excluding exclusion box) on this cropped array
                    LON, LAT = np.meshgrid(ssh_lons, ssh_lats)
                    inclusion = (LAT >= bbox[0]) & (LAT <= bbox[1]) & (LON >= bbox[2]) & (LON <= bbox[3])
                    exclusion = (LAT < CFG.EXCLUSION_LAT) & (LON > CFG.EXCLUSION_LON)
                    region = ssh_arr[inclusion & (~exclusion)]
                    avg = np.nanmean(region) if region.size else np.nan

                    # Use fallback interpolation
                    interpolated_vals = interpolate_with_fallback(ssh_lats, ssh_lons, ssh_arr, coords, "ssh")
                    
                    # Handle all-NaN case gracefully
                    if np.all(np.isnan(interpolated_vals)):
                        print(f"DEBUG[ssh]: WARNING - All interpolated values are NaN for date {y}-{m:02d}-{d:02d}")
                        print(f"DEBUG[ssh]: This suggests the interpolation failed completely - check if target coordinates are within data bounds")
                        print(f"DEBUG[ssh]: Target coords: {coords}")
                        print(f"DEBUG[ssh]: Data bounds: lat[{np.min(ssh_lats):.3f}, {np.max(ssh_lats):.3f}], lon[{np.min(ssh_lons):.3f}, {np.max(ssh_lons):.3f}]")
                    else:
                        print(f"DEBUG[ssh]: Interpolated values range: {np.nanmin(interpolated_vals):.3f} to {np.nanmax(interpolated_vals):.3f}")
                    
                    vals = interpolated_vals.astype(np.float32)
                    if np.isfinite(avg):
                        vals -= np.float32(avg)
                    ssh[idxs] = vals
                    print(f"DEBUG[ssh]: SSH interpolation successful for {len(idxs)} points")
                except Exception as e:
                    print(f"DEBUG[ssh]: SSH interpolation failed for {y}-{m:02d}-{d:02d}: {e}")
                    # Keep NaN values for failed interpolation
                    pass

            gc.collect()

        dt = time.time() - t0
        print(f"DEBUG[sat]: one-to-one done in {dt:.2f}s  (N={N})")
        
        # Validate that we have at least some data
        valid_sss = np.sum(np.isfinite(sss))
        valid_sst = np.sum(np.isfinite(sst))
        valid_ssh = np.sum(np.isfinite(ssh))
        print(f"DEBUG[sat]: Valid data counts - SSS: {valid_sss}/{N}, SST: {valid_sst}/{N}, SSH: {valid_ssh}/{N}")
        
        # If we have no valid data, use climatological fallbacks
        if valid_sss == 0 and valid_sst == 0 and valid_ssh == 0:
            print("WARNING: No valid satellite data found, using climatological fallbacks")
            # Use reasonable climatological values for the Gulf of Mexico region
            sss[:] = 36.0  # Typical Gulf of Mexico salinity
            sst[:] = 25.0  # Typical Gulf of Mexico temperature (Celsius)
            ssh[:] = 0.0   # SSH anomaly (centered around 0)
            print("DEBUG[sat]: Applied climatological fallbacks - SSS: 36.0, SST: 25.0, SSH: 0.0")
        elif valid_sss == 0:
            print("WARNING: No valid SSS data, using climatological fallback")
            sss[:] = 36.0
        elif valid_sst == 0:
            print("WARNING: No valid SST data, using climatological fallback")
            sst[:] = 25.0
        elif valid_ssh == 0:
            print("WARNING: No valid SSH data, using climatological fallback")
            ssh[:] = 0.0
        
        return sss, sst, ssh

    # ------------- Legacy cross-product mode -------------
    T, N = num_times, num_locations
    sss = np.full((T, N), np.nan, dtype=np.float32)
    sst = np.full((T, N), np.nan, dtype=np.float32)
    ssh = np.full((T, N), np.nan, dtype=np.float32)

    min_lat, max_lat = float(np.min(lat)) - CFG.BBOX_PADDING_DEG, float(np.max(lat)) + CFG.BBOX_PADDING_DEG
    min_lon, max_lon = float(np.min(lon)) - CFG.BBOX_PADDING_DEG, float(np.max(lon)) + CFG.BBOX_PADDING_DEG
    bbox = (min_lat, max_lat, min_lon, max_lon)
    bbox_r = _round_bbox(bbox)
    coords = np.column_stack((lat, lon)).astype(np.float32)

    for ti, t in enumerate(times):
        y, m, d = t.year, t.month, t.day
        try:
            sss_lats, sss_lons, sss_arr = cached_sss_arrays(CFG.SSS_ROOT, y, m, d, bbox_r)
        except Exception:
            sss_arr = None
        try:
            sst_lats, sst_lons, sst_arr = cached_sst_arrays(CFG.SST_ROOT, y, m, d, bbox_r)
        except Exception:
            sst_arr = None
        try:
            ssh_lats, ssh_lons, ssh_arr = cached_aviso_arrays(CFG.AVISO_ROOT, y, m, d, bbox_r)
        except Exception:
            ssh_arr = None

        if sss_arr is not None:
            try:
                # Use fallback interpolation
                interpolated_vals = interpolate_with_fallback(sss_lats, sss_lons, sss_arr, coords, "sss")
                sss[ti] = interpolated_vals.astype(np.float32)
            except Exception:
                pass

        if sst_arr is not None:
            try:
                # Use fallback interpolation
                interpolated_vals = interpolate_with_fallback(sst_lats, sst_lons, sst_arr, coords, "sst")
                vals = interpolated_vals.astype(np.float32)
                bad = (vals < 0) | (vals > 350)
                if np.any(bad): vals[bad] = np.nan
                sst[ti] = vals
            except Exception:
                pass

        if ssh_arr is not None:
            try:
                LON, LAT = np.meshgrid(ssh_lons, ssh_lats)
                inclusion = (LAT >= min_lat) & (LAT <= max_lat) & (LON >= min_lon) & (LON <= max_lon)
                exclusion = (LAT < CFG.EXCLUSION_LAT) & (LON > CFG.EXCLUSION_LON)
                region = ssh_arr[inclusion & (~exclusion)]
                avg = np.nanmean(region) if region.size else np.nan

                # Use fallback interpolation
                interpolated_vals = interpolate_with_fallback(ssh_lats, ssh_lons, ssh_arr, coords, "ssh")
                vals = interpolated_vals.astype(np.float32)
                if np.isfinite(avg):
                    vals -= np.float32(avg)
                ssh[ti] = vals
            except Exception:
                pass

        gc.collect()

    dt = time.time() - t0
    print(f"DEBUG[sat]: cross-product done in {dt:.2f}s  (T={T}, N={N})")
    
    # Validate that we have at least some data
    valid_sss = np.sum(np.isfinite(sss))
    valid_sst = np.sum(np.isfinite(sst))
    valid_ssh = np.sum(np.isfinite(ssh))
    print(f"DEBUG[sat]: Valid data counts - SSS: {valid_sss}/{T*N}, SST: {valid_sst}/{T*N}, SSH: {valid_ssh}/{T*N}")
    
    # If we have no valid data, use climatological fallbacks
    if valid_sss == 0 and valid_sst == 0 and valid_ssh == 0:
        print("WARNING: No valid satellite data found, using climatological fallbacks")
        # Use reasonable climatological values for the Gulf of Mexico region
        sss[:] = 36.0  # Typical Gulf of Mexico salinity
        sst[:] = 25.0  # Typical Gulf of Mexico temperature (Celsius)
        ssh[:] = 0.0   # SSH anomaly (centered around 0)
        print("DEBUG[sat]: Applied climatological fallbacks - SSS: 36.0, SST: 25.0, SSH: 0.0")
    elif valid_sss == 0:
        print("WARNING: No valid SSS data, using climatological fallback")
        sss[:] = 36.0
    elif valid_sst == 0:
        print("WARNING: No valid SST data, using climatological fallback")
        sst[:] = 25.0
    elif valid_ssh == 0:
        print("WARNING: No valid SSH data, using climatological fallback")
        ssh[:] = 0.0
    
    return sss, sst, ssh

def prepare_inputs(time, lat, lon, sss, sst, ssh, input_params: dict):
    """
    Build input tensor for model, one row per (time, lat, lon) tuple.
    If len(time) == len(lat) > 1, iterate in parallel (one-to-one).
    Otherwise, use cross-product (T*N).
    Filters out locations with NaN satellite data before building inputs.
    """

    time = np.asarray(time, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    sss = np.asarray(sss, dtype=np.float64)
    sst = np.asarray(sst, dtype=np.float64)
    ssh = np.asarray(ssh, dtype=np.float64)

    T = time.shape[0]
    N = lat.shape[0]
    
    # Check satellite data validity (filtering is now done in API layer)
    print(f"DEBUG[prepare_inputs]: Checking satellite data validity...")
    print(f"DEBUG[prepare_inputs]: Current data shapes - time: {time.shape}, lat: {lat.shape}, lon: {lon.shape}")
    print(f"DEBUG[prepare_inputs]: Current satellite data shapes - sss: {sss.shape}, sst: {sst.shape}, ssh: {ssh.shape}")
    
    # Quick validation that satellite data dimensions match coordinate dimensions
    if input_params.get("sat"):
        # Verify that satellite data arrays have consistent dimensions
        if sss.ndim == 1 and sst.ndim == 1 and ssh.ndim == 1:
            if len(sss) != len(lat) or len(sst) != len(lat) or len(ssh) != len(lat):
                print(f"WARNING[prepare_inputs]: Satellite data length mismatch with coordinates")
                print(f"  SSS: {len(sss)}, SST: {len(sst)}, SSH: {len(ssh)}, Coordinates: {len(lat)}")
        elif sss.ndim == 2 and sst.ndim == 2 and ssh.ndim == 2:
            if sss.shape[1] != len(lat) or sst.shape[1] != len(lat) or ssh.shape[1] != len(lat):
                print(f"WARNING[prepare_inputs]: Satellite data width mismatch with coordinates")
                print(f"  SSS: {sss.shape[1]}, SST: {sst.shape[1]}, SSH: {ssh.shape[1]}, Coordinates: {len(lat)}")
    
    print(f"DEBUG[prepare_inputs]: Final dataset size: {T} time points × {N} locations = {T*N} total samples")

    # Helper to compute features for a single sample
    def compute_features(t, la, lo, sss_val, sst_val, ssh_val):
        features = []
        t_mod = t % 365.0
        la_rad = np.deg2rad(la)
        lo_rad = np.deg2rad(lo)
        if input_params.get("timecos"):
            features.append(np.cos(2 * np.pi * (t_mod / 365.0)))
        if input_params.get("timesin"):
            features.append(np.sin(2 * np.pi * (t_mod / 365.0)))
        if input_params.get("latcos"):
            features.append(np.cos(2 * np.pi * (la_rad / np.pi)))
        if input_params.get("latsin"):
            features.append(np.sin(2 * np.pi * (la_rad / np.pi)))
        if input_params.get("loncos"):
            features.append(np.cos(2 * np.pi * (lo_rad / (2 * np.pi))))
        if input_params.get("lonsin"):
            features.append(np.sin(2 * np.pi * (lo_rad / (2 * np.pi))))
        if input_params.get("sat"):
            if input_params.get("sss"):
                features.append(sss_val)
            if input_params.get("sst"):
                features.append(sst_val - CFG.KELVIN_OFFSET)
            if input_params.get("ssh"):
                features.append(ssh_val)
        return features

    # Case 1: One-to-one correspondence
    if N > 1 and T == N:
        # Allow either (T, N) or (N,) shapes for one-to-one mode
        valid_shape = (sss.shape == (T, N) and sst.shape == (T, N) and ssh.shape == (T, N)) or \
                     (sss.shape == (N,) and sst.shape == (N,) and ssh.shape == (N,))
        assert valid_shape, f"Accessor arrays must be (T, N)={(T, N)} or (N,)={(N,)}, got sss={sss.shape}, sst={sst.shape}, ssh={ssh.shape}"
        inputs = []
        for i in range(N):
            # Each i corresponds to (time[i], lat[i], lon[i])
            sss_val = sss[i, i] if sss.ndim == 2 else sss[i]
            sst_val = sst[i, i] if sst.ndim == 2 else sst[i]
            ssh_val = ssh[i, i] if ssh.ndim == 2 else ssh[i]
            feats = compute_features(time[i], lat[i], lon[i], sss_val, sst_val, ssh_val)
            feats = [0.0 if (isinstance(f, float) and not np.isfinite(f)) else f for f in feats]
            inputs.append(torch.tensor(feats, dtype=torch.float32))
        X = torch.stack(inputs, dim=0)
        print(f"DEBUG[prepare_inputs]: Returning input tensor with shape: {X.shape}")
        return X

    # Case 2: Cross-product (T*N)
    elif N > 1:
        assert sss.shape == (T, N) and sst.shape == (T, N) and ssh.shape == (T, N), "Accessor arrays must be (T, N)"
        inputs = []
        for t_idx in range(T):
            for n_idx in range(N):
                sss_val = sss[t_idx, n_idx]
                sst_val = sst[t_idx, n_idx]
                ssh_val = ssh[t_idx, n_idx]
                feats = compute_features(time[t_idx], lat[n_idx], lon[n_idx], sss_val, sst_val, ssh_val)
                feats = [0.0 if (isinstance(f, float) and not np.isfinite(f)) else f for f in feats]
                inputs.append(torch.tensor(feats, dtype=torch.float32))
        X = torch.stack(inputs, dim=0)
        print(f"DEBUG[prepare_inputs]: Returning input tensor with shape: {X.shape}")
        return X

    # Case 3: Single location (N == 1)
    else:
        assert sss.shape == (T, 1) and sst.shape == (T, 1) and ssh.shape == (T, 1), "Accessor arrays must be (T, 1)"
        inputs = []
        for t_idx in range(T):
            sss_val = sss[t_idx, 0]
            sst_val = sst[t_idx, 0]
            ssh_val = ssh[t_idx, 0]
            feats = compute_features(time[t_idx], lat[0], lon[0], sss_val, sst_val, ssh_val)
            feats = [0.0 if (isinstance(f, float) and not np.isfinite(f)) else f for f in feats]
            inputs.append(torch.tensor(feats, dtype=torch.float32))
        X = torch.stack(inputs, dim=0)
        print(f"DEBUG[prepare_inputs]: Returning input tensor with shape: {X.shape}")
        return X
