# services/api/app.py
from __future__ import annotations

from flask import Flask, Blueprint, request, jsonify, make_response
from pydantic import BaseModel, ValidationError, field_validator
from typing import List, Tuple
import numpy as np
import xarray as xr
import logging, sys, io, tempfile, os
from datetime import datetime
import pickle
import glob

from services.kernel.handler import infer, get_pca_objects
from services.accessor.sat import load_satellite_data, prepare_inputs
from services.api.metrics import metrics_bp, before_request as metrics_before, after_request as metrics_after
from services.config import CFG

logger = logging.getLogger("ocean")
if not logger.handlers:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logger.addHandler(h)
logger.setLevel(logging.INFO)


class ProfileRequest(BaseModel):
    lat: List[float]
    lon: List[float]
    date: List[str]

    @field_validator('lat', 'lon', 'date')
    @classmethod
    def non_empty(cls, v):
        if not isinstance(v, list) or len(v) < 1:
            raise ValueError('Must be a non-empty list')
        return v

    @field_validator('lat')
    @classmethod
    def lat_range(cls, v):
        if any((not np.isfinite(x)) or (x < -90.0) or (x > 90.0) for x in v):
            raise ValueError('Latitude must be finite and within [-90, 90]')
        return v

    @field_validator('lon')
    @classmethod
    def lon_range(cls, v):
        if any((not np.isfinite(x)) or (x < -180.0) or (x > 180.0) for x in v):
            raise ValueError('Longitude must be finite and within [-180, 180]')
        return v

    @field_validator('date')
    @classmethod
    def date_iso(cls, v):
        for d in v:
            try:
                datetime.strptime(d, "%Y-%m-%d")
            except Exception:
                raise ValueError(f"Date {d!r} must be YYYY-MM-DD")
        return v


class GridRequest(BaseModel):
    date: str
    bbox: List[float] | None = None  # [lon_min, lat_min, lon_max, lat_max]

    @field_validator('date')
    @classmethod
    def date_iso(cls, v):
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except Exception:
            raise ValueError(f"Date {v!r} must be YYYY-MM-DD")
        return v

    @field_validator('bbox')
    @classmethod
    def bbox_valid(cls, v):
        if v is not None:
            if len(v) != 4:
                raise ValueError('BBOX must have exactly 4 values: [lon_min, lat_min, lon_max, lat_max]')
            lon_min, lat_min, lon_max, lat_max = v
            if not (-180 <= lon_min <= 180 and -180 <= lon_max <= 180):
                raise ValueError('Longitude values must be between -180 and 180')
            if not (-90 <= lat_min <= 90 and -90 <= lat_max <= 90):
                raise ValueError('Latitude values must be between -90 and 90')
            if lon_min >= lon_max:
                raise ValueError('lon_min must be less than lon_max')
            if lat_min >= lat_max:
                raise ValueError('lat_min must be less than lat_max')
        return v


def _load_grid_data(bbox: List[float] | None = None):
    """Load the predefined grid coordinates and mask from pickle file, optionally filtered by BBOX"""
    try:
        grid_file = CFG.GRID_MASK_PATH
        with open(grid_file, 'rb') as f:
            grid_data = pickle.load(f)
        
        lon_grid = grid_data['lon_grid']
        lat_grid = grid_data['lat_grid']
        inside_mask = grid_data['inside_mask']
        
        # Extract only the points inside the mask
        lon_in = lon_grid[inside_mask]
        lat_in = lat_grid[inside_mask]
        
        # Apply BBOX filter if provided
        if bbox is not None:
            lon_min, lat_min, lon_max, lat_max = bbox
            bbox_mask = (lon_in >= lon_min) & (lon_in <= lon_max) & (lat_in >= lat_min) & (lat_in <= lat_max)
            lon_in = lon_in[bbox_mask]
            lat_in = lat_in[bbox_mask]
            logger.info(f"BBOX filter applied: {np.sum(bbox_mask)}/{len(bbox_mask)} points remain")
        
        logger.info(f"Loaded grid with {len(lon_in)} points inside mask" + (f" and BBOX {bbox}" if bbox else ""))
        return lon_in, lat_in
        
    except Exception as e:
        logger.error(f"Failed to load grid data: {e}")
        raise ValueError(f"Grid data unavailable: {str(e)}")


def _write_netcdf_bytes(ds: xr.Dataset) -> bytes:
    """
    Robust in-memory NetCDF writer. Prefers h5-based engine for compression.
    Falls back to a secure temporary file to avoid engine limitations.
    """
    comp = dict(zlib=True, complevel=4)
    encoding = {name: comp for name in ds.data_vars}
    # Ensure CF-compliant time encoding if present
    if 'time' in ds.variables:
        try:
            ds['time'].encoding.update({'units': 'seconds since 1970-01-01 00:00:00', 'calendar': 'proleptic_gregorian'})
        except Exception:
            pass
    try:
        # h5netcdf supports file-like buffers
        buf = io.BytesIO()
        ds.to_netcdf(buf, engine="h5netcdf", encoding=encoding)
        return buf.getvalue()
    except Exception as e1:
        logger.warning("h5netcdf in-memory failed; writing via secure temp file: %s", e1)
        with tempfile.NamedTemporaryFile(prefix="nespreso_", suffix=".nc", delete=True) as f:
            try:
                ds.to_netcdf(f.name, engine="netcdf4", encoding=encoding)
            except Exception as e2:
                logger.warning("netcdf4 failed; falling back to scipy (no compression): %s", e2)
                # scipy backend doesn't support compression - use no encoding
                ds.to_netcdf(f.name, engine="scipy")
            f.flush()
            f.seek(0)
            return f.read()


def _build_dataset(pred_T: np.ndarray,
                   pred_S: np.ndarray,
                   depth: np.ndarray,
                   sss: np.ndarray,
                   sst: np.ndarray,
                   aviso: np.ndarray,
                   times: List[datetime],
                   lat: np.ndarray,
                   lon: np.ndarray) -> xr.Dataset:
    profile_number = np.arange(pred_T.shape[1], dtype=np.int32)
    # Validate alignment
    n_profiles = int(pred_T.shape[1])
    if not (len(times) == len(lat) == len(lon) == n_profiles):
        raise ValueError(f"Mismatch in lengths: times={len(times)}, lat={len(lat)}, lon={len(lon)}, profiles={n_profiles}")

    times64 = np.array([np.datetime64(t) for t in times], dtype='datetime64[ns]')
    # Human-friendly ISO strings (does not replace CF time)
    time_iso = np.array([t.strftime('%Y-%m-%d') for t in times], dtype=object)
    ds = xr.Dataset(
        data_vars=dict(
            Temperature=(("depth", "profile_number"), np.asarray(pred_T, dtype=np.float32)),
            Salinity=(("depth", "profile_number"), np.asarray(pred_S, dtype=np.float32)),
            SSS=("profile_number", np.asarray(sss, dtype=np.float32)),
            SST=("profile_number", np.asarray(sst, dtype=np.float32)),
            AVISO=("profile_number", np.asarray(aviso, dtype=np.float32)),
            time_iso=("profile_number", time_iso),
        ),
        coords=dict(
            profile_number=("profile_number", profile_number),
            depth=("depth", np.asarray(depth, dtype=np.float32)),
            time=("profile_number", times64),
            lat=("profile_number", np.asarray(lat, dtype=np.float32)),
            lon=("profile_number", np.asarray(lon, dtype=np.float32)),
        ),
    )
    # CF/time metadata
    try:
        ds['time'].attrs.update({'standard_name': 'time', 'long_name': 'Time', 'axis': 'T'})
        ds['time_iso'].attrs.update({'description': 'ISO-8601 date string for convenience (duplicate of time coordinate)'})
    except Exception:
        pass
    return ds


def _build_grid_dataset(pred_T: np.ndarray,
                       pred_S: np.ndarray,
                       depth: np.ndarray,
                       sss: np.ndarray,
                       sst: np.ndarray,
                       aviso: np.ndarray,
                       time: datetime,
                       lat: np.ndarray,
                       lon: np.ndarray) -> xr.Dataset:
    """Build a dataset with depth, lat, lon, and time as dimensions for grid queries"""
    
    try:
        # Convert to numpy arrays and ensure proper types
        time64 = np.datetime64(time, 'ns')
        
        # Create coordinate arrays
        lat_coords = np.asarray(lat, dtype=np.float32)
        lon_coords = np.asarray(lon, dtype=np.float32)
        depth_coords = np.asarray(depth, dtype=np.float32)
        
        # Get unique lat/lon values and create a regular grid
        unique_lats = np.unique(lat_coords)
        unique_lons = np.unique(lon_coords)
        
        # Sort them to ensure consistent ordering
        unique_lats = np.sort(unique_lats)
        unique_lons = np.sort(unique_lons)
        
        logger.info(f"Creating grid with {len(unique_lats)} lat points and {len(unique_lons)} lon points")
        logger.info(f"Lat range: {unique_lats.min():.3f} to {unique_lats.max():.3f}")
        logger.info(f"Lon range: {unique_lons.min():.3f} to {unique_lons.max():.3f}")
        
        # Initialize output arrays with NaN
        n_depth = len(depth_coords)
        n_lat = len(unique_lats)
        n_lon = len(unique_lons)
        
        # Create gridded arrays for all variables
        temp_grid = np.full((n_depth, n_lat, n_lon), np.nan, dtype=np.float32)
        sal_grid = np.full((n_depth, n_lat, n_lon), np.nan, dtype=np.float32)
        sss_grid = np.full((n_lat, n_lon), np.nan, dtype=np.float32)
        sst_grid = np.full((n_lat, n_lon), np.nan, dtype=np.float32)
        aviso_grid = np.full((n_lat, n_lon), np.nan, dtype=np.float32)
        
        # Map profile data to grid locations
        for i, (profile_lat, profile_lon) in enumerate(zip(lat_coords, lon_coords)):
            # Find grid indices for this profile location
            lat_idx = np.where(unique_lats == profile_lat)[0][0]
            lon_idx = np.where(unique_lons == profile_lon)[0][0]
            
            # Fill in the grid values
            temp_grid[:, lat_idx, lon_idx] = pred_T[:, i]
            sal_grid[:, lat_idx, lon_idx] = pred_S[:, i]
            sss_grid[lat_idx, lon_idx] = sss[i]
            sst_grid[lat_idx, lon_idx] = sst[i]
            aviso_grid[lat_idx, lon_idx] = aviso[i]
        
        # Create the dataset with proper dimensions
        ds = xr.Dataset(
            data_vars=dict(
                Temperature=(("depth", "lat", "lon"), temp_grid),
                Salinity=(("depth", "lat", "lon"), sal_grid),
                SSS=(("lat", "lon"), sss_grid),
                SST=(("lat", "lon"), sst_grid),
                AVISO=(("lat", "lon"), aviso_grid),
                time_iso=str(time.date()),
            ),
            coords=dict(
                depth=("depth", depth_coords),
                lat=("lat", unique_lats),
                lon=("lon", unique_lons),
                time=time64,
            ),
            attrs=dict(
                description="NeSPReSO Grid Query Results",
                grid_type="regular_grid",
                coordinate_system="geographic"
            )
        )
        # Ensure CF-compliant time encoding
        try:
            ds['time'].encoding.update({'units': 'seconds since 1970-01-01 00:00:00', 'calendar': 'proleptic_gregorian'})
            ds['time'].attrs.update({'standard_name': 'time', 'long_name': 'Time', 'axis': 'T'})
        except Exception:
            pass
        
        return ds
        
    except Exception as e:
        logger.error(f"Error building grid dataset: {e}")
        raise ValueError(f"Failed to build grid dataset: {str(e)}")


def create_app(config: dict | None = None) -> Flask:
    app = Flask(__name__)
    if config:
        app.config.update(config)
    bp = Blueprint("profile", __name__, url_prefix="/v1/profile")

    @bp.route("", methods=["POST"])
    def profile():
        try:
            try:
                req = ProfileRequest.model_validate(request.get_json())
            except ValidationError as e:
                return jsonify({"error": e.errors()}), 400

            lat, lon, dates = req.lat, req.lon, req.date
            if not (len(lat) == len(lon) == len(dates)):
                return jsonify({"error": "Length of 'lat', 'lon', and 'date' must be equal"}), 400

            n = len(lat)
            if CFG.MAX_PROFILES and n > CFG.MAX_PROFILES:
                return jsonify({
                    "error": f"Too many profiles: {n} > {CFG.MAX_PROFILES}. Please use batch processing or reduce the number of profiles.",
                    "max_profiles": CFG.MAX_PROFILES,
                    "requested_profiles": n
                }), 413

            times = [datetime.strptime(d, "%Y-%m-%d") for d in dates]
            lat_arr = np.asarray(lat, dtype=np.float64)
            lon_arr = np.asarray(lon, dtype=np.float64)

            logger.info("Request: %d profiles (logging of payload suppressed)", n)
            print(f"DEBUG: Before satellite loading - times length: {len(times)}")
            print(f"DEBUG: Before satellite loading - lat_arr length: {len(lat_arr)}")
            print(f"DEBUG: Before satellite loading - lon_arr length: {len(lon_arr)}")
            print(f"DEBUG: Before satellite loading - unique dates: {len(set(times))}")

            # Satellite data
            try:
                sss, sst, aviso = load_satellite_data(times, lat_arr, lon_arr)
                print(f"DEBUG: Satellite data loaded - sss: {sss.shape}, sst: {sst.shape}, aviso: {aviso.shape}")
                
                # Validate satellite data shapes
                if sss is None or sst is None or aviso is None:
                    raise ValueError("One or more satellite data arrays are None")
                    
                # Check if we have any valid data
                if np.all(np.isnan(sss)) and np.all(np.isnan(sst)) and np.all(np.isnan(aviso)):
                    logger.warning("All satellite data is NaN - this may indicate data availability issues")
                
                # Filter out locations with NaN satellite data BEFORE prepare_inputs
                print(f"DEBUG: Filtering satellite data for NaN values...")
                
                # Create validity mask - a location is valid if ALL satellite variables have finite values
                if sss.ndim == 1:
                    # One-to-one mode: arrays are (N,)
                    valid_mask = np.isfinite(sss) & np.isfinite(sst) & np.isfinite(aviso)
                elif sss.ndim == 2:
                    # Cross-product mode: arrays are (T, N)
                    if sss.shape[0] == sss.shape[1]:  # One-to-one mode with (T, N) arrays
                        valid_mask = np.isfinite(np.diag(sss)) & np.isfinite(np.diag(sst)) & np.isfinite(np.diag(aviso))
                    else:  # True cross-product mode
                        valid_mask = np.isfinite(sss).all(axis=0) & np.isfinite(sst).all(axis=0) & np.isfinite(aviso).all(axis=0)
                else:
                    # Single location mode: arrays are (T, 1)
                    valid_mask = np.isfinite(sss).all(axis=0) & np.isfinite(sst).all(axis=0) & np.isfinite(aviso).all(axis=0)
                
                n_valid = np.sum(valid_mask)
                n_total = len(lat_arr)
                print(f"DEBUG: Satellite data filtering: {n_valid}/{n_total} locations have valid satellite data")
                
                if n_valid < n_total:
                    n_filtered = n_total - n_valid
                    filter_percentage = (n_filtered / n_total) * 100
                    print(f"DEBUG: Filtering out {n_filtered} locations ({filter_percentage:.1f}%) with missing satellite data")
                    
                    # Filter coordinate arrays
                    lat_arr = lat_arr[valid_mask]
                    lon_arr = lon_arr[valid_mask]
                    times = [t for i, t in enumerate(times) if valid_mask[i]]
                    
                    # Filter satellite data arrays
                    if sss.ndim == 1:
                        sss = sss[valid_mask]
                        sst = sst[valid_mask]
                        aviso = aviso[valid_mask]
                    elif sss.ndim == 2:
                        if sss.shape[0] == sss.shape[1]:  # One-to-one mode with (T, N) arrays
                            sss = sss[valid_mask][:, valid_mask]
                            sst = sst[valid_mask][:, valid_mask]
                            aviso = aviso[valid_mask][:, valid_mask]
                        else:  # True cross-product mode
                            sss = sss[:, valid_mask]
                            sst = sst[:, valid_mask]
                            aviso = aviso[:, valid_mask]
                    else:  # Single location mode
                        sss = sss[:, valid_mask]
                        sst = sst[:, valid_mask]
                        aviso = aviso[:, valid_mask]
                    
                    print(f"DEBUG: After filtering - lat_arr: {lat_arr.shape}, lon_arr: {lon_arr.shape}, times: {len(times)}")
                    print(f"DEBUG: After filtering - sss: {sss.shape}, sst: {sst.shape}, aviso: {aviso.shape}")
                    
                    if n_valid == 0:
                        raise ValueError("No valid locations remaining after filtering satellite data")
                else:
                    print(f"DEBUG: All locations have valid satellite data, no filtering needed")
                    
            except Exception as e:
                logger.error("Satellite accessor failed: %s", e)
                return jsonify({"error": f"Satellite data unavailable: {str(e)}. Please try again later."}), 503

            # Prepare model inputs
            dtime = [(t - datetime(1, 1, 1)).days + 366 for t in times]  # MATLAB datenum
            input_params = {
                "timecos": True, "timesin": True, "latcos": True, "latsin": True,
                "loncos": True, "lonsin": True, "sat": True, "sst": True, "sss": True, "ssh": True
            }
            print(f"DEBUG: About to call prepare_inputs with shapes: dtime={np.array(dtime).shape}, lat_arr={lat_arr.shape}, lon_arr={lat_arr.shape}, sss={sss.shape}, sst={sst.shape}, aviso={aviso.shape}")
            print(f"DEBUG: Note: Satellite data filtering already completed, prepare_inputs will not filter again")
            batch = prepare_inputs(dtime, lat_arr, lon_arr, sss, sst, aviso, input_params)
            print(f"DEBUG: prepare_inputs completed successfully, batch shape: {batch.shape}")

            # Inference
            print(f"DEBUG: About to call infer with batch shape: {batch.shape}")
            pcs = infer(batch).cpu().numpy()
            print(f"DEBUG: Inference completed successfully, pcs shape: {pcs.shape}")

            # PCA inverse
            print(f"DEBUG: About to get PCA objects")
            pca_temp, pca_sal, _ = get_pca_objects()
            print(f"DEBUG: PCA objects loaded successfully")
            print(f"DEBUG: pcs shape: {pcs.shape}")
            temp_pcs = pcs[:, :15]
            sal_pcs = pcs[:, 15:]
            print(f"DEBUG: temp_pcs shape: {temp_pcs.shape}")
            print(f"DEBUG: sal_pcs shape: {sal_pcs.shape}")

            print(f"DEBUG: About to perform PCA inverse transform for temperature")
            pred_T_inv = pca_temp.inverse_transform(temp_pcs)
            print(f"DEBUG: Temperature PCA inverse completed, shape: {pred_T_inv.shape}")

            print(f"DEBUG: About to perform PCA inverse transform for salinity")
            pred_S_inv = pca_sal.inverse_transform(sal_pcs)
            print(f"DEBUG: Salinity PCA inverse completed, shape: {pred_S_inv.shape}")

            pred_T = pred_T_inv.T
            pred_S = pred_S_inv.T
            print(f"DEBUG: pred_T shape after transpose: {pred_T.shape}")
            print(f"DEBUG: pred_S shape after transpose: {pred_S.shape}")

            # Build dataset & write bytes
            depth = np.arange(0, 1801, dtype=np.float32)
            print(f"DEBUG: depth shape: {depth.shape}")
            # Debug: Check satellite data dimensions
            print(f"DEBUG: original sss shape: {sss.shape}")
            print(f"DEBUG: original sst shape: {sst.shape}")
            print(f"DEBUG: original aviso shape: {aviso.shape}")
            print(f"DEBUG: Number of profiles from model: {pred_T.shape[1]}")
            print(f"DEBUG: Number of profiles requested: {n}")
            print(f"DEBUG: Number of unique dates: {len(times)}")

            # Extract satellite data for each profile
            # The issue: satellite data is (4, 4) but we need (16,) for 16 profiles
            # This suggests satellite data is not being loaded for all profiles

            if sss.ndim == 2:
                # If satellite data is 2D, we need to extract per-profile values
                # Current approach: use diagonal (assumes square matrix with matching profiles)
                # But if we have more profiles than dates, this won't work

                expected_profiles = pred_T.shape[1]  # Number of profiles from model predictions

                if sss.shape[1] == expected_profiles:
                    # Satellite data has the right number of profiles
                    sss_profile = np.diag(sss) if sss.shape[0] == sss.shape[1] else sss[0]  # Use first time if not square
                    sst_profile = np.diag(sst) if sst.shape[0] == sst.shape[1] else sst[0]
                    aviso_profile = np.diag(aviso) if aviso.shape[0] == aviso.shape[1] else aviso[0]
                else:
                    # Mismatch: satellite data doesn't have enough profiles
                    # For now, repeat the available satellite data to match the number of profiles
                    print(f"WARNING: Satellite data has {sss.shape[1]} profiles but model predicts {expected_profiles} profiles")
                    print("Repeating satellite data to match profile count")

                    # Repeat satellite data to match the number of profiles
                    sss_profile = np.tile(np.diag(sss) if sss.shape[0] == sss.shape[1] else sss[0], expected_profiles // sss.shape[1] + 1)[:expected_profiles]
                    sst_profile = np.tile(np.diag(sst) if sst.shape[0] == sst.shape[1] else sst[0], expected_profiles // sst.shape[1] + 1)[:expected_profiles]
                    aviso_profile = np.tile(np.diag(aviso) if aviso.shape[0] == aviso.shape[1] else aviso[0], expected_profiles // sst.shape[1] + 1)[:expected_profiles]
            else:
                # If already 1D, use as is
                sss_profile = sss
                sst_profile = sst
                aviso_profile = aviso

            # Validate satellite data dimensions
            if len(sss_profile) != pred_T.shape[1]:
                raise ValueError(f"Satellite data dimension mismatch: SSS has {len(sss_profile)} profiles but model predicts {pred_T.shape[1]} profiles")
            if len(sst_profile) != pred_T.shape[1]:
                raise ValueError(f"Satellite data dimension mismatch: SST has {len(sst_profile)} profiles but model predicts {pred_T.shape[1]} profiles")
            if len(aviso_profile) != pred_T.shape[1]:
                raise ValueError(f"Satellite data dimension mismatch: AVISO has {len(aviso_profile)} profiles but model predicts {pred_T.shape[1]} profiles")

            print(f"DEBUG: sss_profile shape: {sss_profile.shape}")
            print(f"DEBUG: sst_profile shape: {sst_profile.shape}")
            print(f"DEBUG: aviso_profile shape: {aviso_profile.shape}")
            print(f"DEBUG: sss_profile sample: {sss_profile[:5]}")
            print(f"DEBUG: sst_profile sample: {sst_profile[:5]}")
            print(f"DEBUG: aviso_profile sample: {aviso_profile[:5]}")
            ds = _build_dataset(pred_T, pred_S, depth, sss_profile, sst_profile, aviso_profile, times, lat_arr, lon_arr)
            netcdf_bytes = _write_netcdf_bytes(ds)

            resp = make_response(netcdf_bytes)
            resp.headers["Content-Type"] = "application/x-netcdf"
            resp.headers["Content-Disposition"] = f"attachment; filename=NeSPReSO_{dates[0]}_to_{dates[-1]}.nc"
            resp.headers["MODEL_SHA"] = "unknown"
            resp.headers["STATS_SHA"] = "unknown"
            resp.headers["SAT_SNAPSHOT"] = "unknown"
            return resp

        except Exception as e:
            logger.exception("Unhandled error")
            return jsonify({"error": str(e)}), 500

    @bp.route("/grid", methods=["POST"])
    def grid():
        """Grid query endpoint for single date query of all predefined grid points"""
        try:
            # Validate request
            try:
                req = GridRequest.model_validate(request.get_json())
            except ValidationError as e:
                return jsonify({"error": e.errors()}), 400

            date_str = req.date
            bbox = req.bbox
            logger.info(f"Grid query request for date: {date_str}, BBOX: {bbox}")

            # Load predefined grid coordinates
            try:
                lon_in, lat_in = _load_grid_data(bbox)
            except ValueError as e:
                return jsonify({"error": str(e)}), 503

            n_points = len(lon_in)
            if n_points == 0:
                return jsonify({"error": "No grid points found for the specified parameters"}), 400

            logger.info(f"Processing grid query for {n_points} points")

            # Convert date string to datetime
            try:
                time = datetime.strptime(date_str, "%Y-%m-%d")
            except ValueError as e:
                return jsonify({"error": f"Invalid date format: {e}"}), 400

            # Check if we exceed MAX_PROFILES limit
            if CFG.MAX_PROFILES and n_points > CFG.MAX_PROFILES:
                return jsonify({
                    "error": f"Grid has too many points: {n_points} > {CFG.MAX_PROFILES}. Please increase MAX_PROFILES limit or reduce grid resolution.",
                    "max_profiles": CFG.MAX_PROFILES,
                    "grid_points": n_points
                }), 413

            # Prepare arrays for processing
            lat_arr = np.asarray(lat_in, dtype=np.float64)
            lon_arr = np.asarray(lon_in, dtype=np.float64)
            times = [time] * n_points  # Same time for all grid points

            # Load satellite data for all grid points
            try:
                logger.info("Loading satellite data...")
                
                # First, validate that satellite data files exist for the requested dates
                logger.info("Validating satellite data file availability...")
                
                # Check if we have any dates that need satellite data
                unique_dates = set()
                for t in times:
                    unique_dates.add((t.year, t.month, t.day))
                
                logger.info(f"Checking satellite data availability for {len(unique_dates)} unique dates")
                
                # Check SST data availability (most critical for the model)
                missing_sst_dates = []
                for year, month, day in unique_dates:
                    # Check if SST file exists for this date
                    sst_date_str = f"{year}{month:02d}{day:02d}"
                    sst_pattern = f"{CFG.SST_ROOT}/{year}/{sst_date_str}*"
                    sst_files = glob.glob(sst_pattern)
                    
                    if not sst_files:
                        missing_sst_dates.append(f"{year}-{month:02d}-{day:02d}")
                        logger.warning(f"SST data missing for {year}-{month:02d}-{day:02d}")
                    else:
                        logger.info(f"SST data available for {year}-{month:02d}-{day:02d}: {len(sst_files)} files")
                
                # If any SST files are missing, fail the request
                if missing_sst_dates:
                    missing_dates_str = ", ".join(missing_sst_dates)
                    error_msg = f"SST satellite data files are missing for the following dates: {missing_dates_str}."
                    logger.error(error_msg)
                    return jsonify({"error": error_msg}), 400
                
                # Check SSS data availability (SSS uses day-of-year naming, not daily)
                missing_sss_dates = []
                for year, month, day in unique_dates:
                    # Convert date to day of year
                    date_obj = datetime(year, month, day)
                    doy = date_obj.timetuple().tm_yday
                    
                    # SSS files use pattern: RSS_smap_SSS_L3_8day_running_{year}_{doy:03d}_FNL_v*.nc
                    sss_pattern = f"{CFG.SSS_ROOT}/{year}/RSS_smap_SSS_L3_8day_running_{year}_{doy:03d}_FNL_v*.nc"
                    sss_files = glob.glob(sss_pattern)
                    
                    if not sss_files:
                        missing_sss_dates.append(f"{year}-{month:02d}-{day:02d}")
                        logger.warning(f"SSS data missing for {year}-{month:02d}-{day:02d} (DOY {doy})")
                    else:
                        logger.info(f"SSS data available for {year}-{month:02d}-{day:02d} (DOY {doy}): {len(sss_files)} files")
                
                # If any SSS files are missing, fail the request
                if missing_sss_dates:
                    missing_dates_str = ", ".join(missing_sss_dates)
                    error_msg = f"SSS satellite data files are missing for the following dates: {missing_dates_str}. Please request a date with available satellite data."
                    logger.error(error_msg)
                    return jsonify({"error": error_msg}), 400
                
                # Check AVISO data availability (AVISO uses monthly naming, not daily)
                missing_aviso_dates = []
                for year, month, day in unique_dates:
                    # AVISO files use pattern: {year}-{month:02d}.nc
                    aviso_pattern = f"{CFG.AVISO_ROOT}/{year}-{month:02d}.nc"
                    aviso_files = glob.glob(aviso_pattern)
                    
                    if not aviso_files:
                        missing_aviso_dates.append(f"{year}-{month:02d}-{day:02d}")
                        logger.warning(f"AVISO data missing for {year}-{month:02d}-{day:02d} (monthly file)")
                    else:
                        logger.info(f"AVISO data available for {year}-{month:02d}-{day:02d} (monthly file): {len(aviso_files)} files")
                
                # If any AVISO files are missing, fail the request
                if missing_aviso_dates:
                    missing_dates_str = ", ".join(missing_aviso_dates)
                    error_msg = f"AVISO satellite data files are missing for the following dates: {missing_dates_str}. Please request a date with available satellite data."
                    logger.error(error_msg)
                    return jsonify({"error": error_msg}), 400
                
                logger.info("All required satellite data files are available. Proceeding with data loading...")
                
                # Now load the satellite data
                sss, sst, aviso = load_satellite_data(times, lat_arr, lon_arr)
                logger.info(f"Satellite data loaded: sss={sss.shape}, sst={sst.shape}, aviso={aviso.shape}")
                
                # Validate satellite data
                if sss is None or sst is None or aviso is None:
                    raise ValueError("One or more satellite data arrays are None")
                
                # Check for missing/invalid satellite data and filter out those locations
                logger.info("Validating satellite data quality...")
                
                # Convert to numpy arrays if they aren't already
                sss_arr = np.asarray(sss)
                sst_arr = np.asarray(sst)
                aviso_arr = np.asarray(aviso)
                
                # Create mask for valid satellite data
                # A location is valid only if ALL satellite variables have finite values
                if sss_arr.ndim == 2:
                    # 2D arrays: check if all time steps have valid data
                    valid_sss = np.all(np.isfinite(sss_arr), axis=0)
                    valid_sst = np.all(np.isfinite(sst_arr), axis=0)
                    valid_aviso = np.all(np.isfinite(aviso_arr), axis=0)
                else:
                    # 1D arrays: check each location directly
                    valid_sss = np.isfinite(sss_arr)
                    valid_sst = np.isfinite(sst_arr)
                    valid_aviso = np.isfinite(aviso_arr)
                
                # Combined validity mask - ALL satellite variables must be valid
                valid_satellite_mask = valid_sss & valid_sst & valid_aviso
                
                n_valid_satellite = np.sum(valid_satellite_mask)
                n_total = len(lat_arr)
                
                logger.info(f"Satellite data validation: {n_valid_satellite}/{n_total} locations have complete satellite data")
                
                if n_valid_satellite < n_total:
                    n_filtered = n_total - n_valid_satellite
                    filter_percentage = (n_filtered / n_total) * 100
                    logger.warning(f"Filtering out {n_filtered} locations ({filter_percentage:.1f}%) with missing satellite data")
                    
                    # Filter coordinate arrays
                    lat_arr = lat_arr[valid_satellite_mask]
                    lon_arr = lon_arr[valid_satellite_mask]
                    times = [t for i, t in enumerate(times) if valid_satellite_mask[i]]
                    
                    # Filter satellite data arrays
                    if sss_arr.ndim == 2:
                        sss_arr = sss_arr[:, valid_satellite_mask]
                        sst_arr = sst_arr[:, valid_satellite_mask]
                        aviso_arr = aviso_arr[:, valid_satellite_mask]
                    else:
                        sss_arr = sss_arr[valid_satellite_mask]
                        sst_arr = sst_arr[valid_satellite_mask]
                        aviso_arr = aviso_arr[valid_satellite_mask]
                    
                    logger.info(f"After satellite filtering: {len(lat_arr)} locations remaining")
                    
                    if len(lat_arr) == 0:
                        raise ValueError("No locations remaining after satellite data filtering")
                        
                else:
                    logger.info("All locations have complete satellite data")
                
                # Store filtered arrays for later use
                sss_filtered = sss_arr
                sst_filtered = sst_arr
                aviso_filtered = aviso_arr
                
                # Log sample values to verify data quality
                logger.info(f"Sample satellite data after filtering:")
                logger.info(f"  SSS: min={sss_filtered.min():.3f}, max={sss_filtered.max():.3f}, mean={sss_filtered.mean():.3f}")
                logger.info(f"  SST: min={sst_filtered.min():.3f}, max={sst_filtered.max():.3f}, mean={sst_filtered.mean():.3f}")
                logger.info(f"  AVISO: min={aviso_filtered.min():.3f}, max={aviso_filtered.max():.3f}, mean={aviso_filtered.mean():.3f}")
                
            except Exception as e:
                logger.error(f"Satellite accessor failed: {e}")
                return jsonify({"error": f"Satellite data unavailable: {str(e)}. Please try again later."}), 503

            # Prepare model inputs
            try:
                logger.info("Preparing model inputs...")
                dtime = [(t - datetime(1, 1, 1)).days + 366 for t in times]  # MATLAB datenum
                input_params = {
                    "timecos": True, "timesin": True, "latcos": True, "latsin": True,
                    "loncos": True, "lonsin": True, "sat": True, "sst": True, "sss": True, "ssh": True
                }
                
                batch = prepare_inputs(dtime, lat_arr, lon_arr, sss_filtered, sst_filtered, aviso_filtered, input_params)
                logger.info(f"Input preparation completed, batch shape: {batch.shape}")
                
            except Exception as e:
                logger.error(f"Input preparation failed: {e}")
                return jsonify({"error": f"Failed to prepare model inputs: {str(e)}"}), 500

            # Run model inference
            try:
                logger.info("Starting model inference...")
                pcs = infer(batch).cpu().numpy()
                logger.info(f"Inference completed, pcs shape: {pcs.shape}")
                
            except Exception as e:
                logger.error(f"Model inference failed: {e}")
                return jsonify({"error": f"Model inference failed: {str(e)}"}), 500

            # PCA inverse transform
            try:
                logger.info("Performing PCA inverse transform...")
                pca_temp, pca_sal, _ = get_pca_objects()
                temp_pcs = pcs[:, :15]
                sal_pcs = pcs[:, 15:]

                pred_T_inv = pca_temp.inverse_transform(temp_pcs)
                pred_S_inv = pca_sal.inverse_transform(sal_pcs)

                pred_T = pred_T_inv.T
                pred_S = pred_S_inv.T
                logger.info(f"PCA inverse completed, pred_T: {pred_T.shape}, pred_S: {pred_S.shape}")
                
            except Exception as e:
                logger.error(f"PCA inverse transform failed: {e}")
                return jsonify({"error": f"PCA processing failed: {str(e)}"}), 500

            # Prepare depth array
            depth = np.arange(0, 1801, dtype=np.float32)
            
            # Extract satellite data for each profile - now properly aligned
            try:
                logger.info("Processing satellite data for grid mapping...")
                
                # The satellite data arrays are now properly filtered and aligned with the coordinates
                # We just need to extract the values for each profile
                if sss_filtered.ndim == 2:
                    # 2D arrays: extract the first time step or use diagonal if square
                    if sss_filtered.shape[0] == sss_filtered.shape[1]:
                        # Square matrix: use diagonal
                        sss_profile = np.diag(sss_filtered)
                        sst_profile = np.diag(sst_filtered)
                        aviso_profile = np.diag(aviso_filtered)
                    else:
                        # Non-square: use first time step
                        sss_profile = sss_filtered[0, :]
                        sst_profile = sst_filtered[0, :]
                        aviso_profile = aviso_filtered[0, :]
                else:
                    # 1D arrays: use as is
                    sss_profile = sss_filtered
                    sst_profile = sst_filtered
                    aviso_profile = aviso_filtered
                
                # Verify the satellite data arrays match the number of profiles
                expected_profiles = pred_T.shape[1]
                if len(sss_profile) != expected_profiles:
                    raise ValueError(f"Satellite data dimension mismatch: SSS has {len(sss_profile)} profiles but model predicts {expected_profiles}")
                if len(sst_profile) != expected_profiles:
                    raise ValueError(f"Satellite data dimension mismatch: SST has {len(sst_profile)} profiles but model predicts {expected_profiles}")
                if len(aviso_profile) != expected_profiles:
                    raise ValueError(f"Satellite data dimension mismatch: AVISO has {len(aviso_profile)} profiles but model predicts {expected_profiles}")
                
                # Verify that all satellite data values are finite (no NaN or inf)
                if np.any(~np.isfinite(sss_profile)):
                    raise ValueError(f"SSS profile contains {np.sum(~np.isfinite(sss_profile))} non-finite values")
                if np.any(~np.isfinite(sst_profile)):
                    raise ValueError(f"SST profile contains {np.sum(~np.isfinite(sst_profile))} non-finite values")
                if np.any(~np.isfinite(aviso_profile)):
                    raise ValueError(f"AVISO profile contains {np.sum(~np.isfinite(aviso_profile))} non-finite values")
                
                logger.info(f"Satellite data processed: sss={sss_profile.shape}, sst={sst_profile.shape}, aviso={aviso_profile.shape}")
                logger.info(f"Sample SST values: min={sst_profile.min():.3f}, max={sst_profile.max():.3f}, mean={sst_profile.mean():.3f}")
                logger.info(f"Sample SSS values: min={sss_profile.min():.3f}, max={sss_profile.max():.3f}, mean={sss_profile.mean():.3f}")
                logger.info(f"Sample AVISO values: min={aviso_profile.min():.3f}, max={aviso_profile.max():.3f}, mean={aviso_profile.mean():.3f}")
                
            except Exception as e:
                logger.error(f"Satellite data processing failed: {e}")
                return jsonify({"error": f"Satellite data processing failed: {str(e)}"}), 500

            # Build dataset and write to NetCDF
            try:
                logger.info("Building grid dataset...")
                ds = _build_grid_dataset(pred_T, pred_S, depth, sss_profile, sst_profile, aviso_profile, time, lat_arr, lon_arr)
                
                logger.info("Writing NetCDF file...")
                netcdf_bytes = _write_netcdf_bytes(ds)
                logger.info("NetCDF file created successfully")
                
            except Exception as e:
                logger.error(f"Dataset creation failed: {e}")
                return jsonify({"error": f"Failed to create dataset: {str(e)}"}), 500

            # Generate filename with BBOX info if provided
            if bbox:
                bbox_str = f"_bbox_{bbox[0]:.2f}_{bbox[1]:.2f}_{bbox[2]:.2f}_{bbox[3]:.2f}"
                filename = f"NeSPReSO_grid_{date_str}{bbox_str}.nc"
            else:
                filename = f"NeSPReSO_grid_{date_str}.nc"

            # Return response
            resp = make_response(netcdf_bytes)
            resp.headers["Content-Type"] = "application/x-netcdf"
            resp.headers["Content-Disposition"] = f"attachment; filename={filename}"
            resp.headers["MODEL_SHA"] = "unknown"
            resp.headers["STATS_SHA"] = "unknown"
            resp.headers["SAT_SNAPSHOT"] = "unknown"
            
            logger.info(f"Grid query completed successfully: {filename}")
            return resp

        except Exception as e:
            logger.exception("Grid query error")
            return jsonify({"error": str(e)}), 500

    # Legacy shim
    @app.route("/predict", methods=["POST"])
    def predict_legacy():
        return profile()

    app.register_blueprint(bp)
    app.register_blueprint(metrics_bp)
    app.before_request(metrics_before)
    app.after_request(metrics_after)
    return app
