from flask import Flask, Blueprint, request, jsonify, make_response
from pydantic import BaseModel, ValidationError, field_validator
from typing import List
import numpy as np
import torch
import xarray as xr
import logging
import sys
from services.kernel.handler import infer, get_pca_objects
from services.accessor.sat import load_satellite_data, prepare_inputs
from datetime import datetime
from services.api.metrics import metrics_bp, before_request as metrics_before, after_request as metrics_after
from tenacity import RetryError

# --- Pydantic model for request validation ---
class ProfileRequest(BaseModel):
    lat: List[float]
    lon: List[float]
    date: List[str]

    @field_validator('lat', 'lon', 'date')
    @classmethod
    def check_nonempty(cls, v):
        if not isinstance(v, list) or len(v) < 1:
            raise ValueError('Must be a non-empty list')
        return v

    @field_validator('date')
    @classmethod
    def check_date_format(cls, v):
        for d in v:
            try:
                datetime.strptime(d, "%Y-%m-%d")
            except Exception:
                raise ValueError(f"Date {d} is not in YYYY-MM-DD format")
        return v

# --- Structured JSON logger ---
def get_logger():
    logger = logging.getLogger("ocean")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('{"time": "%(asctime)s", "level": "%(levelname)s", "msg": %(message)s}')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger
logger = get_logger()

# --- NetCDF in-memory writer ---
def write_netcdf_to_bytes(pred_T, pred_S, depth, sss, sst, aviso, times, lat, lon):
    profile_number = np.arange(pred_T.shape[1])
    depth = depth.astype(np.float32)
    times = np.array([np.datetime64(time) for time in times])
    ds = xr.Dataset({
        'Temperature': (('depth', 'profile_number'), pred_T),
        'Salinity': (('depth', 'profile_number'), pred_S),
        'SSS': (('profile_number'), sss),
        'SST': (('profile_number'), sst),
        'AVISO': (('profile_number'), aviso),
        'time': (('profile_number'), times),
        'lat': (('profile_number'), lat),
        'lon': (('profile_number'), lon)
    }, coords={
        'profile_number': profile_number,
        'depth': depth
    })
    # Prefer NetCDF4 engine with compression, fall back to scipy without compression.
    comp = dict(zlib=True, complevel=9)
    encoding = {var: comp for var in ds.data_vars}
    encoding.update({var: comp for var in ds.coords if var != 'profile_number'})

    try:
        # Write directly to bytes using the NetCDF4 engine (keeps file handle management internal).
        return ds.to_netcdf(None, mode="w", engine="netcdf4", encoding=encoding)
    except Exception as e:
        logger.warning(f"Falling back to default NetCDF engine (no compression): {e}")
        # The scipy engine does not support compression, so we omit the encoding.
        return ds.to_netcdf(None, mode="w", engine="scipy")

# --- Blueprint and app factory ---
def create_app(config: dict = None) -> Flask:
    app = Flask(__name__)
    if config:
        app.config.update(config)
    bp = Blueprint("profile", __name__, url_prefix="/v1/profile")

    # No batch size limits - removed MAX_BATCH and enforce_max_batch
    
    @bp.route("", methods=["POST"])
    def profile():
        try:
            # Validate request
            try:
                req = ProfileRequest.model_validate(request.get_json())
            except ValidationError as e:
                logger.error(f"Validation error: {e}")
                return jsonify({"error": e.errors()}), 400
            lat, lon, dates = req.lat, req.lon, req.date
            if not (len(lat) == len(lon) == len(dates)):
                return jsonify({"error": "Length of 'lat', 'lon', and 'date' must be equal"}), 400
            times = [datetime.strptime(d, "%Y-%m-%d") for d in dates]
            lat = np.array(lat)
            lon = np.array(lon)
            # --- Satellite accessor ---
            try:
                logger.info(f"Loading satellite data for {len(times)} dates and {len(lat)} locations...")
                sss, sst, aviso = load_satellite_data(times, lat, lon)
                logger.info("Satellite data loading completed successfully")
            except RetryError:
                logger.error("Satellite accessor circuit-breaker tripped")
                return jsonify({"error": "Satellite data unavailable, please try again later."}), 503
            # --- Prepare model input ---
            dtime = [(t - datetime(1, 1, 1)).days + 366 for t in times]  # MATLAB datenum
            logger.info(f"Converted dates to MATLAB datenum: {dtime[:5]}... (showing first 5)")
            logger.info(f"Latitude range: {lat.min():.4f} to {lat.max():.4f}")
            logger.info(f"Longitude range: {lon.min():.4f} to {lon.max():.4f}")
            
            input_params = {
                "timecos": True, "timesin": True, "latcos": True, "latsin": True,
                "loncos": True, "lonsin": True, "sat": True, "sst": True, "sss": True, "ssh": True
            }
            logger.info("Preparing model inputs...")
            input_data = prepare_inputs(dtime, lat, lon, sss, sst, aviso, input_params)
            logger.info(f"Input data shape: {input_data.shape}")
            logger.info(f"Input data sample (first 3 rows): {input_data[:3]}")
            logger.info(f"Input data contains NaN: {np.isnan(input_data).any()}")
            
            # --- Model inference ---
            logger.info("Running model inference...")
            pcs_predictions = infer(input_data)
            pcs_predictions = pcs_predictions.cpu().numpy()
            logger.info(f"Model predictions shape: {pcs_predictions.shape}")
            logger.info(f"Model predictions sample (first 3 rows): {pcs_predictions[:3]}")
            logger.info(f"Model predictions contain NaN: {np.isnan(pcs_predictions).any()}")
            
            # Clean up input data to free memory
            del input_data
            import gc
            gc.collect()
            
            # --- PCA inverse transform ---
            logger.info("Applying PCA inverse transform...")
            pca_temp, pca_sal, _ = get_pca_objects()
            pred_T = pca_temp.inverse_transform(pcs_predictions[:, :15]).T
            pred_S = pca_sal.inverse_transform(pcs_predictions[:, 15:]).T
            logger.info(f"Temperature predictions shape: {pred_T.shape}")
            logger.info(f"Salinity predictions shape: {pred_S.shape}")
            logger.info(f"Temperature contains NaN: {np.isnan(pred_T).any()}")
            logger.info(f"Salinity contains NaN: {np.isnan(pred_S).any()}")
            
            # Clean up PCA predictions to free memory
            del pcs_predictions
            gc.collect()
            
            depth = np.arange(0, 1801)
            # --- NetCDF in-memory ---
            logger.info("Creating NetCDF output...")
            netcdf_bytes = write_netcdf_to_bytes(pred_T, pred_S, depth, sss, sst, aviso, times, lat, lon)
            
            # Clean up final arrays to free memory
            del pred_T, pred_S, sss, sst, aviso
            gc.collect()
            # --- Response ---
            response = make_response(netcdf_bytes)
            response.headers["Content-Type"] = "application/x-netcdf"
            response.headers["Content-Disposition"] = f"attachment; filename=NeSPReSO_{dates[0]}_to_{dates[-1]}.nc"
            # Add snapshot headers (placeholders)
            response.headers["MODEL_SHA"] = "unknown"
            response.headers["STATS_SHA"] = "unknown"
            response.headers["SAT_SNAPSHOT"] = "unknown"
            logger.info(f"Request served: {len(lat)} profiles, {request.remote_addr}")
            return response
        except Exception as e:
            logger.error(f"Internal error: {e}")
            return jsonify({"error": str(e)}), 500

    # ------------------------------------------------------------------
    # Legacy endpoint – keeps old clients functional without changes.
    # ------------------------------------------------------------------
    @app.route("/predict", methods=["POST"])
    def predict_legacy():  # noqa: D401 – simple pass-through
        """Backward-compat shim that forwards to /v1/profile."""
        return profile()

    app.register_blueprint(bp)
    app.register_blueprint(metrics_bp)
    app.before_request(metrics_before)
    app.after_request(metrics_after)
    return app 