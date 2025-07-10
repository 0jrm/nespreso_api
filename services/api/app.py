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

    MAX_BATCH = 32
    @app.before_request
    def enforce_max_batch():
        if request.path.startswith("/v1/profile") and request.method == "POST":
            data = request.get_json(silent=True)
            if data:
                n = max(len(data.get("lat", [])), len(data.get("lon", [])), len(data.get("date", [])))
                if n > MAX_BATCH:
                    return jsonify({"error": f"Batch size exceeds max {MAX_BATCH}"}), 413

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
                sss, sst, aviso = load_satellite_data(times, lat, lon)
            except RetryError:
                logger.error("Satellite accessor circuit-breaker tripped")
                return jsonify({"error": "Satellite data unavailable, please try again later."}), 503
            # --- Prepare model input ---
            dtime = [(t - datetime(1, 1, 1)).days + 366 for t in times]  # MATLAB datenum
            input_params = {
                "timecos": True, "timesin": True, "latcos": True, "latsin": True,
                "loncos": True, "lonsin": True, "sat": True, "sst": True, "sss": True, "ssh": True
            }
            input_data = prepare_inputs(dtime, lat, lon, sss, sst, aviso, input_params)
            # --- Model inference ---
            pcs_predictions = infer(input_data)
            pcs_predictions = pcs_predictions.cpu().numpy()
            # --- PCA inverse transform ---
            pca_temp, pca_sal, _ = get_pca_objects()
            pred_T = pca_temp.inverse_transform(pcs_predictions[:, :15]).T
            pred_S = pca_sal.inverse_transform(pcs_predictions[:, 15:]).T
            depth = np.arange(0, 1801)
            # --- NetCDF in-memory ---
            netcdf_bytes = write_netcdf_to_bytes(pred_T, pred_S, depth, sss, sst, aviso, times, lat, lon)
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