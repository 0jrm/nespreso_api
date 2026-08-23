"""DA-facing ``/v1_profile/{model}`` routes. SAT ``/v1_profile`` stays in ``app.py``."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import numpy as np
import xarray as xr
from flask import Blueprint, jsonify, make_response, request
from pydantic import ValidationError

from services.common.v2_spec import (
    R_KIND,
    SERVED_MODELS,
    SigmaOTable,
    V2BadRequestError,
    V2UnavailableError,
    load_sigma_o,
    resolve_seed,
)
from services.config import CFG
from services.kernel.v2_cells import V2Decode, get_registry, predict_profiles

logger = logging.getLogger("ocean")


def _parse_seed(model: str) -> int:
    raw = request.args.get("seed")
    seed: int | None
    if raw is None or raw == "":
        seed = None
    else:
        try:
            seed = int(raw)
        except ValueError as exc:
            raise V2BadRequestError("seed must be an integer") from exc
    return resolve_seed(model, seed)


def _attach_sigma_o(ds: xr.Dataset, table: SigmaOTable) -> xr.Dataset:
    """Attach Dai σ_o as a 41-layer sidecar coordinate, not CRPS-head σ."""
    k = np.arange(table.zmid_m.shape[0], dtype=np.int32)
    ds = ds.assign_coords(hycom_k=("hycom_k", k))
    ds["sigma_o_zmid"] = ("hycom_k", np.asarray(table.zmid_m, dtype=np.float32))
    ds["sigma_o_T"] = ("hycom_k", np.asarray(table.sigma_t, dtype=np.float32))
    ds["sigma_o_S"] = ("hycom_k", np.asarray(table.sigma_s, dtype=np.float32))
    ds["sigma_o_T"].attrs.update(
        {"units": "degree_C", "long_name": "Dai sigma_o T after H"}
    )
    ds["sigma_o_S"].attrs.update({"units": "psu", "long_name": "Dai sigma_o S after H"})
    if table.sigma_t_lc is not None and table.sigma_s_lc is not None:
        ds["sigma_o_T_lc"] = ("hycom_k", np.asarray(table.sigma_t_lc, dtype=np.float32))
        ds["sigma_o_S_lc"] = ("hycom_k", np.asarray(table.sigma_s_lc, dtype=np.float32))
    if table.sigma_t_complement is not None and table.sigma_s_complement is not None:
        ds["sigma_o_T_complement"] = (
            "hycom_k",
            np.asarray(table.sigma_t_complement, dtype=np.float32),
        )
        ds["sigma_o_S_complement"] = (
            "hycom_k",
            np.asarray(table.sigma_s_complement, dtype=np.float32),
        )
    ds.attrs["sigma_o_regime"] = table.regime
    ds.attrs["sigma_o_seed"] = table.seed_label
    ds.attrs["sigma_o_floor_T"] = str(0.05)
    ds.attrs["sigma_o_floor_S"] = str(0.02)
    return ds


def _cell_attrs(decoded: V2Decode) -> dict[str, str]:
    attrs = {
        "model": decoded.model,
        "seed": str(decoded.seed),
        "checkpoint": decoded.checkpoint,
        "cache_hash": decoded.cache_hash,
        "decode": decoded.decode,
        "r_kind": R_KIND,
    }
    if decoded.cache_kind:
        attrs["cache_kind"] = decoded.cache_kind
    return attrs


def _stamp_cell(ds: xr.Dataset, decoded: V2Decode) -> xr.Dataset:
    ds.attrs.update(_cell_attrs(decoded))
    try:
        spec = get_registry().models[decoded.model]
        ds = _attach_sigma_o(
            ds, load_sigma_o(spec.sigma_o_csv, decoded.model, decoded.seed)
        )
    except (FileNotFoundError, OSError, ValueError, KeyError) as exc:
        logger.warning("sigma_o table not attached: %s", exc)
    return ds


def _netcdf_response(ds: xr.Dataset, filename: str, decoded: V2Decode) -> Any:
    from services.api.app import _write_netcdf_bytes

    ds = _stamp_cell(ds, decoded)
    if "sigma" in ds.data_vars or "err" in ds.data_vars:
        raise RuntimeError("refusing to write CRPS-head sigma into NetCDF")
    body = _write_netcdf_bytes(ds)
    resp = make_response(body)
    resp.headers["Content-Type"] = "application/x-netcdf"
    resp.headers["Content-Disposition"] = f"attachment; filename={filename}"
    resp.headers["MODEL_SHA"] = decoded.checkpoint_stem
    resp.headers["STATS_SHA"] = decoded.cache_hash or "unknown"
    resp.headers["SAT_SNAPSHOT"] = "unknown"
    return resp


def _run_cell(
    model: str,
    seed: int,
    times: list[datetime],
    lat: np.ndarray,
    lon: np.ndarray,
) -> tuple[V2Decode, np.ndarray, np.ndarray, np.ndarray]:
    from services.accessor.v2_inputs import build_v2_inputs, load_sat_or_503

    sss, sst, ssh = load_sat_or_503(times, lat, lon)
    features = build_v2_inputs(model, times, lat, lon, sss, sst, ssh)
    decoded = predict_profiles(model, features, seed=seed)
    return decoded, sss, sst, ssh


def handle_v2_profile(model: str) -> Any:
    """POST ``/v1_profile/{model}`` — same body as the SAT profile route."""
    from services.api.app import ProfileRequest, _build_dataset

    if model not in SERVED_MODELS:
        return (
            jsonify(
                {"error": f"Unknown model {model!r}", "served": list(SERVED_MODELS)}
            ),
            404,
        )
    try:
        seed = _parse_seed(model)
        try:
            req = ProfileRequest.model_validate(request.get_json())
        except ValidationError as e:
            return jsonify({"error": e.errors()}), 400
        lat, lon, dates = req.lat, req.lon, req.date
        if not (len(lat) == len(lon) == len(dates)):
            return (
                jsonify({"error": "Length of 'lat', 'lon', and 'date' must be equal"}),
                400,
            )
        n = len(lat)
        if CFG.MAX_PROFILES and n > CFG.MAX_PROFILES:
            return (
                jsonify(
                    {
                        "error": f"Too many profiles: {n} > {CFG.MAX_PROFILES}.",
                        "max_profiles": CFG.MAX_PROFILES,
                        "requested_profiles": n,
                    }
                ),
                413,
            )
        times = [datetime.strptime(d, "%Y-%m-%d") for d in dates]
        lat_arr = np.asarray(lat, dtype=np.float64)
        lon_arr = np.asarray(lon, dtype=np.float64)
        decoded, sss, sst, ssh = _run_cell(model, seed, times, lat_arr, lon_arr)
        ds = _build_dataset(
            decoded.temperature,
            decoded.salinity,
            decoded.depth,
            sss,
            sst,
            ssh,
            times,
            lat_arr,
            lon_arr,
        )
        filename = f"NeSPReSO_{model}_{dates[0]}_to_{dates[-1]}.nc"
        return _netcdf_response(ds, filename, decoded)
    except V2BadRequestError as exc:
        return jsonify({"error": str(exc)}), 400
    except V2UnavailableError as exc:
        logger.error("v2 profile unavailable: %s", exc)
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:
        logger.exception("v2 profile error")
        return jsonify({"error": str(exc)}), 500


def handle_v2_grid(model: str) -> Any:
    """POST ``/v1_profile/{model}/grid`` — same body as the SAT grid route."""
    from services.api.app import GridRequest, _build_grid_dataset, _load_grid_data

    if model not in SERVED_MODELS:
        return (
            jsonify(
                {"error": f"Unknown model {model!r}", "served": list(SERVED_MODELS)}
            ),
            404,
        )
    try:
        seed = _parse_seed(model)
        try:
            req = GridRequest.model_validate(request.get_json())
        except ValidationError as e:
            return jsonify({"error": e.errors()}), 400
        lon_in, lat_in = _load_grid_data(req.bbox, req.resolution)
        n_points = len(lon_in)
        if n_points == 0:
            return (
                jsonify({"error": "No grid points found for the specified parameters"}),
                400,
            )
        if CFG.MAX_PROFILES and n_points > CFG.MAX_PROFILES:
            return (
                jsonify(
                    {
                        "error": f"Grid has too many points: {n_points} > {CFG.MAX_PROFILES}.",
                        "max_profiles": CFG.MAX_PROFILES,
                        "grid_points": n_points,
                    }
                ),
                413,
            )
        time = datetime.strptime(req.date, "%Y-%m-%d")
        lat_arr = np.asarray(lat_in, dtype=np.float64)
        lon_arr = np.asarray(lon_in, dtype=np.float64)
        times = [time] * n_points
        decoded, sss, sst, ssh = _run_cell(model, seed, times, lat_arr, lon_arr)
        ds = _build_grid_dataset(
            decoded.temperature,
            decoded.salinity,
            decoded.depth,
            sss,
            sst,
            ssh,
            time,
            lat_arr,
            lon_arr,
        )
        parts = [f"NeSPReSO_{model}_grid_{req.date}"]
        if req.bbox:
            parts.append(
                f"_bbox_{req.bbox[0]:.2f}_{req.bbox[1]:.2f}_{req.bbox[2]:.2f}_{req.bbox[3]:.2f}"
            )
        if req.resolution is not None:
            parts.append(f"_res_{req.resolution:.3f}")
        return _netcdf_response(ds, "".join(parts) + ".nc", decoded)
    except V2BadRequestError as exc:
        return jsonify({"error": str(exc)}), 400
    except V2UnavailableError as exc:
        logger.error("v2 grid unavailable: %s", exc)
        return jsonify({"error": str(exc)}), 503
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:
        logger.exception("v2 grid error")
        return jsonify({"error": str(exc)}), 500


def register_v2_routes(bp: Blueprint) -> None:
    """Attach ``/{model}`` and ``/{model}/grid`` onto the SAT profile blueprint."""

    bp.add_url_rule("/<model>", view_func=handle_v2_profile, methods=["POST"])
    bp.add_url_rule("/<model>/grid", view_func=handle_v2_grid, methods=["POST"])
