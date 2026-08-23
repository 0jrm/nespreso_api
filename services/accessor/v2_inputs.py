"""v2 request features: SAT 9-d path, ONI/RONI splice, sat-archive operators.

Satellite I/O stays here. Model inference stays in ``services.kernel``.
Ops gradients/tendencies/geo are computed at request time from the same
SSS/SST/SSH archive used by v1 / A_CRPS / HeaveFast. Never from
``gom_cube.zarr``.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator

from services.accessor.sat import (
    _round_bbox,
    _select_aviso_root,
    cached_aviso_arrays,
    cached_sss_arrays,
    cached_sst_arrays,
    load_satellite_data,
    prepare_inputs,
)
from services.common.v2_spec import (
    V2UnavailableError,
    ensure_v2_on_path,
    load_serve_spec,
)
from services.config import CFG

logger = logging.getLogger("ocean")

_SAT_PARAMS: dict[str, bool] = {
    "timecos": True,
    "timesin": True,
    "latcos": True,
    "latsin": True,
    "loncos": True,
    "lonsin": True,
    "sat": True,
    "sst": True,
    "sss": True,
    "ssh": True,
}

OPS_FEATURE_SPEC: dict[str, Any] = {
    "spec_version": 1,
    "scalars": [],
    "operators": [
        {"op": "grad", "channels": ["sst"], "scales": ["local", "1.0deg"]},
        {"op": "grad", "channels": ["sss"], "scales": ["local", "1.0deg"]},
        {"op": "grad", "channels": ["ssh"], "scales": ["local", "1.0deg"]},
        {"op": "laplacian", "channels": ["ssh"], "scales": ["1.0deg"]},
        {"op": "tendency", "channels": ["sst", "ssh"], "window_days": 7},
        {"op": "geo_uv", "channels": ["ssh"], "scales": ["local", "1.0deg"]},
    ],
}

# 1.0deg operators plus a few gaussian sigmas of padding. Not the 0.5° SAT
# interpolation halo: derivatives need the neighborhood.
OPS_HALO_DEG: float = 4.0
_OP_NAME_CACHE: tuple[str, ...] | None = None


def _matlab_datenum(times: Sequence[datetime]) -> np.ndarray:
    """Same MATLAB-style datenum the SAT ``prepare_inputs`` path uses."""
    return np.asarray(
        [(t - datetime(1, 1, 1)).days + 366 for t in times], dtype=np.float64
    )


def sat_vectors(
    sss: np.ndarray, sst: np.ndarray, ssh: np.ndarray, n: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collapse accessor arrays to one scalar per profile.

    Args:
        sss: SSS from ``load_satellite_data``.
        sst: SST (Kelvin).
        ssh: SSH / AVISO.
        n: Expected profile count.

    Returns:
        Three 1-D arrays of length ``n``.

    Raises:
        V2UnavailableError: Shape cannot be aligned to ``n``.
    """
    sss_a = np.asarray(sss)
    sst_a = np.asarray(sst)
    ssh_a = np.asarray(ssh)

    def _one(arr: np.ndarray, name: str) -> np.ndarray:
        if arr.ndim == 1:
            out = arr
        elif arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
            out = np.diag(arr)
        elif arr.ndim == 2:
            out = arr[0]
        else:
            raise V2UnavailableError(f"{name} has unexpected shape {arr.shape}")
        out = np.asarray(out, dtype=np.float64).reshape(-1)
        if out.shape[0] != n:
            raise V2UnavailableError(f"{name} length {out.shape[0]} != {n} profiles")
        return out

    return _one(sss_a, "SSS"), _one(sst_a, "SST"), _one(ssh_a, "SSH")


def load_sat_or_503(
    times: Sequence[datetime], lat: np.ndarray, lon: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load SSS/SST/SSH. Any non-finite value is a 503, not a dropped cast.

    Args:
        times: Profile timestamps.
        lat: Latitudes, shape ``(N,)``.
        lon: Longitudes, shape ``(N,)``.

    Returns:
        1-D SSS, SST, SSH aligned to profiles.

    Raises:
        V2UnavailableError: Accessor failure or missing values.
    """
    try:
        sss, sst, ssh = load_satellite_data(list(times), lat, lon)
    except Exception as exc:
        raise V2UnavailableError(f"Satellite data unavailable: {exc}") from exc
    if sss is None or sst is None or ssh is None:
        raise V2UnavailableError("Satellite data unavailable: accessor returned None")
    n = int(lat.shape[0])
    sss_p, sst_p, ssh_p = sat_vectors(
        np.asarray(sss), np.asarray(sst), np.asarray(ssh), n
    )
    if not (
        np.isfinite(sss_p).all()
        and np.isfinite(sst_p).all()
        and np.isfinite(ssh_p).all()
    ):
        raise V2UnavailableError(
            "Satellite data unavailable: missing SSS/SST/SSH at requested locations"
        )
    return sss_p, sst_p, ssh_p


def _sat_batch(
    times: Sequence[datetime],
    lat: np.ndarray,
    lon: np.ndarray,
    sss: np.ndarray,
    sst: np.ndarray,
    ssh: np.ndarray,
) -> np.ndarray:
    """Build the SAT 9-d matrix. ``prepare_inputs`` wants ``(T, 1)`` when N=1."""
    dtime = _matlab_datenum(times)
    sss_b = np.asarray(sss, dtype=np.float64).reshape(-1)
    sst_b = np.asarray(sst, dtype=np.float64).reshape(-1)
    ssh_b = np.asarray(ssh, dtype=np.float64).reshape(-1)
    if sss_b.shape[0] == 1:
        sss_b = sss_b.reshape(1, 1)
        sst_b = sst_b.reshape(1, 1)
        ssh_b = ssh_b.reshape(1, 1)
    batch = prepare_inputs(dtime, lat, lon, sss_b, sst_b, ssh_b, _SAT_PARAMS)
    return np.asarray(
        batch.detach().cpu().numpy() if isinstance(batch, torch.Tensor) else batch,
        dtype=np.float32,
    )


def _inject_enso(
    x9: np.ndarray,
    times: Sequence[datetime],
    spec_index_dir: Path,
    input_params: dict[str, bool],
) -> np.ndarray:
    ensure_v2_on_path()
    from base.split_utils import dates_to_juld
    from preproc.enso import inject_enso_columns

    iso = [t.strftime("%Y-%m-%d") for t in times]
    juld = dates_to_juld(iso, dataset_tag="argo_v2")
    ip = dict(input_params)
    ip.setdefault("oni", True)
    ip.setdefault("roni", True)
    return inject_enso_columns(
        x9,
        juld,
        dataset_tag="argo_v2",
        input_params=ip,
        n_enc_base=6,
        index_dir=str(spec_index_dir),
        expected_dim=None,
    )


def _operator_names() -> tuple[str, ...]:
    global _OP_NAME_CACHE
    if _OP_NAME_CACHE is None:
        ensure_v2_on_path()
        from preproc.export_heave_ablation_cache import OP_NAMES

        _OP_NAME_CACHE = tuple(OP_NAMES)
    return _OP_NAME_CACHE


def _canonicalize_plane(
    lats: np.ndarray, lons: np.ndarray, field: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(lat, lon, field)`` with increasing axes and shape ``(nlat, nlon)``."""
    lat = np.asarray(lats, dtype=np.float64).reshape(-1)
    lon = np.asarray(lons, dtype=np.float64).reshape(-1)
    arr = np.asarray(field, dtype=np.float32)
    while arr.ndim > 2 and 1 in arr.shape:
        arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise V2UnavailableError(f"ops plane is not 2-D: shape {arr.shape}")
    if arr.shape == (lon.size, lat.size) and arr.shape != (lat.size, lon.size):
        arr = arr.T
    if arr.shape != (lat.size, lon.size):
        raise V2UnavailableError(
            f"ops plane shape {arr.shape} != ({lat.size}, {lon.size})"
        )
    lon = np.where(lon > 180.0, lon - 360.0, lon)
    if lat.size >= 2 and lat[0] > lat[-1]:
        lat = lat[::-1].copy()
        arr = arr[::-1, :]
    if lon.size >= 2 and lon[0] > lon[-1]:
        lon = lon[::-1].copy()
        arr = arr[:, ::-1]
    if lat.size < 3 or lon.size < 3:
        raise V2UnavailableError(
            f"ops plane too small for derivatives: {(lat.size, lon.size)}"
        )
    return lat, lon, np.ascontiguousarray(arr)


def _grid_step_deg(lats: np.ndarray, lons: np.ndarray) -> float:
    dlat = float(np.mean(np.abs(np.diff(lats)))) if lats.size > 1 else 0.05
    dlon = float(np.mean(np.abs(np.diff(lons)))) if lons.size > 1 else 0.05
    return float(max(min(dlat, dlon), 1.0e-6))


def _ops_bbox(lat: np.ndarray, lon: np.ndarray) -> tuple[float, float, float, float]:
    bbox = (
        float(np.min(lat)) - OPS_HALO_DEG,
        float(np.max(lat)) + OPS_HALO_DEG,
        float(np.min(lon)) - OPS_HALO_DEG,
        float(np.max(lon)) + OPS_HALO_DEG,
    )
    rounded = _round_bbox(bbox)
    if rounded is None:
        raise V2UnavailableError("ops bbox is empty")
    return rounded


def _load_channel_plane(
    channel: str, day: datetime, bbox: tuple[float, float, float, float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load one SAT-archive plane. Missing file is a 503, not a zero fill."""
    y, m, d = day.year, day.month, day.day
    try:
        if channel == "sst":
            raw = cached_sst_arrays(CFG.SST_ROOT, y, m, d, bbox)
        elif channel == "sss":
            raw = cached_sss_arrays(CFG.SSS_ROOT, y, m, d, bbox)
        elif channel == "ssh":
            raw = cached_aviso_arrays(_select_aviso_root(day), y, m, d, bbox)
        else:
            raise V2UnavailableError(f"unknown ops channel {channel!r}")
    except V2UnavailableError:
        raise
    except Exception as exc:
        raise V2UnavailableError(
            f"Satellite data unavailable: {channel} {day:%Y-%m-%d}: {exc}"
        ) from exc
    if raw is None or raw[0] is None or raw[2] is None:
        raise V2UnavailableError(
            f"Satellite data unavailable: {channel} {day:%Y-%m-%d} returned None"
        )
    return _canonicalize_plane(raw[0], raw[1], raw[2])


def _align_to_grid(
    lats: np.ndarray,
    lons: np.ndarray,
    field: np.ndarray,
    ref_lats: np.ndarray,
    ref_lons: np.ndarray,
) -> np.ndarray:
    """Resample ``field`` onto the request-day grid. Does not fill NaNs."""
    if (
        lats.shape == ref_lats.shape
        and lons.shape == ref_lons.shape
        and np.allclose(lats, ref_lats)
        and np.allclose(lons, ref_lons)
    ):
        return field
    interp = RegularGridInterpolator(
        (lats, lons),
        np.asarray(field, dtype=np.float64),
        bounds_error=False,
        fill_value=np.nan,
    )
    lat_g, lon_g = np.meshgrid(ref_lats, ref_lons, indexing="ij")
    out = interp(np.column_stack((lat_g.ravel(), lon_g.ravel())))
    return np.asarray(out, dtype=np.float32).reshape(ref_lats.size, ref_lons.size)


def _tendency_stack(
    channel: str,
    end_day: datetime,
    bbox: tuple[float, float, float, float],
    window: int,
    ref_lats: np.ndarray,
    ref_lons: np.ndarray,
) -> np.ndarray:
    """``window`` daily planes ending at ``end_day`` (oldest first)."""
    planes: list[np.ndarray] = []
    for k in range(window - 1, -1, -1):
        day = end_day - timedelta(days=k)
        glat, glon, field = _load_channel_plane(channel, day, bbox)
        planes.append(_align_to_grid(glat, glon, field, ref_lats, ref_lons))
    return np.stack(planes, axis=0)


def _group_by_day(
    times: Sequence[datetime],
) -> list[tuple[datetime, np.ndarray]]:
    groups: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for i, t in enumerate(times):
        groups[(t.year, t.month, t.day)].append(i)
    return [
        (datetime(y, m, d), np.asarray(idx, dtype=int))
        for (y, m, d), idx in groups.items()
    ]


def sample_ops_or_503(
    times: Sequence[datetime], lat: np.ndarray, lon: np.ndarray
) -> np.ndarray:
    """Compute the 19 heave-ops columns from the live SAT archive.

    Uses the same ``apply_operator`` / bilinear sample path as training, on
    MUR / SMAP / AVISO planes instead of ``gom_cube.zarr``. Missing days or
    non-finite samples are 503. Never zero-filled.

    Args:
        times: Profile timestamps.
        lat: Latitudes.
        lon: Longitudes.

    Returns:
        Array of shape ``(N, 19)`` in ``OP_NAMES`` order.

    Raises:
        V2UnavailableError: Archive hole, empty neighborhood, or NaN sample.
    """
    ensure_v2_on_path()
    from preproc.features.operators import GRAVITY, apply_operator
    from preproc.features.sampler import (
        EARTH_ROTATION_RATE,
        build_bilinear_weights,
        expand_feature_names,
        resolve_scale_deg,
        sample_plane,
    )

    names = _operator_names()
    expanded = expand_feature_names(OPS_FEATURE_SPEC)
    expanded_names = [e[0] for e in expanded]
    missing = [n for n in names if n not in expanded_names]
    if missing:
        raise V2UnavailableError(f"ops feature spec missing {missing}")

    lat = np.asarray(lat, dtype=np.float64).reshape(-1)
    lon = np.asarray(lon, dtype=np.float64).reshape(-1)
    n = int(lat.shape[0])
    if n == 0 or lon.shape[0] != n or len(list(times)) != n:
        raise V2UnavailableError("ops sample size mismatch")

    bbox = _ops_bbox(lat, lon)
    values = np.full((n, len(names)), np.nan, dtype=np.float32)
    valid = np.zeros((n, len(names)), dtype=bool)
    name_col = {n: i for i, n in enumerate(names)}

    plane_cache: dict[
        tuple[str, datetime], tuple[np.ndarray, np.ndarray, np.ndarray]
    ] = {}
    derived: dict[tuple[Any, ...], Any] = {}
    stack_cache: dict[tuple[str, datetime, int], np.ndarray] = {}
    weights_cache: dict[tuple[str, datetime], Any] = {}

    def _field(ch: str, day: datetime) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        key = (ch, day)
        if key not in plane_cache:
            plane_cache[key] = _load_channel_plane(ch, day, bbox)
        return plane_cache[key]

    def _weights(ch: str, day: datetime, glat: np.ndarray, glon: np.ndarray) -> Any:
        key = (ch, day)
        if key not in weights_cache:
            weights_cache[key] = build_bilinear_weights(glat, glon, lat, lon)
        return weights_cache[key]

    for day, idx_arr in _group_by_day(times):
        for feat_name, op, ch, scale_lbl, param in expanded:
            if feat_name not in name_col:
                continue
            j = name_col[feat_name]
            glat, glon, field = _field(ch, day)
            grid_step = _grid_step_deg(glat, glon)
            w_full = _weights(ch, day, glat, glon)
            w_group = w_full[idx_arr]

            if op == "tendency":
                window = int(param)
                dkey: tuple[Any, ...] = ("tendency", ch, day, window)
            elif op in ("grad", "geo_uv"):
                dkey = ("grad", ch, day, scale_lbl)
            else:
                dkey = (op, ch, day, scale_lbl)

            if dkey not in derived:
                if op == "tendency":
                    window = int(param)
                    skey = (ch, day, window)
                    if skey not in stack_cache:
                        stack_cache[skey] = _tendency_stack(
                            ch, day, bbox, window, glat, glon
                        )
                    derived[dkey] = apply_operator(
                        "tendency",
                        field,
                        stack=stack_cache[skey],
                        window_days=window,
                    )
                else:
                    scale_deg = resolve_scale_deg(ch, scale_lbl, OPS_FEATURE_SPEC)
                    if op in ("grad", "geo_uv"):
                        derived[dkey] = apply_operator(
                            "grad",
                            field,
                            lats=glat,
                            lons=glon,
                            scale_deg=scale_deg,
                            grid_step_deg=grid_step,
                        )
                    elif op == "laplacian":
                        derived[dkey] = apply_operator(
                            "laplacian",
                            field,
                            lats=glat,
                            lons=glon,
                            scale_deg=scale_deg,
                            grid_step_deg=grid_step,
                        )
                    else:
                        raise V2UnavailableError(f"unknown ops operator {op}")

            if op == "grad":
                gx, gy = derived[dkey]
                plane_pick = gx if ".grad_x@" in feat_name else gy
                sampled, val_w = sample_plane(w_group, plane_pick)
                values[idx_arr, j] = sampled
                valid[idx_arr, j] = val_w
            elif op == "geo_uv":
                gx, gy = derived[dkey]
                lat_group = lat[idx_arr]
                coriolis = 2.0 * EARTH_ROTATION_RATE * np.sin(np.radians(lat_group))
                coriolis = np.where(np.abs(coriolis) < 1e-8, 1e-8, coriolis)
                if ".geo_u@" in feat_name:
                    gy_s, gy_valid = sample_plane(w_group, gy)
                    values[idx_arr, j] = ((-GRAVITY / coriolis) * gy_s).astype(
                        np.float32
                    )
                    valid[idx_arr, j] = gy_valid
                else:
                    gx_s, gx_valid = sample_plane(w_group, gx)
                    values[idx_arr, j] = ((GRAVITY / coriolis) * gx_s).astype(
                        np.float32
                    )
                    valid[idx_arr, j] = gx_valid
            else:
                sampled, val_w = sample_plane(w_group, derived[dkey])
                values[idx_arr, j] = sampled
                valid[idx_arr, j] = val_w

    if not (valid.all() and np.isfinite(values).all()):
        raise V2UnavailableError(
            "Satellite data unavailable: operator sample is non-finite "
            "(not zero-filled)"
        )
    return values


def build_v2_inputs(
    model: str,
    times: Sequence[datetime],
    lat: np.ndarray,
    lon: np.ndarray,
    sss: np.ndarray,
    sst: np.ndarray,
    ssh: np.ndarray,
) -> np.ndarray:
    """Assemble the cell feature matrix from SAT (+ ENSO / sat-archive ops).

    Args:
        model: ``A_CRPS``, ``HeaveFast``, or ``ops``.
        times: Profile times.
        lat: Latitudes.
        lon: Longitudes.
        sss: 1-D SSS.
        sst: 1-D SST in Kelvin (``prepare_inputs`` subtracts 273.15).
        ssh: 1-D SSH.

    Returns:
        Float32 array of shape ``(N, input_dim)``.
    """
    spec = load_serve_spec().models[model]
    x = _sat_batch(times, lat, lon, sss, sst, ssh)
    if model == "A_CRPS":
        if x.shape[1] != spec.input_dim:
            raise V2UnavailableError(
                f"A_CRPS input width {x.shape[1]} != {spec.input_dim}"
            )
        return x
    ip = {
        "timecos": True,
        "timesin": True,
        "latcos": True,
        "latsin": True,
        "loncos": True,
        "lonsin": True,
        "oni": True,
        "roni": True,
        "sss": True,
        "sst": True,
        "ssh": True,
        "sat": True,
    }
    x = _inject_enso(x, times, spec.index_dir, ip)
    if model == "HeaveFast":
        if x.shape[1] != spec.input_dim:
            raise V2UnavailableError(
                f"HeaveFast input width {x.shape[1]} != {spec.input_dim}"
            )
        return np.asarray(x, dtype=np.float32)
    ops = sample_ops_or_503(times, lat, lon)
    x = np.concatenate([np.asarray(x, dtype=np.float32), ops], axis=1)
    if x.shape[1] != spec.input_dim:
        raise V2UnavailableError(f"ops input width {x.shape[1]} != {spec.input_dim}")
    return x
