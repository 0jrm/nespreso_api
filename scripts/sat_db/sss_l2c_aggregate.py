#!/usr/bin/env python3
"""Build L3-like SMAP SSS fields from L2C FINAL/NRT granules for a target date."""
from __future__ import annotations

import logging
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger("satdb")

# Standard L3 0.25° grid (matches RSS L3 V6)
L3_LAT = np.arange(-89.875, 90.0, 0.25, dtype=np.float64)
L3_LON = np.arange(0.125, 360.0, 0.25, dtype=np.float64)

# GoM bbox used to filter L2C searches (lon lat order for earthaccess bounding_box)
GOM_BBOX = (-98.0, 18.0, -81.0, 31.0)  # lon_min, lat_min, lon_max, lat_max


def l2c_output_path(sss_root: str, date_: datetime) -> Path:
    year_dir = Path(sss_root) / f"{date_.year:04d}"
    doy = date_.timetuple().tm_yday
    return year_dir / f"RSS_smap_SSS_L3_8day_running_{date_.year}_{doy:03d}_FNL_v06.0_l2c.nc"


def _bin_points(
    lats: np.ndarray,
    lons: np.ndarray,
    vals: np.ndarray,
    sum_grid: np.ndarray,
    cnt_grid: np.ndarray,
) -> None:
    """Accumulate points onto L3 lat/lon grids (lon 0..360)."""
    lons = np.asarray(lons, dtype=np.float64)
    lats = np.asarray(lats, dtype=np.float64)
    vals = np.asarray(vals, dtype=np.float64)
    lons = np.where(lons < 0, lons + 360.0, lons)
    valid = np.isfinite(lats) & np.isfinite(lons) & np.isfinite(vals)
    valid &= (vals > 0) & (vals < 45)
    if not np.any(valid):
        return
    lats = lats[valid]
    lons = lons[valid]
    vals = vals[valid]
    # indices into 0.25° grids
    i_lat = np.rint((lats - (-89.875)) / 0.25).astype(np.int64)
    i_lon = np.rint((lons - 0.125) / 0.25).astype(np.int64)
    ok = (i_lat >= 0) & (i_lat < len(L3_LAT)) & (i_lon >= 0) & (i_lon < len(L3_LON))
    i_lat, i_lon, vals = i_lat[ok], i_lon[ok], vals[ok]
    np.add.at(sum_grid, (i_lat, i_lon), vals)
    np.add.at(cnt_grid, (i_lat, i_lon), 1)


def _ingest_l2c_file(path: str, sum_grid: np.ndarray, cnt_grid: np.ndarray) -> int:
    import xarray as xr

    ds = xr.open_dataset(path, decode_times=False)
    try:
        if "sss_smap" not in ds or "cellat" not in ds or "cellon" not in ds:
            return 0
        sss = ds["sss_smap"].values
        lat = ds["cellat"].values
        lon = ds["cellon"].values
        # Average over look dimension if present
        if sss.ndim == 3:
            sss = np.nanmean(sss, axis=-1)
            lat = np.nanmean(lat, axis=-1)
            lon = np.nanmean(lon, axis=-1)
        before = int(cnt_grid.sum())
        _bin_points(lat.ravel(), lon.ravel(), sss.ravel(), sum_grid, cnt_grid)
        return int(cnt_grid.sum()) - before
    finally:
        ds.close()


def _search_l2c(short_name: str, start: datetime, end: datetime, count: int = 2000):
    import earthaccess

    return earthaccess.search_data(
        short_name=short_name,
        temporal=(
            start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        ),
        bounding_box=GOM_BBOX,
        count=count,
    )


def build_l2c_aggregate(
    sss_root: str,
    date_: datetime,
    *,
    template_l3: Optional[str] = None,
    max_granules: int = 24,
) -> Optional[str]:
    """
    Create an L3-like SSS file from L2C FINAL (preferred) + NRT fill for an 8-day
    window centered on date_. Returns output path or None.
    """
    import earthaccess
    import xarray as xr

    out = l2c_output_path(sss_root, date_)
    if out.exists():
        logger.info(f"L2C aggregate exists for {date_.date()} -> {out}")
        return str(out)

    out.parent.mkdir(parents=True, exist_ok=True)
    half = timedelta(days=3, hours=12)
    start = date_.replace(hour=0, minute=0, second=0, microsecond=0) - half
    end = date_.replace(hour=0, minute=0, second=0, microsecond=0) + half
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
        end = end.replace(tzinfo=timezone.utc)

    earthaccess.login(strategy="environment")
    sources: List[str] = []
    granules = []
    for short in ("SMAP_RSS_L2_SSS_V6", "SMAP_RSS_L2_SSS_NRT_V6"):
        try:
            found = _search_l2c(short, start, end)
        except Exception as e:
            logger.warning(f"L2C search failed for {short}: {e}")
            found = []
        if found:
            granules.extend(found[: max(0, max_granules - len(granules))])
            sources.append(short)
        if len(granules) >= max_granules:
            break

    if not granules:
        logger.info(f"No L2C granules for SSS fallback on {date_.date()}")
        return None

    sum_grid = np.zeros((len(L3_LAT), len(L3_LON)), dtype=np.float64)
    cnt_grid = np.zeros((len(L3_LAT), len(L3_LON)), dtype=np.int64)
    n_files = 0
    n_points = 0

    with tempfile.TemporaryDirectory(dir=str(out.parent)) as tmp:
        for g in granules:
            try:
                paths = earthaccess.download(g, local_path=tmp)
            except Exception as e:
                logger.warning(f"L2C download failed: {e}")
                continue
            if not paths:
                continue
            try:
                added = _ingest_l2c_file(str(paths[0]), sum_grid, cnt_grid)
                n_files += 1
                n_points += added
            except Exception as e:
                logger.warning(f"L2C ingest failed for {paths[0]}: {e}")
            finally:
                try:
                    os.remove(paths[0])
                except Exception:
                    pass

    if n_points == 0 or not np.any(cnt_grid > 0):
        logger.info(f"L2C aggregate produced no valid points for {date_.date()}")
        return None

    mean = np.full_like(sum_grid, np.nan, dtype=np.float32)
    mask = cnt_grid > 0
    mean[mask] = (sum_grid[mask] / cnt_grid[mask]).astype(np.float32)

    # Optional: copy non-SSS scaffold from template (coords only needed)
    time_val = np.array([np.datetime64(date_.strftime("%Y-%m-%dT12:00:00"))])
    ds = xr.Dataset(
        data_vars={
            "sss_smap": (("lat", "lon"), mean),
            "nobs": (("lat", "lon"), cnt_grid.astype(np.float32)),
        },
        coords={
            "lat": ("lat", L3_LAT.astype(np.float32)),
            "lon": ("lon", L3_LON.astype(np.float32)),
            "time": ("time", time_val),
        },
        attrs={
            "title": "SMAP SSS 8-day aggregate from L2C (NeSPReSO fallback)",
            "source": "L2C_8day_aggregate",
            "source_short_names": ",".join(sources),
            "aggregation_window": f"{start.isoformat()} .. {end.isoformat()}",
            "warning": "Not an official RSS L3 product; synthesized for NeSPReSO continuity",
            "n_l2c_files": n_files,
            "n_points": int(n_points),
            "history": f"Created {datetime.now(timezone.utc).isoformat()} by sss_l2c_aggregate.py",
        },
    )
    ds["sss_smap"].attrs.update({"long_name": "SMAP sea surface salinity", "units": "psu"})
    tmp_out = out.with_suffix(".nc.tmp")
    ds.to_netcdf(tmp_out)
    ds.close()
    os.replace(tmp_out, out)
    logger.info(
        f"L2C aggregate wrote {out} files={n_files} points={n_points} "
        f"coverage_cells={int(mask.sum())}"
    )
    return str(out)
