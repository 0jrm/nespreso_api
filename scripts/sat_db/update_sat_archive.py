#!/usr/bin/env python3
#usage: python update_sat_archive.py --bootstrap --start-year 1992 --end-year 2025
#examples:
#  - Default incremental update (rolling window + last month AVISO):
#      python update_sat_archive.py
#  - Rescan a large rolling window in one run (days):
#      python update_sat_archive.py --rescan-days 365
#  - One-time bootstrap (seed DB from filesystem only; no downloads):
#      python update_sat_archive.py --bootstrap --start-year 1993 --end-year 2025
#  - Historical backfill (attempt downloads for a date range):
#      python update_sat_archive.py --backfill --start-date 1993-01-01 --end-date 2025-12-31 --sources sst,sss,aviso
"""
update_sat_archive.py

End-to-end, idempotent updater for your satellite archive with **stateful control**
so we avoid rescanning and pointless retries on every run.

Design:
- Single SQLite file that tracks, per UTC date, which sources (SST, SSS, AVISO) are available,
  plus retry metadata (try_count, last_error, backoff schedule).
- Daily run targets: yesterday for SST/SSS, and the previous month for AVISO monthly aggregate.
- Exponential backoff for transient "not available yet" conditions. We don't hammer providers.
- Bootstrap mode can scan existing filesystem once to seed the DB, then normal runs are incremental.
- CLI offers: default update, --bootstrap (one-time), --report (human summary), --rescan-days N.

You can safely invoke this via cron or a systemd timer (run_update.sh included).

Environment variables (override defaults):
  SATDB_DB=/Net/work/ozavala/DATA/SubSurfaceFields/NeSPReSO/satdb/state.db
  SATDB_SST_ROOT=/Net/work/ozavala/DATA/GOFFISH/SST/OISST/
  SATDB_SSS_ROOT=/Net/work/ozavala/DATA/GOFFISH/SSS/SMAP_Global/
  SATDB_AVISO_ROOT=/Net/work/ozavala/DATA/GOFFISH/AVISO/GoM/
  SATDB_LOG=/unity/g2/jmiranda/nespreso_api/scripts/sat_db/update.log

Requires: earthaccess, copernicusmarine, xarray, netCDF4, h5netcdf

usage:
python update_sat_archive.py --reset-attempts --backfill --start-date 2025-09-03 --end-date 2025-09-10 --sources sss,aviso,sst --log-stdout
"""

import os
import sys
import gc
import time
import json
import logging
import sqlite3
import shutil
import tempfile
from pathlib import Path
from typing import Optional, Tuple, Dict, Set
from datetime import datetime, timedelta, timezone

# ---------------------------- config ----------------------------
def env(name: str, default: str) -> str:
    return os.environ.get(name, default)

DB_PATH     = Path(env("SATDB_DB", "/Net/work/ozavala/DATA/SubSurfaceFields/NeSPReSO/satdb/state.db"))
SST_ROOT    = Path(env("SATDB_SST_ROOT", "/Net/work/ozavala/DATA/GOFFISH/SST/OISST/"))
SSS_ROOT    = Path(env("SATDB_SSS_ROOT", "/Net/work/ozavala/DATA/GOFFISH/SSS/SMAP_Global/"))
AVISO_ROOT  = Path(env("SATDB_AVISO_ROOT", "/Net/work/ozavala/DATA/GOFFISH/AVISO/GoM/"))
LOG_PATH    = Path(env("SATDB_LOG", "/unity/g2/jmiranda/nespreso_api/scripts/sat_db/update.log"))

# Alternate AVISO source (post-2024-10): CMEMS GLOBAL_ANALYSISFORECAST PHY ANFC monthly subsets
ANFC_ROOT            = Path(env("SATDB_ANFC_ROOT", "/Net/work/ozavala/DATA/GOFFISH/AVISO/GoM/CMEMS_GLOBAL_PHY_ANFC"))
ANFC_DATASET_ID      = env("SATDB_ANFC_DATASET_ID", "cmems_mod_glo_phy_anfc_0.083deg_P1D-m")
ANFC_DATASET_VERSION = env("SATDB_ANFC_VERSION", "")  # optional; empty means latest
ANFC_VARIABLE        = env("SATDB_ANFC_VARIABLE", "zos")  # SSH variable in ANFC
ANFC_BBOX            = env("SATDB_ANFC_BBOX", "-98,-81,18,31")  # lon_min,lon_max,lat_min,lat_max

# Conversion to approximate DUACS ADT from ANFC SSH (zos)
ADT_FROM_SSH_SLOPE     = float(env("SATDB_ADT_FROM_SSH_SLOPE", "1.015492"))
ADT_FROM_SSH_INTERCEPT = float(env("SATDB_ADT_FROM_SSH_INTERCEPT", "0.423671"))

# Backoff parameters
MAX_TRIES          = int(env("SATDB_MAX_TRIES", "8"))
BASE_BACKOFF_MIN   = float(env("SATDB_BASE_BACKOFF_MIN", "30"))   # 30 minutes
BACKOFF_FACTOR     = float(env("SATDB_BACKOFF_FACTOR", "2.0"))    # exponential

# How far back to consider for "late-arriving" daily products during normal runs
DAILY_LOOKBACK_DAYS = int(env("SATDB_DAILY_LOOKBACK_DAYS", "10"))

# ---------------------------- helpers ----------------------------

def _assert_writable(p: Path):
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    testfile = p / ".writetest.tmp"
    try:
        with open(testfile, "w") as f:
            f.write("ok")
    finally:
        if testfile.exists():
            testfile.unlink()

def _atomic_move(src: Path, dst: Path):
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.replace(src, dst)

def _parse_ymd(date_str: str) -> datetime:
    """Parse YYYY-MM-DD as UTC midnight."""
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return dt.replace(tzinfo=timezone.utc, hour=0, minute=0, second=0, microsecond=0)

def _level_from_string(level_str: str) -> int:
    value = (level_str or "").strip().upper()
    return {
        "CRITICAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
        "WARN": logging.WARNING,
        "WARNING": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
        "NOTSET": logging.NOTSET,
    }.get(value, logging.INFO)

class _UTCFormatter(logging.Formatter):
    converter = time.gmtime

def setup_logging(log_path: Optional[str], level: str = "INFO", also_stdout: bool = False):
    """Configure module logger. Safe to call multiple times."""
    path = Path(log_path) if log_path else LOG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    _assert_writable(path.parent)

    logger = logging.getLogger("satdb")
    logger.setLevel(_level_from_string(level))

    # Remove existing handlers to avoid duplicates on multiple invocations
    for h in list(logger.handlers):
        logger.removeHandler(h)

    fmt = _UTCFormatter("%(asctime)sZ %(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S")

    fh = logging.FileHandler(path, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(_level_from_string(level))
    logger.addHandler(fh)

    if also_stdout:
        sh = logging.StreamHandler(stream=sys.stdout)
        sh.setFormatter(fmt)
        sh.setLevel(_level_from_string(level))
        logger.addHandler(sh)

    # Quiet noisy third-party libraries
    logging.getLogger("earthaccess").setLevel(logging.WARNING)
    logging.getLogger("copernicusmarine").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)

    logger.debug(f"Logging initialized at {path} level={level} also_stdout={also_stdout}")

def _parse_bbox(bbox_str: str) -> Tuple[float, float, float, float]:
    parts = [p.strip() for p in (bbox_str or "").split(",")]
    if len(parts) != 4:
        raise ValueError("bbox must have 4 comma-separated numbers: lon_min,lon_max,lat_min,lat_max")
    lon_min, lon_max, lat_min, lat_max = [float(p) for p in parts]
    return lon_min, lon_max, lat_min, lat_max

# ---------------------------- external deps ----------------------------
# We import lazily so that --report, --bootstrap can run without network libs if needed.
earthaccess = None
copernicusmarine = None
xr = None

def _lazy_import_ingest_libs():
    global earthaccess, copernicusmarine, xr
    if earthaccess is None:
        import earthaccess as _ea
        earthaccess = _ea
    if copernicusmarine is None:
        import copernicusmarine as _cm
        copernicusmarine = _cm
    if xr is None:
        import xarray as _xr
        xr = _xr

# ---------------------------- ensure_* (from your earlier code) ----------------------------

def ensure_sst_available(sst_root: str, date_: datetime) -> Optional[str]:
    """
    GHRSST / MUR (daily). Publishes:
    <SST_ROOT>/<YYYY>/<YYYYMMDD>090000-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1_subset.nc
    """
    _lazy_import_ingest_libs()
    logger = logging.getLogger("satdb")
    from datetime import timedelta
    year_dir  = os.path.join(sst_root, f"{date_.year:04d}")
    fname     = f"{date_.strftime('%Y%m%d')}090000-" \
                "JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1_subset.nc"
    final     = os.path.join(year_dir, fname)

    if os.path.exists(final):
        logger.debug(f"SST exists for {date_.date()} -> {final}")
        return final

    if not os.path.exists(year_dir):
        os.makedirs(year_dir)

    _assert_writable(Path(year_dir))

    t0 = date_.strftime("%Y-%m-%dT09:00:00Z")
    start_time = (date_ - timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%SZ")
    end_time = date_.strftime("%Y-%m-%dT%H:%M:%SZ")
    logger.debug(f"SST search {start_time}..{end_time} short_name=MUR-JPL-L4-GLOB-v4.1")
    results = earthaccess.search_data(
        short_name="MUR-JPL-L4-GLOB-v4.1",
        temporal=(start_time, end_time)
    )

    if not results:
        logger.info(f"SST not available for {t0}")
        return None

    with tempfile.TemporaryDirectory(dir=year_dir) as tmp:
        tmpfile = earthaccess.download(results[0], local_path=tmp)[0]
        os.replace(tmpfile, final)

    logger.info(f"SST downloaded {final}")
    return final

def ensure_sss_available(sss_root: str, date_: datetime) -> Optional[str]:
    """
    SMAP SSS (8-day running mean). Publishes:
    <SSS_ROOT>/<YYYY>/RSS_smap_SSS_L3_8day_running_<YYYY>_<DOY>_FNL_v06.0.nc
    """
    _lazy_import_ingest_libs()
    logger = logging.getLogger("satdb")
    from datetime import timedelta
    year_dir = os.path.join(sss_root, f"{date_.year:04d}")
    doy      = date_.timetuple().tm_yday
    fname    = f"RSS_smap_SSS_L3_8day_running_{date_.year}_{doy:03d}_FNL_v06.0.nc"
    final    = os.path.join(year_dir, fname)

    if os.path.exists(final):
        logger.debug(f"SSS exists for {date_.date()} -> {final}")
        return final

    if not os.path.exists(year_dir):
        os.makedirs(year_dir)

    _assert_writable(Path(year_dir))

    t0 = date_.strftime("%Y-%m-%dT12:00:00Z")
    start_time = (date_ - timedelta(days=8)).strftime("%Y-%m-%dT%H:%M:%SZ")
    end_time = date_.strftime("%Y-%m-%dT%H:%M:%SZ")
    logger.debug(f"SSS search {start_time}..{end_time} short_name=SMAP_RSS_L3_SSS_SMI_8DAY-RUNNINGMEAN_V6")
    results = earthaccess.search_data(
        short_name="SMAP_RSS_L3_SSS_SMI_8DAY-RUNNINGMEAN_V6",
        temporal=(start_time, end_time)
    )

    if not results:
        logger.info(f"SSS V6 not available for {t0}, trying V5...")
        results = earthaccess.search_data(
            short_name="SMAP_RSS_L3_SSS_SMI_8DAY-RUNNINGMEAN_V5",
            temporal=(start_time, end_time)
        )
        if not results:
            logger.info(f"SSS not available for {t0} (neither V6 nor V5)")
            return None

    with tempfile.TemporaryDirectory(dir=year_dir) as tmp:
        tmpfile = earthaccess.download(results[0], local_path=tmp)[0]
        os.replace(tmpfile, final)

    logger.info(f"SSS downloaded {final}")
    return final

def ensure_aviso_available(aviso_root: str, date_: datetime) -> Path:
    """
    AVISO / DUACS incremental monthly aggregate for the month containing *date_*.
    Downloads daily files incrementally and rebuilds monthly file each day.
    Writes <AVISO_ROOT>/<YYYY>-<MM>.nc (idempotent).
    """
    _lazy_import_ingest_libs()
    import copernicusmarine
    import xarray as xr
    logger = logging.getLogger("satdb")

    aviso_path = Path(aviso_root)
    month_tag = f"{date_.year}-{date_.month:02d}"
    final = aviso_path / f"{month_tag}.nc"

    # Determine days in month for completeness checks
    import calendar
    days_in_month = calendar.monthrange(date_.year, date_.month)[1]

    _assert_writable(aviso_path)

    # Decide source by month: DUACS (through 2024-10) vs ANFC (2024-11+)
    use_duacs = (date_.year < 2024) or (date_.year == 2024 and date_.month <= 10)

    if use_duacs:
        # If a monthly file exists and appears complete for DUACS, return early
        if final.exists():
            try:
                ds_existing = xr.open_dataset(final, decode_cf=True)
                n_time = int(ds_existing.sizes.get("time", 0))
                ds_existing.close()
                if n_time >= days_in_month:
                    logger.debug(f"AVISO monthly exists and complete {final} ({n_time}/{days_in_month} days)")
                    return final
                else:
                    logger.info(f"AVISO monthly exists but incomplete {final} ({n_time}/{days_in_month} days) -> will refresh")
            except Exception as _e:
                logger.warning(f"AVISO monthly exists at {final} but could not read: {_e} -> will refresh")
        # Legacy path: download daily DUACS and aggregate monthly, refreshing as new days arrive
        # Create daily downloads directory
        daily_dir = aviso_path / "daily_downloads" / month_tag
        daily_dir.mkdir(parents=True, exist_ok=True)

        tmpdir = tempfile.mkdtemp(dir=str(aviso_path))
        try:
            # Download newly available daily files directly into daily_dir (skip existing)
            existing_files = sorted(daily_dir.glob("*.nc"))
            existing_count = len(existing_files)
            if existing_count < days_in_month:
                pattern = f"*{date_.year}{date_.month:02d}??*.nc"  # YYYYMMDD
                logger.debug(f"AVISO DUACS fetching daily files for {month_tag} into {daily_dir} (have {existing_count}/{days_in_month})")
                copernicusmarine.get(
                    dataset_id=("cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D"),
                    filter=pattern,
                    output_directory=str(daily_dir),
                    overwrite=False
                )

            # Get all available daily files after fetch
            daily_files = sorted(daily_dir.glob("*.nc"))
            if not daily_files:
                logger.info(f"AVISO {month_tag}: no daily files available yet")
                raise FileNotFoundError(f"No DUACS files for {month_tag}")

            available_days = len(daily_files)

            logger.info(f"AVISO {month_tag}: {available_days}/{days_in_month} daily files available")

            # Build or refresh monthly file from daily files (partial or complete)
            ds = xr.open_mfdataset(
                daily_files, combine="nested", concat_dim="time",
                parallel=False, decode_times=False, engine="netcdf4"
            )
            ds = xr.decode_cf(ds)
            ds = ds.chunk({"time": -1, "latitude": 171, "longitude": 173})
            encoding = {v: {"zlib": True, "complevel": 0} for v in ds.data_vars}

            tmp_month = Path(tmpdir) / f"{month_tag}.nc"
            ds.to_netcdf(tmp_month, engine="h5netcdf", encoding=encoding)
            ds.close(); del ds
            gc.collect(); time.sleep(0.1)

            final.parent.mkdir(parents=True, exist_ok=True)
            _atomic_move(tmp_month, final)

            if available_days >= days_in_month:
                # Clean up daily files after successful monthly completion
                shutil.rmtree(daily_dir, ignore_errors=True)
                logger.info(f"AVISO {month_tag}: monthly file completed -> {final}")
            else:
                logger.info(f"AVISO {month_tag}: partial monthly file created/refreshed ({available_days}/{days_in_month} days) -> {final}")

        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        return final

    # New path (ANFC): subset monthly SSH, convert to ADT, and write compatible file
    tmpdir = tempfile.mkdtemp(dir=str(aviso_path))
    try:
        # Always fetch or refresh the ANFC monthly subset, then convert to ADT
        anfc_month = ANFC_ROOT / f"{month_tag}.nc"
        ANFC_ROOT.mkdir(parents=True, exist_ok=True)
        _assert_writable(ANFC_ROOT)

        # Compute month start/end
        month_start = date_.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        month_end = date_.replace(day=days_in_month, hour=23, minute=59, second=59, microsecond=0)

        lon_min, lon_max, lat_min, lat_max = _parse_bbox(ANFC_BBOX)

        subset_kwargs = dict(
            dataset_id=ANFC_DATASET_ID,
            variables=[ANFC_VARIABLE],
            minimum_longitude=lon_min,
            maximum_longitude=lon_max,
            minimum_latitude=lat_min,
            maximum_latitude=lat_max,
            start_datetime=month_start.strftime("%Y-%m-%dT%H:%M:%S"),
            end_datetime=month_end.strftime("%Y-%m-%dT%H:%M:%S"),
            output_filename=str(Path(tmpdir) / f"{month_tag}.raw.nc"),
        )
        if ANFC_DATASET_VERSION:
            subset_kwargs["dataset_version"] = ANFC_DATASET_VERSION

        logger.info(f"AVISO {month_tag}: downloading ANFC {ANFC_DATASET_ID} var={ANFC_VARIABLE} bbox={ANFC_BBOX}")
        copernicusmarine.subset(**subset_kwargs)
        os.replace(Path(tmpdir) / f"{month_tag}.raw.nc", anfc_month)
        logger.info(f"AVISO {month_tag}: wrote/updated ANFC subset {anfc_month}")

        # Open ANFC monthly and build ADT
        ds_src = xr.open_dataset(anfc_month, decode_cf=True)
        if ANFC_VARIABLE not in ds_src.data_vars and ANFC_VARIABLE not in ds_src.variables:
            ds_src.close()
            raise KeyError(f"Variable '{ANFC_VARIABLE}' not found in {anfc_month}")

        ssh = ds_src[ANFC_VARIABLE]
        adt = (ssh * ADT_FROM_SSH_SLOPE) + ADT_FROM_SSH_INTERCEPT
        adt = adt.rename("adt")

        # Variable attributes for compatibility
        adt.attrs.update({
            "long_name": "Absolute Dynamic Topography (converted from SSH)",
            "units": ssh.attrs.get("units", "m"),
            "source_variable": ANFC_VARIABLE,
            "conversion": f"ADT = {ADT_FROM_SSH_SLOPE} * SSH + {ADT_FROM_SSH_INTERCEPT}",
        })

        ds_out = adt.to_dataset(name="adt")

        # Global metadata
        history_line = f"Converted from {ANFC_VARIABLE} using ADT = {ADT_FROM_SSH_SLOPE} * SSH + {ADT_FROM_SSH_INTERCEPT} on {datetime.now(timezone.utc).isoformat()}"
        new_history = "; ".join([x for x in [ds_src.attrs.get("history"), history_line] if x])
        ds_out.attrs.update({
            "source_product": "CMEMS GLOBAL_ANALYSISFORECAST_PHY_001_024",
            "source_dataset_id": ANFC_DATASET_ID,
            "source_dataset_version": ANFC_DATASET_VERSION or "latest",
            "conversion_formula": f"ADT = {ADT_FROM_SSH_SLOPE} * SSH + {ADT_FROM_SSH_INTERCEPT}",
            "conversion_from": ANFC_VARIABLE,
            "history": new_history,
        })

        # Chunk and write with compression
        lat_dim = "latitude" if "latitude" in ds_out.dims else ("lat" if "lat" in ds_out.dims else None)
        lon_dim = "longitude" if "longitude" in ds_out.dims else ("lon" if "lon" in ds_out.dims else None)
        if lat_dim and lon_dim:
            ds_out = ds_out.chunk({"time": -1, lat_dim: min(171, ds_out.sizes[lat_dim]), lon_dim: min(173, ds_out.sizes[lon_dim])})
        encoding = {v: {"zlib": True, "complevel": 0} for v in ds_out.data_vars}

        tmp_month = Path(tmpdir) / f"{month_tag}.nc"
        ds_out.to_netcdf(tmp_month, engine="h5netcdf", encoding=encoding)
        ds_src.close(); ds_out.close(); del ds_src; del ds_out
        gc.collect(); time.sleep(0.1)

        _atomic_move(tmp_month, final)
        logger.info(f"AVISO {month_tag}: monthly file created from ANFC -> {final}")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return final

# ---------------------------- SQLite state ----------------------------

SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS days(
  date_ TEXT PRIMARY KEY,                 -- YYYY-MM-DD (UTC)
  sst   INTEGER NOT NULL DEFAULT 0,
  sss   INTEGER NOT NULL DEFAULT 0,
  aviso INTEGER NOT NULL DEFAULT 0,
  first_seen   TEXT NOT NULL,
  last_checked TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS attempts(
  date_ TEXT NOT NULL,
  source TEXT NOT NULL CHECK(source IN ('sst','sss','aviso')),
  try_count INTEGER NOT NULL DEFAULT 0,
  last_attempt TEXT NOT NULL,
  last_error TEXT,
  PRIMARY KEY(date_, source)
);
CREATE TABLE IF NOT EXISTS months(
  month TEXT PRIMARY KEY,                 -- YYYY-MM
  aviso INTEGER NOT NULL DEFAULT 0,
  first_seen   TEXT NOT NULL,
  last_checked TEXT NOT NULL
);
"""

def connect_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA busy_timeout=5000")
    conn.executescript(SCHEMA)
    logging.getLogger("satdb").debug(f"Connected to DB at {DB_PATH}")
    return conn

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def ensure_day_row(conn: sqlite3.Connection, d: datetime):
    date_key = d.date().isoformat()
    cur = conn.execute("SELECT 1 FROM days WHERE date_=?", (date_key,))
    if cur.fetchone() is None:
        conn.execute(
            "INSERT INTO days(date_, first_seen, last_checked) VALUES(?,?,?)",
            (date_key, _now_iso(), _now_iso())
        )
    else:
        conn.execute("UPDATE days SET last_checked=? WHERE date_=?", (_now_iso(), date_key))

def record_success(conn: sqlite3.Connection, d: datetime, source: str):
    date_key = d.date().isoformat()
    ensure_day_row(conn, d)
    if source == "aviso":
        # Mark the containing month
        mkey = f"{d.year:04d}-{d.month:02d}"
        cur = conn.execute("SELECT 1 FROM months WHERE month=?", (mkey,)).fetchone()
        if cur is None:
            conn.execute(
                "INSERT INTO months(month, aviso, first_seen, last_checked) VALUES(?,?,?,?)",
                (mkey, 1, _now_iso(), _now_iso())
            )
        else:
            conn.execute("UPDATE months SET aviso=1, last_checked=? WHERE month=?", (_now_iso(), mkey))
        # Also mark day-level 'aviso' to 1 so intersection queries are trivial
        conn.execute("UPDATE days SET aviso=1 WHERE date_=?", (date_key,))
    else:
        conn.execute(f"UPDATE days SET {source}=1 WHERE date_=?", (date_key,))
    # Reset attempts row on success
    conn.execute(
        "INSERT INTO attempts(date_, source, try_count, last_attempt, last_error) VALUES(?,?,?,?,NULL) "
        "ON CONFLICT(date_,source) DO UPDATE SET try_count=0, last_attempt=excluded.last_attempt, last_error=NULL",
        (date_key, source, 0, _now_iso())
    )
    logging.getLogger("satdb").info(f"SUCCESS {source} {date_key}")

def record_attempt(conn: sqlite3.Connection, d: datetime, source: str, error: Optional[str] = None):
    date_key = d.date().isoformat()
    ensure_day_row(conn, d)
    row = conn.execute(
        "SELECT try_count FROM attempts WHERE date_=? AND source=?", (date_key, source)
    ).fetchone()
    if row is None:
        conn.execute(
            "INSERT INTO attempts(date_, source, try_count, last_attempt, last_error) VALUES(?,?,?,?,?)",
            (date_key, source, 1, _now_iso(), error)
        )
        new_try_count = 1
    else:
        try_count = int(row[0]) + 1
        conn.execute(
            "UPDATE attempts SET try_count=?, last_attempt=?, last_error=? WHERE date_=? AND source=?",
            (try_count, _now_iso(), error, date_key, source)
        )
        new_try_count = try_count
    if error:
        logging.getLogger("satdb").warning(f"ATTEMPT {source} {date_key} try={new_try_count} error={error}")
    else:
        logging.getLogger("satdb").info(f"ATTEMPT {source} {date_key} try={new_try_count}")

def should_retry(conn: sqlite3.Connection, d: datetime, source: str) -> bool:
    """Exponential backoff gate based on last attempt + try_count."""
    date_key = d.date().isoformat()
    row = conn.execute(
        "SELECT try_count, last_attempt FROM attempts WHERE date_=? AND source=?",
        (date_key, source)
    ).fetchone()
    if row is None:
        logging.getLogger("satdb").debug(f"RETRY {source} {date_key}: no prior attempts -> yes")
        return True  # never tried
    try_count, last_attempt = int(row[0]), row[1]
    if try_count >= MAX_TRIES:
        logging.getLogger("satdb").info(f"RETRY {source} {date_key}: max tries {MAX_TRIES} reached -> no")
        return False
    try:
        last_dt = datetime.fromisoformat(last_attempt)
    except Exception:
        logging.getLogger("satdb").debug(f"RETRY {source} {date_key}: bad last_attempt -> yes")
        return True
    # backoff minutes = BASE * FACTOR^(try_count-1)
    wait_min = BASE_BACKOFF_MIN * (BACKOFF_FACTOR ** max(try_count-1, 0))
    ok = datetime.now(timezone.utc) >= last_dt + timedelta(minutes=wait_min)
    logging.getLogger("satdb").debug(
        f"RETRY {source} {date_key}: try={try_count} last={last_attempt} wait_min={wait_min:.2f} -> {'yes' if ok else 'no'}"
    )
    return ok

# ---------------------------- filesystem seeding (bootstrap) ----------------------------

def _exists_sst(sst_root: Path, d: datetime) -> bool:
    year_dir = sst_root / f"{d.year:04d}"
    fname = f"{d.strftime('%Y%m%d')}090000-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1_subset.nc"
    return (year_dir / fname).exists()

def _exists_sss(sss_root: Path, d: datetime) -> bool:
    year_dir = sss_root / f"{d.year:04d}"
    doy = d.timetuple().tm_yday
    fname = f"RSS_smap_SSS_L3_8day_running_{d.year}_{doy:03d}_FNL_v06.0.nc"
    return (year_dir / fname).exists()

def _exists_aviso_month(aviso_root: Path, d: datetime) -> bool:
    return (aviso_root / f"{d.year:04d}-{d.month:02d}.nc").exists()

def bootstrap_scan(conn: sqlite3.Connection, start_year: int = 2010, end_year: Optional[int] = None):
    """
    One-time seeding: walk years on disk (cheap filename checks). After this,
    normal runs are incremental and DO NOT rescan.
    """
    if end_year is None:
        end_year = datetime.now(timezone.utc).year
    logging.getLogger("satdb").info(f"BOOTSTRAP scan filesystem years {start_year}..{end_year}")
    for year in range(start_year, end_year + 1):
        logging.getLogger("satdb").debug(f"BOOTSTRAP scanning year {year}")
        # Iterate days in year without listing directories aggressively
        d = datetime(year, 1, 1, tzinfo=timezone.utc)
        while d.year == year:
            ensure_day_row(conn, d)
            if _exists_sst(SST_ROOT, d):
                record_success(conn, d, "sst")
            if _exists_sss(SSS_ROOT, d):
                record_success(conn, d, "sss")
            if _exists_aviso_month(AVISO_ROOT, d):
                record_success(conn, d, "aviso")
            d += timedelta(days=1)
    conn.commit()
    logging.getLogger("satdb").info("BOOTSTRAP completed")

# ---------------------------- core update routine ----------------------------

def update_once(conn: sqlite3.Connection, rescan_days: Optional[int] = None):
    now = datetime.now(timezone.utc)
    yday = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)

    # Targets: rolling window for daily late arrivals
    lookback_days = DAILY_LOOKBACK_DAYS if not rescan_days or rescan_days <= 0 else int(rescan_days)
    logging.getLogger("satdb").info(f"UPDATE rolling daily window lookback_days={lookback_days}")
    targets = [yday - timedelta(days=i) for i in range(lookback_days)]
    # For each target, compute the nominal timestamps expected by your ensure_* functions
    for d in targets:
        d_sst = d.replace(hour=9)
        d_sss = d.replace(hour=12)

        # SST
        try:
            ensure_day_row(conn, d_sst)
            sst_done = conn.execute("SELECT sst FROM days WHERE date_= ?", (d.date().isoformat(),)).fetchone()[0]
            if not sst_done and should_retry(conn, d, "sst"):
                logging.getLogger("satdb").info(f"UPDATE SST attempt {d.date()}")
                path = ensure_sst_available(str(SST_ROOT), d_sst)
                if path:
                    record_success(conn, d, "sst")
                else:
                    record_attempt(conn, d, "sst", error="not-available")
        except Exception as e:
            record_attempt(conn, d, "sst", error=str(e)[:500])

        # SSS
        try:
            ensure_day_row(conn, d_sss)
            sss_done = conn.execute("SELECT sss FROM days WHERE date_= ?", (d.date().isoformat(),)).fetchone()[0]
            if not sss_done and should_retry(conn, d, "sss"):
                logging.getLogger("satdb").info(f"UPDATE SSS attempt {d.date()}")
                path = ensure_sss_available(str(SSS_ROOT), d_sss)
                if path:
                    record_success(conn, d, "sss")
                else:
                    record_attempt(conn, d, "sss", error="not-available")
        except Exception as e:
            record_attempt(conn, d, "sss", error=str(e)[:500])

    # AVISO monthly: anchor mid of previous month, idempotent
    first_this_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    mid_prev_month = first_this_month - timedelta(days=15)
    try:
        import calendar
        import xarray as xr
        mkey = f"{mid_prev_month.year:04d}-{mid_prev_month.month:02d}"
        # Decide whether the monthly file needs refresh regardless of DB state
        month_file = AVISO_ROOT / f"{mkey}.nc"
        days_in_month = calendar.monthrange(mid_prev_month.year, mid_prev_month.month)[1]
        needs_refresh = True
        if month_file.exists():
            try:
                ds_tmp = xr.open_dataset(month_file, decode_cf=True)
                n_time = int(ds_tmp.sizes.get("time", 0))
                ds_tmp.close()
                if n_time >= days_in_month:
                    needs_refresh = False
            except Exception:
                needs_refresh = True

        if needs_refresh and should_retry(conn, mid_prev_month, "aviso"):
            logging.getLogger("satdb").info(f"UPDATE AVISO monthly attempt for {mkey}")
            path = ensure_aviso_available(str(AVISO_ROOT), mid_prev_month)
            if path:
                try:
                    ds_chk = xr.open_dataset(path, decode_cf=True)
                    n_time = int(ds_chk.sizes.get("time", 0))
                    ds_chk.close()
                    if n_time >= days_in_month:
                        record_success(conn, mid_prev_month, "aviso")
                    else:
                        record_attempt(conn, mid_prev_month, "aviso", error=f"partial-month {n_time}/{days_in_month}")
                except Exception as e_chk:
                    record_attempt(conn, mid_prev_month, "aviso", error=f"open-failed: {str(e_chk)[:300]}")
    except Exception as e:
        record_attempt(conn, mid_prev_month, "aviso", error=str(e)[:500])

    conn.commit()
    logging.getLogger("satdb").info("UPDATE cycle completed")

# ---------------------------- reporting ----------------------------

def report(conn: sqlite3.Connection, limit: int = 30) -> str:
    cur = conn.cursor()
    totals = cur.execute(
        "SELECT COUNT(*), SUM(sst), SUM(sss), SUM(aviso) FROM days"
    ).fetchone()
    days_total, sst_total, sss_total, aviso_total = totals or (0,0,0,0)
    pending = cur.execute(
        "SELECT date_, (1-sst)+(1-sss)+(1-aviso) AS missing FROM days WHERE (sst+sss+aviso)<3 ORDER BY date_ DESC LIMIT ?",
        (limit,)
    ).fetchall()
    blocked = cur.execute(
        "SELECT date_, source, try_count, last_attempt, last_error FROM attempts WHERE try_count >= ? ORDER BY last_attempt DESC LIMIT ?",
        (MAX_TRIES, limit)
    ).fetchall()
    months_done = cur.execute("SELECT COUNT(*) FROM months WHERE aviso=1").fetchone()[0]

    lines = []
    lines.append("State summary")
    lines.append(f"  days tracked: {days_total}")
    lines.append(f"  SST complete: {sst_total} days")
    lines.append(f"  SSS complete: {sss_total} days")
    lines.append(f"  AVISO marked: {aviso_total} days (day-level flag)")
    lines.append(f"  AVISO months: {months_done} monthly files\n")
    if pending:
        lines.append("Recent pending (max 30):")
        for d, miss in pending[:30]:
            lines.append(f"  {d}: missing {int(miss)} sources")
    if blocked:
        lines.append("\nBlocked (max tries reached):")
        for date_, src, n, last, err in blocked[:30]:
            lines.append(f"  {date_} {src} tries={n} last={last} err={err}")
    return "\n".join(lines)

# ---------------------------- backfill ----------------------------

def _month_iter_midpoints(start_date: datetime, end_date: datetime):
    """Yield mid-of-month datetimes (UTC) from start to end inclusive."""
    if start_date.tzinfo is None:
        start_date = start_date.replace(tzinfo=timezone.utc)
    if end_date.tzinfo is None:
        end_date = end_date.replace(tzinfo=timezone.utc)
    # Normalize to first day of month for start
    current = start_date.replace(day=15, hour=0, minute=0, second=0, microsecond=0)
    last = end_date.replace(day=15, hour=0, minute=0, second=0, microsecond=0)
    while current <= last:
        yield current
        year = current.year + (1 if current.month == 12 else 0)
        month = 1 if current.month == 12 else current.month + 1
        current = current.replace(year=year, month=month)

def backfill_range(conn: sqlite3.Connection, start_date: datetime, end_date: datetime, sources: Set[str]):
    """
    Attempt to download historical data for SST/SSS (daily) and AVISO (monthly) across the date range.
    Honors backoff/try limits stored in the SQLite state.
    """
    logging.getLogger("satdb").info(f"BACKFILL start {start_date.date()}..{end_date.date()} sources={','.join(sorted(sources))}")
    # Daily products
    day = start_date
    while day <= end_date:
        day_utc = day if day.tzinfo else day.replace(tzinfo=timezone.utc)
        if "sst" in sources:
            try:
                d_sst = day_utc.replace(hour=9)
                ensure_day_row(conn, d_sst)
                sst_done = conn.execute("SELECT sst FROM days WHERE date_= ?", (day_utc.date().isoformat(),)).fetchone()[0]
                if not sst_done and should_retry(conn, day_utc, "sst"):
                    logging.getLogger("satdb").info(f"BACKFILL SST attempt {day_utc.date()}")
                    path = ensure_sst_available(str(SST_ROOT), d_sst)
                    if path:
                        record_success(conn, day_utc, "sst")
                    else:
                        record_attempt(conn, day_utc, "sst", error="not-available")
            except Exception as e:
                record_attempt(conn, day_utc, "sst", error=str(e)[:500])

        if "sss" in sources:
            try:
                d_sss = day_utc.replace(hour=12)
                ensure_day_row(conn, d_sss)
                sss_done = conn.execute("SELECT sss FROM days WHERE date_= ?", (day_utc.date().isoformat(),)).fetchone()[0]
                if not sss_done and should_retry(conn, day_utc, "sss"):
                    logging.getLogger("satdb").info(f"BACKFILL SSS attempt {day_utc.date()}")
                    path = ensure_sss_available(str(SSS_ROOT), d_sss)
                    if path:
                        record_success(conn, day_utc, "sss")
                    else:
                        record_attempt(conn, day_utc, "sss", error="not-available")
            except Exception as e:
                record_attempt(conn, day_utc, "sss", error=str(e)[:500])

        # Commit periodically to persist progress
        if int((day_utc - start_date).days) % 30 == 0:
            conn.commit()

        day = day + timedelta(days=1)

    # Monthly AVISO aggregates
    if "aviso" in sources:
        for mid in _month_iter_midpoints(start_date, end_date):
            try:
                import calendar
                import xarray as xr
                mkey = f"{mid.year:04d}-{mid.month:02d}"
                month_file = AVISO_ROOT / f"{mkey}.nc"
                days_in_month = calendar.monthrange(mid.year, mid.month)[1]
                needs_refresh = True
                if month_file.exists():
                    try:
                        ds_tmp = xr.open_dataset(month_file, decode_cf=True)
                        n_time = int(ds_tmp.sizes.get("time", 0))
                        ds_tmp.close()
                        if n_time >= days_in_month:
                            needs_refresh = False
                    except Exception:
                        needs_refresh = True

                if needs_refresh and should_retry(conn, mid, "aviso"):
                    logging.getLogger("satdb").info(f"BACKFILL AVISO monthly attempt {mkey}")
                    path = ensure_aviso_available(str(AVISO_ROOT), mid)
                    if path:
                        try:
                            ds_chk = xr.open_dataset(path, decode_cf=True)
                            n_time = int(ds_chk.sizes.get("time", 0))
                            ds_chk.close()
                            if n_time >= days_in_month:
                                record_success(conn, mid, "aviso")
                            else:
                                record_attempt(conn, mid, "aviso", error=f"partial-month {n_time}/{days_in_month}")
                        except Exception as e_chk:
                            record_attempt(conn, mid, "aviso", error=f"open-failed: {str(e_chk)[:300]}")
            except Exception as e:
                record_attempt(conn, mid, "aviso", error=str(e)[:500])

    conn.commit()
    logging.getLogger("satdb").info("BACKFILL completed")


# ---------------------------- CLI ----------------------------

def main(argv=None):
    argv = argv or sys.argv[1:]
    import argparse
    p = argparse.ArgumentParser(description="Update satellite archive with stateful control")
    p.add_argument("--bootstrap", action="store_true", help="One-time bootstrap: scan filesystem to seed DB")
    p.add_argument("--start-year", type=int, default=2010, help="Bootstrap start year (default 2010)")
    p.add_argument("--end-year", type=int, default=None, help="Bootstrap end year (default: current year)")
    p.add_argument("--report", action="store_true", help="Print a summary report and exit")
    p.add_argument("--report-limit", type=int, default=30, help="Max rows in report sections (default 30)")
    p.add_argument("--rescan-days", type=int, default=None, help="Override rolling window size for daily update")
    p.add_argument("--backfill", action="store_true", help="Download historical data across a date range")
    p.add_argument("--start-date", type=str, default=None, help="Backfill start date YYYY-MM-DD")
    p.add_argument("--end-date", type=str, default=None, help="Backfill end date YYYY-MM-DD")
    p.add_argument("--sources", type=str, default="sst,sss,aviso", help="Comma list of sources to fetch: sst,sss,aviso")
    p.add_argument("--reset-attempts", action="store_true", help="Reset all retry attempts to allow immediate retries")
    p.add_argument("--log", type=str, default=str(LOG_PATH), help="Log file path (default from SATDB_LOG)")
    p.add_argument("--log-level", type=str, default=os.environ.get("SATDB_LOG_LEVEL", "INFO"), help="Log level: DEBUG, INFO, WARNING, ERROR")
    p.add_argument("--log-stdout", action="store_true", help="Also emit logs to stdout")
    args = p.parse_args(argv)

    # Initialize logging early
    try:
        setup_logging(args.log, args.log_level, also_stdout=args.log_stdout)
    except Exception as e:
        print(f"Failed to initialize logging: {e}")

    conn = connect_db()

    if args.reset_attempts:
        logging.getLogger("satdb").info("Resetting all retry attempts...")
        conn.execute("DELETE FROM attempts")
        conn.commit()
        print("All retry attempts have been reset. Previous failures can now be retried immediately.")
        if not any([args.bootstrap, args.report, args.backfill]):
            print("Use --backfill, --bootstrap, or normal update to retry failed downloads.")
            return 0

    if args.bootstrap:
        bootstrap_scan(conn, args.start_year, args.end_year)
        print("Bootstrap complete.")
        print(report(conn, limit=args.report_limit))
        return 0

    if args.report:
        print(report(conn, limit=args.report_limit))
        return 0

    if args.backfill:
        # Infer dates if not provided, using year-range if present
        if args.start_date is None and args.start_year is not None:
            args.start_date = f"{args.start_year:04d}-01-01"
        if args.end_date is None:
            if args.end_year is not None:
                args.end_date = f"{args.end_year:04d}-12-31"
            else:
                # default to yesterday
                args.end_date = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")

        if not args.start_date or not args.end_date:
            print("--backfill requires --start-date and --end-date (or start/end-year)")
            return 2

        try:
            start_dt = _parse_ymd(args.start_date)
            end_dt = _parse_ymd(args.end_date)
        except Exception as e:
            print(f"Invalid date(s): {e}")
            return 2
        if start_dt > end_dt:
            print("start-date must be <= end-date")
            return 2

        srcs = set(s.strip().lower() for s in args.sources.split(",") if s.strip())
        allowed = {"sst", "sss", "aviso"}
        bad = srcs - allowed
        if bad:
            print(f"Unknown sources: {', '.join(sorted(bad))}. Allowed: sst, sss, aviso")
            return 2

        backfill_range(conn, start_dt, end_dt, srcs)
        print(report(conn, limit=args.report_limit))
        return 0

    update_once(conn, rescan_days=args.rescan_days)
    print(report(conn, limit=args.report_limit))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
