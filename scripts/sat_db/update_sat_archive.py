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
        logger.info(f"SSS not available for {t0}")
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

    if final.exists():
        logger.debug(f"AVISO monthly exists {final}")
        return final

    _assert_writable(aviso_path)

    # Create daily downloads directory
    daily_dir = aviso_path / "daily_downloads" / month_tag
    daily_dir.mkdir(parents=True, exist_ok=True)

    tmpdir = tempfile.mkdtemp(dir=str(aviso_path))
    try:
        # Download available daily files for this month
        pattern = f"*{date_.year}{date_.month:02d}??*.nc"  # YYYYMMDD
        logger.debug(f"AVISO download daily files for {month_tag}")
        resp = copernicusmarine.get(
            dataset_id=("cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D"),
            filter=pattern,
            output_directory=tmpdir,
            overwrite=False
        )

        # Move downloaded files to daily directory
        for file_info in resp.files:
            src_path = Path(file_info.file_path)
            dst_path = daily_dir / src_path.name
            if not dst_path.exists():  # Don't overwrite existing files
                shutil.move(str(src_path), str(dst_path))

        # Get all available daily files
        daily_files = sorted(daily_dir.glob("*.nc"))
        if not daily_files:
            logger.info(f"AVISO {month_tag}: no daily files available yet")
            raise FileNotFoundError(f"No DUACS files for {month_tag}")

        # Check if we have all days for the month
        import calendar
        days_in_month = calendar.monthrange(date_.year, date_.month)[1]
        available_days = len(daily_files)

        logger.info(f"AVISO {month_tag}: {available_days}/{days_in_month} daily files available")

        # Only create final monthly file if we have all days
        if available_days >= days_in_month:
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

            _atomic_move(tmp_month, final)

            # Clean up daily files after successful monthly creation
            shutil.rmtree(daily_dir, ignore_errors=True)
            logger.info(f"AVISO {month_tag}: monthly file completed -> {final}")
        else:
            # Create intermediate file for partial month
            if available_days > 0:
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
                logger.info(f"AVISO {month_tag}: partial monthly file created ({available_days}/{days_in_month} days) -> {final}")
            else:
                logger.info(f"AVISO {month_tag}: no files found to build partial monthly file")
                raise FileNotFoundError(f"No DUACS files for {month_tag}")

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
        mkey = f"{mid_prev_month.year:04d}-{mid_prev_month.month:02d}"
        row = conn.execute("SELECT aviso FROM months WHERE month= ?", (mkey,)).fetchone()
        if (row is None or int(row[0]) == 0) and should_retry(conn, mid_prev_month, "aviso"):
            logging.getLogger("satdb").info(f"UPDATE AVISO monthly attempt for {mkey}")
            path = ensure_aviso_available(str(AVISO_ROOT), mid_prev_month)
            if path:
                record_success(conn, mid_prev_month, "aviso")
            else:
                record_attempt(conn, mid_prev_month, "aviso", error="not-available")
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
                mkey = f"{mid.year:04d}-{mid.month:02d}"
                row = conn.execute("SELECT aviso FROM months WHERE month= ?", (mkey,)).fetchone()
                if (row is None or int(row[0]) == 0) and should_retry(conn, mid, "aviso"):
                    logging.getLogger("satdb").info(f"BACKFILL AVISO monthly attempt {mkey}")
                    path = ensure_aviso_available(str(AVISO_ROOT), mid)
                    if path:
                        record_success(conn, mid, "aviso")
                    else:
                        record_attempt(conn, mid, "aviso", error="not-available")
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
