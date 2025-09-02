# Stateful Satellite Archive Updater

This package gives you a **single daily job** that:
- Downloads GHRSST/MUR (SST) and SMAP (SSS) for the proper nominal times.
- Builds the previous month’s AVISO/DUACS aggregate once (then skips).
- **Tracks state in SQLite** so you do **not** rescan the archive or retry blindly.
- Applies exponential backoff for transient missing products.
- Offers a cheap one-time `--bootstrap` scan to seed the DB from files already on disk.

## Files
- `update_sat_archive.py` – main orchestrator (rename-ready replacement for your module)
- `run_update.sh` – wrapper script intended for cron or systemd
- (Optional) Keep your existing audit script for ad hoc reports.

## Install
The scripts are designed to run from their current location with a pre-existing conda environment.

Prerequisites:
- Existing conda environment: `/conda/jmiranda/miniconda/envs/nespreso`
- Required packages: `earthaccess`, `copernicusmarine`, `xarray`, `netCDF4`, `h5netcdf`

No additional installation steps required - the scripts use the existing environment and paths.

## First run
- Configure Earthdata and Copernicus Marine credentials (see earlier instructions).
- Bootstrap once to seed the DB from existing files:
```bash
cd /unity/g2/jmiranda/nespreso_api/scripts/sat_db
/conda/jmiranda/miniconda/envs/nespreso/bin/python update_sat_archive.py --bootstrap --start-year 2020 --end-year 2024
```

## Cron (UTC, daily 13:30Z)
```
CRON_TZ=UTC
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/bin
30 13 * * * /unity/g2/jmiranda/nespreso_api/scripts/sat_db/run_update.sh
```

## Queries
```bash
cd /unity/g2/jmiranda/nespreso_api/scripts/sat_db
/conda/jmiranda/miniconda/envs/nespreso/bin/python update_sat_archive.py --report
```

## Notes
- The DB schema keeps both per-day flags and per-month AVISO flags. On AVISO success, we also flip
  the day-level `aviso` flag for simplicity when intersecting.
- Backoff defaults: BASE=30 min, factor=2, max tries=8 (tune via env).
- By default, the script only checks recent dates (yesterday + 10 days back) for new data.
  Use `--bootstrap` to scan historical data from existing files on disk.
- Recent dates may show "missing" if data hasn't been published yet by the providers.
- **AVISO Download Strategy**: Uses incremental approach - downloads daily files and rebuilds
  monthly aggregates progressively. Daily files are stored in `daily_downloads/YYYY-MM/`
  and cleaned up once the complete monthly file is created.

