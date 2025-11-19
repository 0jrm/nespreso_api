#!/usr/bin/env bash
# set -Eeuo pipefail

# Default envs (override in crontab/systemd if desired)
export SATDB_DB="${SATDB_DB:-/Net/work/ozavala/DATA/SubSurfaceFields/NeSPReSO/satdb/state.db}"
export SATDB_SST_ROOT="${SATDB_SST_ROOT:-/Net/work/ozavala/DATA/GOFFISH/SST/OISST/}"
export SATDB_SSS_ROOT="${SATDB_SSS_ROOT:-/Net/work/ozavala/DATA/GOFFISH/SSS/SMAP_Global/}"
export SATDB_AVISO_ROOT="${SATDB_AVISO_ROOT:-/Net/work/ozavala/DATA/GOFFISH/AVISO/GoM/}"
export SATDB_LOG="${SATDB_LOG:-/unity/g2/jmiranda/nespreso_api/scripts/sat_db/update.log}"

mkdir -p "$(dirname "$SATDB_DB")" "$(dirname "$SATDB_LOG")"

# venv expected at /conda/jmiranda/miniconda/envs/nespreso; adjust if different
VENV="${VENV:-/conda/jmiranda/miniconda/envs/nespreso}"
PY="$VENV/bin/python"

# PATH hygiene for cron
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/bin:$PATH"

# Run the update script
"$PY" "/unity/g2/jmiranda/nespreso_api/scripts/sat_db/update_sat_archive.py" >>"$SATDB_LOG" 2>&1
