#!/bin/bash

# Activate conda environment
eval "$(/usr/bin/conda shell.bash hook)"
conda activate nespreso

# Compute dates
START_DATE=$(date -d "7 days ago" +%Y-%m-%d)
END_DATE=$(date -d "tomorrow" +%Y-%m-%d)

# Run update script
python /unity/g2/jmiranda/nespreso_api/scripts/sat_db/update_sat_archive.py \
  --reset-attempts \
  --backfill \
  --start-date $START_DATE \
  --end-date $END_DATE \
  --sources sss,aviso,sst \
  --log-stdout

# Run archive build script
python /unity/g2/jmiranda/nespreso_api/scripts/build_nespreso_archive.py --resume
