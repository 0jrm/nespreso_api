# NeSPReSO archive / satellite rebuild notes

## Daily flow

`nespreso_daily_update.sh` (cron):

1. Satellite update: `scripts/sat_db/update_sat_archive.py` (SST, SSS, AVISO)
2. Synthetic grids: `scripts/build_nespreso_archive.py --resume`

## SSS V6 naming (Aug 2026 fix)

- L3 V6 filenames must match **content center DOY** (same as V5 / PODAAC).
- Repair tool: `scripts/sat_db/fix_v6_sss_filenames.py` (`--dry-run` default, `--apply`, `--from-manifest`).
- Downloader (`ensure_sss_available`) uses **exact `granule_name`**; never `results[0]` + rename.
- If L3 missing: L2C 8-day aggregate → `*_FNL_v06.0_l2c.nc` (accessor accepts v06 → v05 → l2c).

## Archive builder abandons

- Dates with `retry_count >= --max-retries` (default 3) are marked `abandoned` and skipped on `--resume`.
- `20190708` is seeded abandoned (SMAP safehold 2019).

## Force-rebuild grids

```bash
cd /unity/g2/jmiranda/nespreso_api/scripts
python build_nespreso_archive.py --dates 20251101,20251102 --force-rebuild
# or a file of YYYYMMDD lines:
python build_nespreso_archive.py --dates rebuild_dates/dates_2025.txt --force-rebuild
```

## Satellite backfill

```bash
cd /unity/g2/jmiranda/nespreso_api/scripts/sat_db
python update_sat_archive.py --reset-attempts --backfill \
  --start-date 2026-06-19 --end-date $(date +%Y-%m-%d) \
  --sources sss,aviso,sst --log-stdout
```
