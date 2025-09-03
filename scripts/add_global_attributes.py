#!/usr/bin/env python3
"""
Update global attributes for NetCDF files under a directory.

This script walks a root directory, finds NetCDF files (default: .nc, .nc4),
and sets the following global attributes without removing existing ones:

    - institution: COAPS, FSU
    - author: Jose Roberto Miranda
    - contact: jrm22n@fsu.edu
    - DOI: https://doi.org/10.1016/j.ocemod.2025.102550

Existing attributes with the same names will be updated to the values above.

Usage examples:
  python3 scripts/add_global_attributes.py --root /Net/work/ozavala/DATA/SubSurfaceFields/NeSPReSO --limit 5
  python3 scripts/add_global_attributes.py --root /Net/work/ozavala/DATA/SubSurfaceFields/NeSPReSO
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
from typing import Dict, Iterable, List, Tuple

try:
    from netCDF4 import Dataset  # type: ignore
except Exception as exc:  # pragma: no cover
    sys.stderr.write(
        f"Failed to import netCDF4. Please install it in your Python environment. Error: {exc}\n"
    )
    raise


DEFAULT_ATTRIBUTES: Dict[str, str] = {
    "institution": "COAPS, FSU",
    "author": "Jose Roberto Miranda",
    "contact": "jrm22n@fsu.edu",
    "DOI": "https://doi.org/10.1016/j.ocemod.2025.102550",
}


def configure_logging(log_file_path: str | None) -> None:
    log_level = logging.INFO
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Stream handler
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel(log_level)
    stream_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    stream_handler.setFormatter(stream_formatter)
    logger.addHandler(stream_handler)

    # Optional file handler
    if log_file_path:
        log_dir = os.path.dirname(os.path.abspath(log_file_path))
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(stream_formatter)
        logger.addHandler(file_handler)


def find_netcdf_files(
    root_dir: str, extensions: Tuple[str, ...] = (".nc", ".nc4")
) -> Iterable[str]:
    for current_root, _dirs, files in os.walk(root_dir):
        for filename in files:
            lower_name = filename.lower()
            if any(lower_name.endswith(ext) for ext in extensions):
                yield os.path.join(current_root, filename)


def update_attributes_for_file(
    file_path: str,
    attributes_to_set: Dict[str, str],
    overwrite: bool = True,
) -> Tuple[int, int]:
    """Return (num_updated, num_skipped_same_value)."""
    num_updated = 0
    num_skipped = 0
    try:
        with Dataset(file_path, mode="a") as ds:  # append in-place
            existing_attr_names = set(ds.ncattrs())
            for attr_name, desired_value in attributes_to_set.items():
                if attr_name in existing_attr_names:
                    try:
                        current_value = ds.getncattr(attr_name)
                    except Exception:
                        current_value = None
                    if current_value == desired_value:
                        num_skipped += 1
                        continue
                    if not overwrite:
                        num_skipped += 1
                        continue
                    ds.setncattr(attr_name, desired_value)
                    logging.info(
                        f"{file_path}: updated attribute '{attr_name}' from '{current_value}' to '{desired_value}'"
                    )
                    num_updated += 1
                else:
                    ds.setncattr(attr_name, desired_value)
                    logging.info(
                        f"{file_path}: added attribute '{attr_name}' = '{desired_value}'"
                    )
                    num_updated += 1
    except Exception as exc:
        logging.error(f"{file_path}: FAILED to update attributes: {exc}")
    return num_updated, num_skipped


def process_directory(
    root_dir: str,
    attributes_to_set: Dict[str, str],
    limit: int | None = None,
    overwrite: bool = True,
) -> None:
    files_iter = find_netcdf_files(root_dir)
    total_seen = 0
    total_updated = 0
    total_skipped = 0
    total_failed = 0

    for file_path in files_iter:
        if limit is not None and total_seen >= limit:
            break
        total_seen += 1
        before_updated = total_updated
        before_skipped = total_skipped
        try:
            updated_count, skipped_count = update_attributes_for_file(
                file_path=file_path,
                attributes_to_set=attributes_to_set,
                overwrite=overwrite,
            )
            total_updated += updated_count
            total_skipped += skipped_count
        except Exception as exc:
            total_failed += 1
            logging.error(f"{file_path}: ERROR: {exc}")

        # Periodic progress
        if total_seen % 50 == 0:
            logging.info(
                f"Progress: processed={total_seen}, updated={total_updated}, skipped={total_skipped}, failed={total_failed}"
            )

    logging.info(
        "Completed. processed=%s, updated=%s, skipped=%s, failed=%s",
        total_seen,
        total_updated,
        total_skipped,
        total_failed,
    )


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add or update global attributes for NetCDF files under a directory."
    )
    parser.add_argument(
        "--root",
        required=True,
        help="Root directory to scan for NetCDF files",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit of files to process (for testing)",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Do not overwrite existing attributes if present with any value",
    )
    parser.add_argument(
        "--log-file",
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "logs",
            f"add_global_attributes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        ),
        help="Path to a log file. Defaults to logs/add_global_attributes_<timestamp>.log",
    )
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    configure_logging(args.log_file)

    root_dir = os.path.abspath(args.root)
    if not os.path.isdir(root_dir):
        logging.error("Root directory does not exist: %s", root_dir)
        return 2

    logging.info("Starting attribute update in root: %s", root_dir)
    logging.info("Attributes to set: %s", ", ".join(f"{k}='{v}'" for k, v in DEFAULT_ATTRIBUTES.items()))
    logging.info("Overwrite existing: %s", not args.no_overwrite)
    logging.info("Log file: %s", getattr(args, "log_file", "<stdout only>"))

    process_directory(
        root_dir=root_dir,
        attributes_to_set=DEFAULT_ATTRIBUTES,
        limit=args.limit,
        overwrite=not args.no_overwrite,
    )

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))


