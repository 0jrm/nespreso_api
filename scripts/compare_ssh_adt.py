#!/usr/bin/env python3
"""
SSH (Copernicus/CMEMS) vs ADT (AVISO) comparison.

Modes:
- Daily files: iterate days of a month, collocate and compute daily scatter/regression
- Monthly files: aggregate all days of the month into a single scatter/regression
- Range mode: iterate months between --start-month and --end-month inclusive

Outputs:
- Per-day or per-month PNG scatter plots
- Per-month CSV summary with slope, intercept, r^2, and sample count
- In range mode, an additional combined CSV across months

Defaults:
- Copernicus (daily) directory: /unity/g2/jmiranda/Data/ocean_data-new_ssh
- AVISO directory: /Net/work/ozavala/DATA/GOFFISH/AVISO/GoM
- Output directory: /unity/g2/jmiranda/outputs/ssh_adt_compare/<YYYY-MM>

Notes:
- Requires: xarray, h5netcdf, hdf5plugin, netCDF4, numpy, pandas, matplotlib
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import os
import re
import sys
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr

try:
    import hdf5plugin  # noqa: F401  # Ensures HDF5 filters are registered
    H5_ENGINE = "h5netcdf"
except Exception:  # pragma: no cover - environment-dependent
    H5_ENGINE = "netcdf4"


COPERNICUS_DAILY_REGEX = re.compile(
    r"glo12_rg_1d-m_(\d{8})-\1_2D_hcst_.*\.nc$"
)


@dataclasses.dataclass
class ComparisonConfig:
    copernicus_dir: str = \
        "/unity/g2/jmiranda/Data/ocean_data-new_ssh"
    aviso_dir: str = "/Net/work/ozavala/DATA/GOFFISH/AVISO/GoM"
    month: Optional[str] = None  # Format: YYYY-MM
    output_root: str = "/unity/g2/jmiranda/outputs/ssh_adt_compare"
    ssh_var: str = "zos"  # Copernicus sea surface height variable name
    adt_var: str = "adt"
    # Bounding box for analysis [min_lon, max_lon, min_lat, max_lat]
    min_lon: float = -98.0
    max_lon: float = -81.0
    min_lat: float = 18.0
    max_lat: float = 31.0
    monthly: bool = False
    # Optional: CMEMS monthly SSH directory containing files named YYYY-MM.nc with variable `zos`
    copernicus_monthly_dir: Optional[str] = None
    # Optional: month range processing (inclusive)
    start_month: Optional[str] = None  # YYYY-MM
    end_month: Optional[str] = None    # YYYY-MM
    aggregate_all: bool = False  # If set with a range, compute one regression/plot across all months


def parse_args(argv: Optional[List[str]] = None) -> ComparisonConfig:
    parser = argparse.ArgumentParser(
        description="Compare daily SSH (Copernicus) vs ADT (AVISO) for a month."
    )
    parser.add_argument(
        "--copernicus-dir",
        default="/unity/g2/jmiranda/Data/ocean_data-new_ssh",
        help="Directory containing Copernicus daily SSH NetCDF files",
    )
    parser.add_argument(
        "--aviso-dir",
        default="/Net/work/ozavala/DATA/GOFFISH/AVISO/GoM",
        help="Directory containing AVISO monthly ADT NetCDF files (YYYY-MM.nc)",
    )
    parser.add_argument(
        "--month",
        default=None,
        help="Month to process in YYYY-MM (default: latest common month found)",
    )
    parser.add_argument(
        "--output-root",
        default="/unity/g2/jmiranda/outputs/ssh_adt_compare",
        help="Root directory for outputs (plots and CSV)",
    )
    parser.add_argument(
        "--monthly",
        action="store_true",
        help="Aggregate all days in the month into a single scatter/regression",
    )
    parser.add_argument("--min-lon", type=float, default=-98.0)
    parser.add_argument("--max-lon", type=float, default=-81.0)
    parser.add_argument("--min-lat", type=float, default=18.0)
    parser.add_argument("--max-lat", type=float, default=31.0)
    parser.add_argument(
        "--cmems-monthly-dir",
        default=None,
        help="Directory with CMEMS monthly SSH files named YYYY-MM.nc (variable 'zos')",
    )
    parser.add_argument(
        "--start-month",
        default=None,
        help="Start month for range processing, format YYYY-MM (inclusive)",
    )
    parser.add_argument(
        "--end-month",
        default=None,
        help="End month for range processing, format YYYY-MM (inclusive)",
    )
    parser.add_argument(
        "--aggregate-all",
        action="store_true",
        help="When used with a month range, make one plot/regression across all months",
    )
    args = parser.parse_args(argv)
    return ComparisonConfig(
        copernicus_dir=args.copernicus_dir,
        aviso_dir=args.aviso_dir,
        month=args.month,
        output_root=args.output_root,
        min_lon=args.min_lon,
        max_lon=args.max_lon,
        min_lat=args.min_lat,
        max_lat=args.max_lat,
        monthly=bool(args.monthly),
        copernicus_monthly_dir=args.cmems_monthly_dir,
        start_month=args.start_month,
        end_month=args.end_month,
        aggregate_all=bool(args.aggregate_all),
    )


def discover_copernicus_days(copernicus_dir: str) -> Dict[str, List[dt.date]]:
    """Scan directory for Copernicus daily files and return mapping of month -> dates.

    Expects filenames like: glo12_rg_1d-m_YYYYMMDD-YYYYMMDD_2D_hcst_*.nc
    Returns dict {"YYYY-MM": [date, ...]} with dates sorted.
    """
    month_to_dates: Dict[str, List[dt.date]] = {}
    try:
        file_names = sorted(os.listdir(copernicus_dir))
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Copernicus directory not found: {copernicus_dir}"
        )

    for name in file_names:
        match = COPERNICUS_DAILY_REGEX.match(name)
        if not match:
            continue
        yyyymmdd = match.group(1)
        try:
            day = dt.datetime.strptime(yyyymmdd, "%Y%m%d").date()
        except ValueError:
            continue
        month_key = day.strftime("%Y-%m")
        month_to_dates.setdefault(month_key, []).append(day)

    for month_key in list(month_to_dates.keys()):
        month_to_dates[month_key] = sorted(month_to_dates[month_key])

    return month_to_dates


def select_month(configured_month: Optional[str], cop_months: Iterable[str], aviso_dir: str) -> str:
    """Choose the month to process.

    Priority:
    1) If configured_month provided and AVISO file exists, use it.
    2) Else choose latest month present in Copernicus that also has an AVISO file.
    """
    def aviso_path(month_key: str) -> str:
        return os.path.join(aviso_dir, f"{month_key}.nc")

    if configured_month:
        path = aviso_path(configured_month)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"AVISO file not found for --month {configured_month}: {path}"
            )
        return configured_month

    months_sorted = sorted(cop_months)
    if not months_sorted:
        raise RuntimeError("No Copernicus daily files found.")

    # Pick the latest month with an existing AVISO monthly file
    for month_key in reversed(months_sorted):
        if os.path.exists(aviso_path(month_key)):
            return month_key

    raise RuntimeError(
        "Could not find any month with both Copernicus daily files and an AVISO monthly file."
    )


def open_aviso_month(aviso_dir: str, month_key: str, adt_var: str) -> xr.Dataset:
    path = os.path.join(aviso_dir, f"{month_key}.nc")
    if not os.path.exists(path):
        raise FileNotFoundError(f"AVISO monthly file not found: {path}")
    ds = xr.open_dataset(path)
    if adt_var not in ds:
        raise KeyError(f"Variable '{adt_var}' not found in AVISO file: {path}")
    # Basic sanity on coords
    for coord_name in ("latitude", "longitude", "time"):
        if coord_name not in ds.coords:
            raise KeyError(f"Coordinate '{coord_name}' missing in AVISO file: {path}")
    return ds


def open_copernicus_daily(copernicus_dir: str, day: dt.date, ssh_var: str) -> xr.Dataset:
    pattern = f"glo12_rg_1d-m_{day.strftime('%Y%m%d')}-{day.strftime('%Y%m%d')}_2D_hcst_"
    candidates = [
        name for name in os.listdir(copernicus_dir)
        if name.startswith(pattern) and name.endswith(".nc")
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No Copernicus daily file found for {day} in {copernicus_dir}"
        )
    # Prefer the most recent revision if multiple
    file_name = sorted(candidates)[-1]
    path = os.path.join(copernicus_dir, file_name)
    ds = xr.open_dataset(path, engine=H5_ENGINE)
    if ssh_var not in ds:
        raise KeyError(f"Variable '{ssh_var}' not in Copernicus file: {path}")
    # Expect coords: longitude [-180,180), latitude [-80,90]
    for coord_name in ("latitude", "longitude", "time"):
        if coord_name not in ds.coords:
            raise KeyError(f"Coordinate '{coord_name}' missing in Copernicus file: {path}")
    return ds


def collocate_to_aviso_grid(
    ssh_da: xr.DataArray,
    aviso_lat: xr.DataArray,
    aviso_lon: xr.DataArray,
) -> xr.DataArray:
    """Interpolate SSH to AVISO grid (latitude, longitude)."""
    # Ensure we have 2D field (latitude, longitude)
    missing_dims = [d for d in ("latitude", "longitude") if d not in ssh_da.dims]
    if missing_dims:
        raise ValueError(
            f"SSH data must have latitude/longitude dims; missing {missing_dims}"
        )
    # xarray.interp expects 1D target coordinates
    ssh_interp = ssh_da.interp(latitude=aviso_lat, longitude=aviso_lon)
    return ssh_interp


def compute_regression(x_values: np.ndarray, y_values: np.ndarray) -> Tuple[float, float, float]:
    """Return (slope, intercept, r2)."""
    if x_values.size == 0 or y_values.size == 0:
        return float("nan"), float("nan"), float("nan")
    # Drop NaNs
    mask = np.isfinite(x_values) & np.isfinite(y_values)
    x = x_values[mask]
    y = y_values[mask]
    if x.size < 2:
        return float("nan"), float("nan"), float("nan")
    # Least squares
    slope, intercept = np.polyfit(x, y, deg=1)
    # r^2 from Pearson correlation
    if x.size > 1:
        r = np.corrcoef(x, y)[0, 1]
        r2 = float(r * r)
    else:
        r2 = float("nan")
    return float(slope), float(intercept), r2


def make_daily_plot(
    day: dt.date,
    ssh_flat: np.ndarray,
    adt_flat: np.ndarray,
    slope: float,
    intercept: float,
    r2: float,
    output_path: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Drop NaNs for plotting
    mask = np.isfinite(ssh_flat) & np.isfinite(adt_flat)
    x = ssh_flat[mask]
    y = adt_flat[mask]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(x, y, s=6, c="#1f77b4", alpha=0.4, edgecolors="none")

    # Limits and 1:1 line
    x_min = np.nanpercentile(x, 1) if x.size else -0.5
    x_max = np.nanpercentile(x, 99) if x.size else 0.5
    y_min = np.nanpercentile(y, 1) if y.size else -0.5
    y_max = np.nanpercentile(y, 99) if y.size else 0.5
    axis_min = float(min(x_min, y_min))
    axis_max = float(max(x_max, y_max))
    pad = 0.05 * (axis_max - axis_min) if np.isfinite(axis_max - axis_min) else 0.1
    ax.set_xlim(axis_min - pad, axis_max + pad)
    ax.set_ylim(axis_min - pad, axis_max + pad)

    line_x = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 100)
    ax.plot(line_x, line_x, "k--", linewidth=1, label="1:1 line")

    if np.isfinite(slope) and np.isfinite(intercept):
        reg_y = slope * line_x + intercept
        ax.plot(line_x, reg_y, "r-", linewidth=2, label="Regression")

    ax.set_title(f"SSH vs ADT — {day.isoformat()}")
    ax.set_xlabel("Copernicus SSH (m)")
    ax.set_ylabel("AVISO ADT (m)")

    equation_text = (
        f"ADT = {slope:.4f} · SSH + {intercept:.4f}\n"
        f"R² = {r2:.4f}    N = {int(mask.sum())}"
    )
    ax.text(0.02, 0.98, equation_text, transform=ax.transAxes, va="top", ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="lower right")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def process_month(config: ComparisonConfig) -> Tuple[str, pd.DataFrame]:
    # Determine mode: CMEMS monthly file vs. daily files
    if config.copernicus_monthly_dir:
        # Use a single monthly SSH file (zos) and AVISO monthly ADT
        month_key = config.month or (
            dt.date.today().strftime("%Y-%m")
        )
        cmems_path = os.path.join(config.copernicus_monthly_dir, f"{month_key}.nc")
        if not os.path.exists(cmems_path):
            raise FileNotFoundError(
                f"CMEMS monthly file not found: {cmems_path}"
            )
        cmems_ds = xr.open_dataset(cmems_path, engine=H5_ENGINE)
        if config.ssh_var not in cmems_ds:
            raise KeyError(
                f"Variable '{config.ssh_var}' not found in CMEMS file: {cmems_path}"
            )
        dates_in_month = list(pd.to_datetime(cmems_ds.time.values).date)
    else:
        # Fall back to daily files in config.copernicus_dir
        month_to_dates = discover_copernicus_days(config.copernicus_dir)
        month_key = select_month(config.month, month_to_dates.keys(), config.aviso_dir)
        dates_in_month = month_to_dates.get(month_key, [])
        if not dates_in_month:
            raise RuntimeError(f"No Copernicus daily files for month {month_key}")

    aviso_ds = open_aviso_month(config.aviso_dir, month_key, config.adt_var)
    aviso_lat = aviso_ds["latitude"]
    aviso_lon = aviso_ds["longitude"]

    out_dir = os.path.join(config.output_root, month_key)
    os.makedirs(out_dir, exist_ok=True)

    summary_rows: List[Dict[str, object]] = []
    # For monthly aggregation
    monthly_x: List[np.ndarray] = []
    monthly_y: List[np.ndarray] = []

    # Precompute bbox intersection with AVISO grid extents
    lat_min_allowed = float(aviso_lat.min())
    lat_max_allowed = float(aviso_lat.max())
    lon_min_allowed = float(aviso_lon.min())
    lon_max_allowed = float(aviso_lon.max())

    bbox_lat_min = max(config.min_lat, lat_min_allowed)
    bbox_lat_max = min(config.max_lat, lat_max_allowed)
    bbox_lon_min = max(config.min_lon, lon_min_allowed)
    bbox_lon_max = min(config.max_lon, lon_max_allowed)

    aviso_lat_bbox = aviso_lat.sel(latitude=slice(bbox_lat_min, bbox_lat_max))
    aviso_lon_bbox = aviso_lon.sel(longitude=slice(bbox_lon_min, bbox_lon_max))

    for day in dates_in_month:
        # Select AVISO ADT for this day
        # Use nearest with tolerance to guard against non-midnight timestamps
        target_time = np.datetime64(day.isoformat())
        try:
            aviso_day = aviso_ds[config.adt_var].sel(
                time=target_time, method="nearest", tolerance=np.timedelta64(1, "D")
            )
        except Exception:
            # Fallback: try exact selection without tolerance
            aviso_day = aviso_ds[config.adt_var].sel(time=target_time)

        # Open Copernicus source for this day
        if config.copernicus_monthly_dir:
            # Select the corresponding time slice from monthly dataset
            ssh_day = cmems_ds[config.ssh_var].sel(time=np.datetime64(day))
            ssh = ssh_day
        else:
            cop_ds = open_copernicus_daily(config.copernicus_dir, day, config.ssh_var)
            ssh = cop_ds[config.ssh_var]
            if "time" in ssh.dims and ssh.sizes.get("time", 1) == 1:
                ssh = ssh.isel(time=0)
            if "depth" in ssh.dims and ssh.sizes.get("depth", 1) == 1:
                ssh = ssh.isel(depth=0)

        # Spatial subset to bbox to reduce interpolation work
        lat_min = bbox_lat_min
        lat_max = bbox_lat_max
        lon_min = bbox_lon_min
        lon_max = bbox_lon_max

        # Ensure slicing with increasing coordinates
        lat_slice = slice(lat_min, lat_max) if ssh["latitude"].values[0] < ssh["latitude"].values[-1] else slice(lat_max, lat_min)
        lon_slice = slice(lon_min, lon_max) if ssh["longitude"].values[0] < ssh["longitude"].values[-1] else slice(lon_max, lon_min)

        ssh_region = ssh.sel(latitude=lat_slice, longitude=lon_slice)
        # Reduce AVISO daily to bbox
        aviso_day_bbox = aviso_day.sel(latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max))
        ssh_on_aviso = collocate_to_aviso_grid(ssh_region, aviso_lat_bbox, aviso_lon_bbox)

        # Flatten to vectors (both on the same bbox)
        x_flat = ssh_on_aviso.values.reshape(-1)
        y_flat = aviso_day_bbox.values.reshape(-1)

        if config.monthly:
            monthly_x.append(x_flat)
            monthly_y.append(y_flat)
        else:
            slope, intercept, r2 = compute_regression(x_flat, y_flat)

            plot_path = os.path.join(out_dir, f"ssh_vs_adt_{day.strftime('%Y%m%d')}.png")
            make_daily_plot(day, x_flat, y_flat, slope, intercept, r2, plot_path)

            mask_pairs = np.isfinite(x_flat) & np.isfinite(y_flat)
            summary_rows.append(
                {
                    "date": day.isoformat(),
                    "slope": slope,
                    "intercept": intercept,
                    "r2": r2,
                    "n": int(mask_pairs.sum()),
                    "plot": plot_path,
                }
            )

    # If monthly aggregation requested, compute single regression and plot
    if config.monthly:
        if monthly_x and monthly_y:
            x_all = np.concatenate(monthly_x)
            y_all = np.concatenate(monthly_y)
        else:
            x_all = np.array([])
            y_all = np.array([])

        slope, intercept, r2 = compute_regression(x_all, y_all)
        # Build a monthly plot using the first day's date for a contextual title
        any_day = dates_in_month[0]
        monthly_plot = os.path.join(out_dir, f"ssh_vs_adt_{month_key}.png")
        make_daily_plot(any_day, x_all, y_all, slope, intercept, r2, monthly_plot)

        mask_pairs = np.isfinite(x_all) & np.isfinite(y_all)
        summary_rows.append(
            {
                "date": month_key,
                "slope": slope,
                "intercept": intercept,
                "r2": r2,
                "n": int(mask_pairs.sum()),
                "plot": monthly_plot,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty and "date" in summary_df:
        summary_df = summary_df.sort_values("date").reset_index(drop=True)
    csv_path = os.path.join(out_dir, f"summary_{month_key}.csv")
    summary_df.to_csv(csv_path, index=False)
    return out_dir, summary_df


def month_range(start_month: str, end_month: str) -> List[str]:
    start = dt.datetime.strptime(start_month, "%Y-%m").date().replace(day=1)
    end = dt.datetime.strptime(end_month, "%Y-%m").date().replace(day=1)
    if start > end:
        start, end = end, start
    months: List[str] = []
    current = start
    while current <= end:
        months.append(current.strftime("%Y-%m"))
        # Advance by one month
        year = current.year + (1 if current.month == 12 else 0)
        month = 1 if current.month == 12 else current.month + 1
        current = current.replace(year=year, month=month)
    return months


def main(argv: Optional[List[str]] = None) -> int:
    config = parse_args(argv)
    try:
        # Range mode
        if config.start_month and config.end_month:
            months = month_range(config.start_month, config.end_month)
            combined_rows: List[pd.DataFrame] = []
            if config.aggregate_all:
                all_x: List[np.ndarray] = []
                all_y: List[np.ndarray] = []
            last_out_dir = None
            for month_key in months:
                cfg = dataclasses.replace(config, month=month_key)
                out_dir, summary_df = process_month(cfg)
                last_out_dir = out_dir
                combined_rows.append(summary_df.assign(month=month_key))
                if config.aggregate_all:
                    # If monthly mode, we can reload the month plot inputs by recomputing quickly
                    # Simpler: open the summary month outputs and recompute aggregation via data again
                    # Here, aggregate via re-running process_month with monthly True and grabbing data X/Y isn't persisted.
                    # Instead, rerun the internal logic: open both datasets and accumulate pairs.
                    # To avoid code duplication, do a lightweight recompute: open ds and collect pairs here as well.
                    # However, to keep runtime reasonable, re-open within loop using the same subset steps.
                    # We'll mirror the essential steps for this month.
                    # Open AVISO month
                    aviso_ds = open_aviso_month(config.aviso_dir, month_key, config.adt_var)
                    aviso_lat = aviso_ds["latitude"]
                    aviso_lon = aviso_ds["longitude"]
                    # Bbox
                    lat_min_allowed = float(aviso_lat.min())
                    lat_max_allowed = float(aviso_lat.max())
                    lon_min_allowed = float(aviso_lon.min())
                    lon_max_allowed = float(aviso_lon.max())
                    bbox_lat_min = max(config.min_lat, lat_min_allowed)
                    bbox_lat_max = min(config.max_lat, lat_max_allowed)
                    bbox_lon_min = max(config.min_lon, lon_min_allowed)
                    bbox_lon_max = min(config.max_lon, lon_max_allowed)
                    aviso_lat_bbox = aviso_lat.sel(latitude=slice(bbox_lat_min, bbox_lat_max))
                    aviso_lon_bbox = aviso_lon.sel(longitude=slice(bbox_lon_min, bbox_lon_max))
                    # CMEMS monthly SSH
                    if not config.copernicus_monthly_dir:
                        raise RuntimeError("--aggregate-all requires --cmems-monthly-dir and --monthly")
                    cmems_path = os.path.join(config.copernicus_monthly_dir, f"{month_key}.nc")
                    cmems_ds = xr.open_dataset(cmems_path, engine=H5_ENGINE)
                    dates_in_month = list(pd.to_datetime(cmems_ds.time.values).date)
                    for day in dates_in_month:
                        target_time = np.datetime64(day.isoformat())
                        try:
                            aviso_day = aviso_ds[config.adt_var].sel(
                                time=target_time, method="nearest", tolerance=np.timedelta64(1, "D")
                            )
                        except Exception:
                            aviso_day = aviso_ds[config.adt_var].sel(time=target_time)
                        ssh_day = cmems_ds[config.ssh_var].sel(time=np.datetime64(day))
                        ssh = ssh_day
                        # Spatial subset
                        lat_min = bbox_lat_min
                        lat_max = bbox_lat_max
                        lon_min = bbox_lon_min
                        lon_max = bbox_lon_max
                        lat_slice = slice(lat_min, lat_max) if ssh["latitude"].values[0] < ssh["latitude"].values[-1] else slice(lat_max, lat_min)
                        lon_slice = slice(lon_min, lon_max) if ssh["longitude"].values[0] < ssh["longitude"].values[-1] else slice(lon_max, lon_min)
                        ssh_region = ssh.sel(latitude=lat_slice, longitude=lon_slice)
                        aviso_day_bbox = aviso_day.sel(latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max))
                        ssh_on_aviso = collocate_to_aviso_grid(ssh_region, aviso_lat_bbox, aviso_lon_bbox)
                        x_flat = ssh_on_aviso.values.reshape(-1)
                        y_flat = aviso_day_bbox.values.reshape(-1)
                        all_x.append(x_flat)
                        all_y.append(y_flat)

            if combined_rows:
                combined_df = pd.concat(combined_rows, ignore_index=True)
            else:
                combined_df = pd.DataFrame()

            # Save combined CSV one level above month folders
            combined_root = config.output_root
            os.makedirs(combined_root, exist_ok=True)
            combined_csv = os.path.join(
                combined_root,
                f"summary_{months[0]}_to_{months[-1]}.csv",
            )
            combined_df.to_csv(combined_csv, index=False)
            print(f"Wrote combined summary: {combined_csv}")
            if config.aggregate_all:
                # Compute one regression across all months' data and make one plot
                if all_x and all_y:
                    X = np.concatenate(all_x)
                    Y = np.concatenate(all_y)
                else:
                    X = np.array([])
                    Y = np.array([])
                slope, intercept, r2 = compute_regression(X, Y)
                # Use first month as context date
                first_month = months[0]
                # Dummy day for title
                any_day = dt.datetime.strptime(first_month + "-01", "%Y-%m-%d").date()
                agg_plot = os.path.join(config.output_root, f"ssh_vs_adt_{months[0]}_to_{months[-1]}.png")
                make_daily_plot(any_day, X, Y, slope, intercept, r2, agg_plot)
                print(f"Aggregate plot saved: {agg_plot}")
            if not combined_df.empty:
                print(combined_df.groupby('date')[['slope','intercept','r2','n']].mean().to_string())
            if last_out_dir:
                print(f"Last month outputs in: {last_out_dir}")
            return 0
        else:
            out_dir, summary_df = process_month(config)
            print(f"Wrote outputs to: {out_dir}")
            if not summary_df.empty:
                print(summary_df.to_string(index=False))
            return 0
    except Exception as exc:  # pragma: no cover - CLI convenience
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


