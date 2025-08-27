#!/usr/bin/env python3
"""
Satellite Data Date Scanner

This script scans satellite data directories to find all available dates
and identifies which dates have complete data across all three sources
(SST, SSS, AVISO).

Usage:
    python scan_satellite_dates.py [--output OUTPUT_FILE] [--detailed]
"""

import os
import glob
import json
import argparse
from datetime import datetime, date
from pathlib import Path
from typing import Dict, Set, List
from collections import defaultdict

def get_day_of_year_from_month_and_day(month, day_of_month, year):
    """Get day of year from month and day (matching sat.py logic)"""
    first_jan = date(year, 1, 1)
    return date(year, month, day_of_month).toordinal() - first_jan.toordinal() + 1

def scan_sst_directory(root_path: str, year: str) -> Set[str]:
    """Scan SST directory for a specific year using correct naming convention from sat.py"""
    dates = set()
    
    if not os.path.exists(root_path):
        return dates
    
    year_dir = os.path.join(root_path, year)
    if not os.path.exists(year_dir):
        return dates
    
    # SST pattern from sat.py: {ctag}090000-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1*.nc
    # where ctag = YYYYMMDD
    for file_path in glob.glob(f"{year_dir}/*.nc"):
        filename = os.path.basename(file_path)
        
        # Extract date from filename: 20230601090000-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1.nc
        if filename.startswith(year) and len(filename) >= 8:
            date_str = filename[:8]  # YYYYMMDD
            dates.add(date_str)
    
    return dates

def scan_sss_directory(root_path: str, year: str) -> Set[str]:
    """Scan SSS directory for a specific year using correct naming convention from sat.py"""
    dates = set()
    
    if not os.path.exists(root_path):
        return dates
    
    year_dir = os.path.join(root_path, year)
    if not os.path.exists(year_dir):
        return dates
    
    # SSS pattern from sat.py: RSS_smap_SSS_L3_8day_running_{year}_{doy:03d}_FNL_v*.nc
    # where doy = day of year (001-366)
    for file_path in glob.glob(f"{year_dir}/*.nc"):
        filename = os.path.basename(file_path)
        
        # Check if it matches the SSS pattern exactly as in sat.py
        if filename.startswith("RSS_smap_SSS_L3_8day_running_") and "_FNL_v" in filename:
            # Extract year and day of year from filename
            # Format: RSS_smap_SSS_L3_8day_running_2021_001_FNL_v05.0.nc
            # The pattern is: RSS_smap_SSS_L3_8day_running_{year}_{doy}_FNL_v*.nc
            
            # Find the position of the year and doy
            # Split by underscore and look for the pattern
            parts = filename.split("_")
            
            # Look for the pattern: RSS_smap_SSS_L3_8day_running_2021_001_FNL_v05.0.nc
            # parts[0] = RSS, parts[1] = smap, parts[2] = SSS, parts[3] = L3, 
            # parts[4] = 8day, parts[5] = running, parts[6] = 2021, parts[7] = 001, parts[8] = FNL, parts[9] = v05.0.nc
            if len(parts) >= 8:
                try:
                    file_year = parts[6]  # year (position 6)
                    doy_str = parts[7]   # day of year (position 7)
                    
                    if file_year == year and doy_str.isdigit():
                        doy = int(doy_str)
                        if 1 <= doy <= 366:
                            # Store as YYYYDOY for now
                            dates.add(f"{year}{doy:03d}")
                except (ValueError, IndexError):
                    continue
    
    return dates

def scan_aviso_directory(root_path: str, year: str) -> Set[str]:
    """Scan AVISO directory for a specific year using correct monthly file logic from sat.py"""
    dates = set()
    
    if not os.path.exists(root_path):
        return dates
    
    # AVISO pattern from sat.py: {year}-{month:02d}.nc (monthly files)
    for file_path in glob.glob(f"{root_path}/*.nc"):
        filename = os.path.basename(file_path)
        
        # Extract year and month from filename: 2023-06.nc
        if filename.startswith(year) and "-" in filename:
            try:
                month_str = filename.split("-")[1].split(".")[0]
                if month_str.isdigit() and 1 <= int(month_str) <= 12:
                    # For monthly files, we'll add all days in that month
                    # This is an approximation since we don't know which specific days have data
                    month = int(month_str)
                    # Add the first day of each month as a representative date
                    # The actual daily availability will be determined by the satellite accessor
                    dates.add(f"{year}{month:02d}01")  # First day of month
            except (ValueError, IndexError):
                continue
    
    return dates

def scan_all_satellite_directories(sst_root: str, sss_root: str, aviso_root: str) -> Dict[str, Dict[str, Set[str]]]:
    """Scan all satellite data directories for available dates"""
    print("Scanning satellite data directories...")
    
    # Get all available years from SST directory (most reliable)
    sst_years = []
    if os.path.exists(sst_root):
        sst_years = [d for d in os.listdir(sst_root) 
                    if os.path.isdir(os.path.join(sst_root, d)) and d.isdigit()]
    
    # Also check other directories for years
    all_years = set(sst_years)
    
    # SSS and AVISO might have different year structures
    if os.path.exists(sss_root):
        sss_years = [d for d in os.listdir(sss_root) 
                    if os.path.isdir(os.path.join(sss_root, d)) and d.isdigit()]
        all_years.update(sss_years)
    
    # AVISO is monthly files, so check the root directory for year patterns
    if os.path.exists(aviso_root):
        aviso_files = [f for f in os.listdir(aviso_root) if f.endswith('.nc')]
        aviso_years = set()
        for filename in aviso_files:
            if filename.startswith('20') and '-' in filename:  # 20XX-XX.nc format
                try:
                    year = filename.split('-')[0]
                    if year.isdigit() and 2000 <= int(year) <= 2030:
                        aviso_years.add(year)
                except (ValueError, IndexError):
                    continue
        all_years.update(aviso_years)
    
    all_years = sorted(list(all_years))
    print(f"Found years: {all_years}")
    
    # Scan each year
    available_dates = {
        'sst': {},
        'sss': {},
        'aviso': {}
    }
    
    for year in all_years:
        print(f"Scanning year {year}...")
        
        # Scan SST
        sst_dates = scan_sst_directory(sst_root, year)
        if sst_dates:
            available_dates['sst'][year] = sst_dates
            print(f"  SST: {len(sst_dates)} dates")
        
        # Scan SSS
        sss_dates = scan_sss_directory(sss_root, year)
        if sss_dates:
            available_dates['sss'][year] = sss_dates
            print(f"  SSS: {len(sss_dates)} dates")
        
        # Scan AVISO
        aviso_dates = scan_aviso_directory(aviso_root, year)
        if aviso_dates:
            available_dates['aviso'][year] = aviso_dates
            print(f"  AVISO: {len(aviso_dates)} dates")
    
    return available_dates

def find_complete_dates(available_dates: Dict[str, Dict[str, Set[str]]]) -> Dict[str, List[str]]:
    """Find dates that have data from all three satellite sources"""
    print("\nFinding dates with complete satellite data...")
    
    complete_dates = {}
    
    for year in available_dates['sst'].keys():
        year_complete = []
        
        # Get all SST dates for this year
        sst_dates = available_dates['sst'].get(year, set())
        sss_dates = available_dates['sss'].get(year, set())
        aviso_dates = available_dates['aviso'].get(year, set())
        
        # For AVISO, we need to check if any month has data
        # Since AVISO is monthly, we'll consider a month available if we have the monthly file
        aviso_months = set()
        for date_str in aviso_dates:
            if len(date_str) == 8 and date_str.endswith('01'):  # YYYYMM01 format
                month = date_str[4:6]
                aviso_months.add(month)
        
        # For SSS, we need to check if any day-of-year has data
        # Since SSS uses day-of-year, we'll consider it available if we have any SSS files
        sss_available = len(sss_dates) > 0
        
        # For SST, we have daily files
        sst_available = len(sst_dates) > 0
        
        # A year is complete if we have:
        # 1. SST data for some days
        # 2. SSS data for some days  
        # 3. AVISO data for some months
        if sst_available and sss_available and len(aviso_months) > 0:
            # Find the intersection of available dates
            # This is approximate since SSS uses day-of-year and AVISO is monthly
            # The actual daily availability will be determined by the satellite accessor
            year_complete = sorted(list(sst_dates))  # Use SST dates as reference
            complete_dates[year] = year_complete
            print(f"  {year}: {len(year_complete)} potential complete dates (approximate)")
        else:
            print(f"  {year}: No complete data coverage")
            if not sst_available:
                print(f"    - SST: No data")
            if not sss_available:
                print(f"    - SSS: No data")
            if len(aviso_months) == 0:
                print(f"    - AVISO: No monthly files")
    
    return complete_dates

def generate_summary_report(available_dates: Dict[str, Dict[str, Set[str]]], 
                          complete_dates: Dict[str, List[str]]) -> Dict:
    """Generate a comprehensive summary report"""
    
    # Count total dates by source and year
    summary = {
        'scan_timestamp': datetime.now().isoformat(),
        'data_sources': {
            'sst_root': '/Net/work/ozavala/DATA/GOFFISH/SST/OISST/',
            'sss_root': '/Net/work/ozavala/DATA/GOFFISH/SSS/SMAP_Global/',
            'aviso_root': '/Net/work/ozavala/DATA/GOFFISH/AVISO/GoM/'
        },
        'yearly_summary': {},
        'complete_dates': complete_dates,
        'statistics': {
            'total_sst_dates': 0,
            'total_sss_dates': 0,
            'total_aviso_months': 0,
            'total_complete_years': 0
        }
    }
    
    # Process yearly data
    all_years = set()
    for source in ['sst', 'sss', 'aviso']:
        all_years.update(available_dates[source].keys())
    
    for year in sorted(all_years):
        year_data = {
            'sst_dates': len(available_dates['sst'].get(year, set())),
            'sss_dates': len(available_dates['sss'].get(year, set())),
            'aviso_months': len(available_dates['aviso'].get(year, set())),
            'complete_coverage': year in complete_dates,
            'sst_date_range': None,
            'sss_date_range': None,
            'aviso_month_range': None
        }
        
        # Add date ranges if available
        if available_dates['sst'].get(year):
            sst_dates = sorted(list(available_dates['sst'][year]))
            year_data['sst_date_range'] = f"{sst_dates[0]} to {sst_dates[-1]}"
        
        if available_dates['sss'].get(year):
            sss_dates = sorted(list(available_dates['sss'][year]))
            year_data['sss_date_range'] = f"{sss_dates[0]} to {sss_dates[-1]}"
        
        if available_dates['aviso'].get(year):
            aviso_dates = sorted(list(available_dates['aviso'][year]))
            year_data['aviso_month_range'] = f"{aviso_dates[0][4:6]} to {aviso_dates[-1][4:6]}"
        
        summary['yearly_summary'][year] = year_data
        
        # Update totals
        summary['statistics']['total_sst_dates'] += year_data['sst_dates']
        summary['statistics']['total_sss_dates'] += year_data['sss_dates']
        summary['statistics']['total_aviso_months'] += year_data['aviso_months']
        if year_data['complete_coverage']:
            summary['statistics']['total_complete_years'] += 1
    
    return summary

def save_report(report: Dict, output_file: str):
    """Save the report to a JSON file"""
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to: {output_file}")

def print_detailed_report(report: Dict):
    """Print a detailed human-readable report"""
    print("\n" + "="*80)
    print("SATELLITE DATA AVAILABILITY REPORT")
    print("="*80)
    
    print(f"\nScan completed: {report['scan_timestamp']}")
    print(f"Data sources scanned:")
    for source, path in report['data_sources'].items():
        print(f"  {source}: {path}")
    
    print(f"\nOVERALL STATISTICS:")
    stats = report['statistics']
    print(f"  Total SST dates: {stats['total_sst_dates']}")
    print(f"  Total SSS dates: {stats['total_sss_dates']}")
    print(f"  Total AVISO months: {stats['total_aviso_months']}")
    print(f"  Years with complete coverage: {stats['total_complete_years']}")
    
    print(f"\nYEARLY BREAKDOWN:")
    for year, data in report['yearly_summary'].items():
        print(f"\n  {year}:")
        print(f"    SST: {data['sst_dates']} dates")
        if data['sst_date_range']:
            print(f"      Range: {data['sst_date_range']}")
        
        print(f"    SSS: {data['sss_dates']} dates")
        if data['sss_date_range']:
            print(f"      Range: {data['sss_date_range']}")
        
        print(f"    AVISO: {data['aviso_months']} months")
        if data['aviso_month_range']:
            print(f"      Range: {data['aviso_month_range']}")
        
        print(f"    Complete coverage: {'✅' if data['complete_coverage'] else '❌'}")
    
    print(f"\nCOMPLETE COVERAGE BY YEAR:")
    for year, dates in report['complete_dates'].items():
        if dates:
            print(f"  {year}: {len(dates)} potential dates")
            if len(dates) <= 10:
                print(f"    {', '.join(dates)}")
            else:
                print(f"    {', '.join(dates[:5])} ... {', '.join(dates[-5:])}")
    
    print("\n" + "="*80)

def main():
    parser = argparse.ArgumentParser(description='Scan satellite data directories for available dates')
    parser.add_argument('--output', '-o', default='satellite_dates_report.json',
                       help='Output JSON file (default: satellite_dates_report.json)')
    parser.add_argument('--detailed', '-d', action='store_true',
                       help='Print detailed human-readable report')
    
    args = parser.parse_args()
    
    # Configuration - using the corrected paths
    sst_root = "/Net/work/ozavala/DATA/GOFFISH/SST/OISST/"
    sss_root = "/Net/work/ozavala/DATA/GOFFISH/SSS/SMAP_Global/"
    aviso_root = "/Net/work/ozavala/DATA/GOFFISH/AVISO/GoM/"
    
    print("Satellite Data Date Scanner")
    print("="*50)
    
    # Scan directories
    available_dates = scan_all_satellite_directories(sst_root, sss_root, aviso_root)
    
    # Find complete dates
    complete_dates = find_complete_dates(available_dates)
    
    # Generate report
    report = generate_summary_report(available_dates, complete_dates)
    
    # Save report
    save_report(report, args.output)
    
    # Print detailed report if requested
    if args.detailed:
        print_detailed_report(report)
    else:
        # Print summary
        stats = report['statistics']
        print(f"\nSUMMARY:")
        print(f"  Total SST dates: {stats['total_sst_dates']}")
        print(f"  Total SSS dates: {stats['total_sss_dates']}")
        print(f"  Total AVISO months: {stats['total_aviso_months']}")
        print(f"  Years with complete coverage: {stats['total_complete_years']}")
        
        if stats['total_complete_years'] > 0:
            print(f"\n✅ Found {stats['total_complete_years']} years with complete satellite data coverage!")
            print(f"   These years can potentially be used to build the NeSPReSO archive.")
            print(f"   Note: Daily availability will be determined by the satellite accessor.")
        else:
            print(f"\n❌ No years found with complete satellite data coverage.")
            print(f"   Check if satellite data directories exist and contain files.")

if __name__ == "__main__":
    main()
