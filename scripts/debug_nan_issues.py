#!/usr/bin/env python3
"""
Script to debug NaN issues in satellite data loading.
This will help identify which specific dates and files are causing problems.
"""

import os
import glob
from datetime import datetime, timedelta
import scipy.io
import numpy as np

def debug_nan_issues():
    """Debug NaN issues by checking specific dates and file availability."""
    
    # Load the MATLAB file
    mat_file = "/unity/g2/jmiranda/nespreso_api/uses/Idalia_profiles_Aug2Sep2023.mat"
    if not os.path.exists(mat_file):
        print(f"ERROR: MATLAB file not found: {mat_file}")
        return
    
    print("Loading MATLAB file...")
    mat = scipy.io.loadmat(mat_file)
    dates = mat['timear'].flatten()
    latitudes = mat['latar'].flatten()
    longitudes = mat['lonar'].flatten()
    
    print(f"Found {len(dates)} points in dataset")
    print(f"Latitude range: {latitudes.min():.4f} to {latitudes.max():.4f}")
    print(f"Longitude range: {longitudes.min():.4f} to {longitudes.max():.4f}")
    
    # Convert MATLAB dates to Python dates
    def matlab_datenum_to_date(matlab_date):
        """Convert MATLAB datenum to Python datetime"""
        python_date = datetime(1, 1, 1) + timedelta(days=int(matlab_date) - 366)
        return python_date
    
    python_dates = [matlab_datenum_to_date(d) for d in dates]
    
    # Get unique dates
    unique_dates = list(set(python_dates))
    unique_dates.sort()
    
    print(f"\nUnique dates: {len(unique_dates)}")
    print(f"Date range: {unique_dates[0].strftime('%Y-%m-%d')} to {unique_dates[-1].strftime('%Y-%m-%d')}")
    
    # Check satellite data paths (adjust these to match your actual paths)
    sss_root = "/unity/g2/jmiranda/nespreso_api/data/sss"
    sst_root = "/unity/g2/jmiranda/nespreso_api/data/sst"
    aviso_root = "/unity/g2/jmiranda/nespreso_api/data/aviso"
    
    print(f"\nChecking satellite data availability:")
    print(f"SSS root: {sss_root}")
    print(f"SST root: {sst_root}")
    print(f"AVISO root: {aviso_root}")
    
    # Check each unique date in detail
    for i, date in enumerate(unique_dates[:5]):  # Check first 5 dates
        print(f"\n{'='*60}")
        print(f"DATE {i+1}: {date.strftime('%Y-%m-%d')}")
        print(f"{'='*60}")
        
        # Find points for this date
        date_indices = [j for j, d in enumerate(python_dates) if d == date]
        date_lats = latitudes[date_indices]
        date_lons = longitudes[date_indices]
        
        print(f"Points for this date: {len(date_indices)}")
        print(f"Latitude range: {date_lats.min():.4f} to {date_lats.max():.4f}")
        print(f"Longitude range: {date_lons.min():.4f} to {date_lons.max():.4f}")
        
        # Check SSS
        print(f"\n--- SSS Data ---")
        doy = date.timetuple().tm_yday
        sss_pattern = os.path.join(sss_root, f"{date.year:04d}", 
                                  f"RSS_smap_SSS_L3_8day_running_{date.year}_{doy:03d}_FNL_v*.nc")
        sss_files = glob.glob(sss_pattern)
        print(f"Pattern: {sss_pattern}")
        print(f"Files found: {len(sss_files)}")
        if sss_files:
            for f in sss_files:
                print(f"  ✓ {os.path.basename(f)}")
                # Check file size
                try:
                    size = os.path.getsize(f)
                    print(f"    Size: {size:,} bytes")
                except:
                    print(f"    Size: Unable to determine")
        else:
            print("  ✗ No SSS files found")
            # Try alternative patterns
            alt_patterns = [
                os.path.join(sss_root, f"{date.year:04d}", f"*{date.year}_{doy:03d}*.nc"),
                os.path.join(sss_root, f"{date.year:04d}", f"*{date.strftime('%Y%m%d')}*.nc"),
            ]
            for pattern in alt_patterns:
                alt_files = glob.glob(pattern)
                if alt_files:
                    print(f"  Alternative pattern {pattern}: {len(alt_files)} files")
                    for f in alt_files[:3]:  # Show first 3
                        print(f"    - {os.path.basename(f)}")
        
        # Check SST
        print(f"\n--- SST Data ---")
        sst_pattern = os.path.join(sst_root, f"{date.year:04d}", 
                                  f"{date.strftime('%Y%m%d')}090000-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1*.nc")
        sst_files = glob.glob(sst_pattern)
        print(f"Pattern: {sst_pattern}")
        print(f"Files found: {len(sst_files)}")
        if sst_files:
            for f in sst_files:
                print(f"  ✓ {os.path.basename(f)}")
                # Check file size
                try:
                    size = os.path.getsize(f)
                    print(f"    Size: {size:,} bytes")
                except:
                    print(f"    Size: Unable to determine")
        else:
            print("  ✗ No SST files found")
            # Try alternative patterns
            alt_patterns = [
                os.path.join(sst_root, f"{date.year:04d}", f"*{date.strftime('%Y%m%d')}*.nc"),
                os.path.join(sst_root, f"{date.year:04d}", f"*MUR*.nc"),
            ]
            for pattern in alt_patterns:
                alt_files = glob.glob(pattern)
                if alt_files:
                    print(f"  Alternative pattern {pattern}: {len(alt_files)} files")
                    for f in alt_files[:3]:  # Show first 3
                        print(f"    - {os.path.basename(f)}")
        
        # Check AVISO
        print(f"\n--- AVISO Data ---")
        aviso_file = os.path.join(aviso_root, f"{date.year}-{date.month:02d}.nc")
        aviso_exists = os.path.exists(aviso_file)
        print(f"File: {aviso_file}")
        print(f"Status: {'✓ EXISTS' if aviso_exists else '✗ MISSING'}")
        if aviso_exists:
            try:
                size = os.path.getsize(aviso_file)
                print(f"Size: {size:,} bytes")
            except:
                print(f"Size: Unable to determine")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print("This script helps identify:")
    print("1. Which dates have missing satellite data files")
    print("2. What the actual file patterns should be")
    print("3. Whether files exist but have different naming conventions")
    print("4. File sizes to check for corruption")
    print("\nNext steps:")
    print("1. Check if the satellite data paths are correct")
    print("2. Verify file naming conventions match the patterns")
    print("3. Ensure files are not corrupted (check file sizes)")
    print("4. Look for alternative file naming patterns")

if __name__ == "__main__":
    debug_nan_issues()
