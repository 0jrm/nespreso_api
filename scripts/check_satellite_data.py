#!/usr/bin/env python3
"""
Script to check satellite data availability for the Idalia profiles dataset.
"""

import os
import glob
from datetime import datetime, timedelta
import scipy.io
import numpy as np

def check_satellite_data():
    """Check if satellite data files exist for the dates in the dataset."""
    
    # Load the MATLAB file
    mat_file = "/unity/g2/jmiranda/nespreso_api/uses/Idalia_profiles_Aug2Sep2023.mat"
    if not os.path.exists(mat_file):
        print(f"ERROR: MATLAB file not found: {mat_file}")
        return
    
    print("Loading MATLAB file...")
    mat = scipy.io.loadmat(mat_file)
    dates = mat['timear'].flatten()
    
    print(f"Found {len(dates)} dates in dataset")
    
    # Convert MATLAB dates to Python dates
    def matlab_datenum_to_date(matlab_date):
        """Convert MATLAB datenum to Python datetime"""
        python_date = datetime(1, 1, 1) + timedelta(days=int(matlab_date) - 366)
        return python_date
    
    python_dates = [matlab_datenum_to_date(d) for d in dates]
    
    # Get unique dates
    unique_dates = list(set(python_dates))
    unique_dates.sort()
    
    print(f"Unique dates: {len(unique_dates)}")
    print(f"Date range: {unique_dates[0].strftime('%Y-%m-%d')} to {unique_dates[-1].strftime('%Y-%m-%d')}")
    
    # Check satellite data paths (you may need to adjust these)
    sss_root = "/unity/g2/jmiranda/nespreso_api/data/sss"  # Adjust path as needed
    sst_root = "/unity/g2/jmiranda/nespreso_api/data/sst"  # Adjust path as needed
    aviso_root = "/unity/g2/jmiranda/nespreso_api/data/aviso"  # Adjust path as needed
    
    print(f"\nChecking satellite data availability:")
    print(f"SSS root: {sss_root}")
    print(f"SST root: {sst_root}")
    print(f"AVISO root: {aviso_root}")
    
    # Check each unique date
    for date in unique_dates[:10]:  # Check first 10 dates
        print(f"\n--- {date.strftime('%Y-%m-%d')} ---")
        
        # Check SSS
        doy = date.timetuple().tm_yday
        sss_pattern = os.path.join(sss_root, f"{date.year:04d}", 
                                  f"RSS_smap_SSS_L3_8day_running_{date.year}_{doy:03d}_FNL_v*.nc")
        sss_files = glob.glob(sss_pattern)
        print(f"SSS: {len(sss_files)} files found")
        if sss_files:
            print(f"  Files: {[os.path.basename(f) for f in sss_files]}")
        
        # Check SST
        sst_pattern = os.path.join(sst_root, f"{date.year:04d}", 
                                  f"{date.strftime('%Y%m%d')}090000-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1*.nc")
        sst_files = glob.glob(sst_pattern)
        print(f"SST: {len(sst_files)} files found")
        if sst_files:
            print(f"  Files: {[os.path.basename(f) for f in sst_files]}")
        
        # Check AVISO
        aviso_file = os.path.join(aviso_root, f"{date.year}-{date.month:02d}.nc")
        aviso_exists = os.path.exists(aviso_file)
        print(f"AVISO: {'EXISTS' if aviso_exists else 'MISSING'}")
        if aviso_exists:
            print(f"  File: {os.path.basename(aviso_file)}")
    
    # Check if paths exist
    print(f"\n--- Path Existence Check ---")
    print(f"SSS root exists: {os.path.exists(sss_root)}")
    print(f"SST root exists: {os.path.exists(sst_root)}")
    print(f"AVISO root exists: {os.path.exists(aviso_root)}")
    
    # List some files in each directory to see the structure
    for name, path in [("SSS", sss_root), ("SST", sst_root), ("AVISO", aviso_root)]:
        if os.path.exists(path):
            print(f"\n{name} directory contents (first 5 items):")
            try:
                items = os.listdir(path)[:5]
                for item in items:
                    item_path = os.path.join(path, item)
                    if os.path.isdir(item_path):
                        print(f"  {item}/ (dir)")
                    else:
                        print(f"  {item}")
            except Exception as e:
                print(f"  Error listing directory: {e}")
        else:
            print(f"\n{name} directory does not exist")

if __name__ == "__main__":
    check_satellite_data()
