# profile_client.py
"""
NeSPReSO Profile API Client

A Python client for the NeSPReSO API that provides easy access to synthetic 
temperature and salinity profile predictions in the ocean.

Features:
- Single point and batch predictions
- Automatic batch processing for large datasets
- NetCDF file output
- Async and sync interfaces
- Automatic file merging for batch results

Example:
    from profile_client import get_predictions, get_predictions_batch
    
    # Single prediction
    result = get_predictions([25.0], [-83.0], ["2016-12-31"])
    
    # Batch processing for large datasets
    result = get_predictions_batch(latitudes, longitudes, dates, batch_size=100)
"""

from __future__ import annotations

import asyncio
import warnings
import httpx
import os
import numpy as np
from services.utils import preprocess_inputs, apply_netcdf_global_attributes

# Default API endpoint
DEFAULT_API = "http://0.0.0.0:5000/v1/profile"

# Default timeout settings
DEFAULT_TIMEOUT = 1800  # 30 minutes
DEFAULT_CONNECT_TIMEOUT = 10.0  # 10 seconds


async def fetch_predictions(lat, lon, date, filename="output.nc", api_url=None):
    """
    Fetch predictions from the NeSPReSO API asynchronously.
    
    Args:
        lat: Latitude values (single value, list, or array)
        lon: Longitude values (single value, list, or array)
        date: Date values in YYYY-MM-DD format (single value, list, or array)
        filename: Output NetCDF filename
        api_url: API endpoint URL (defaults to DEFAULT_API)
    
    Returns:
        str: Output filename on success, None on failure
    
    Raises:
        httpx.RequestError: For HTTP request failures
        IOError: For file writing failures
    """
    api_url = api_url or DEFAULT_API
    
    # Warn about deprecated endpoint
    if api_url.endswith("/predict"):
        warnings.warn("You are using the deprecated /predict endpoint. Use /v1/profile.")

    # Prepare request data
    data = {"lat": lat, "lon": lon, "date": date}
    timeout = httpx.Timeout(DEFAULT_TIMEOUT, connect=DEFAULT_CONNECT_TIMEOUT)
    
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
            response = await client.post(api_url, json=data)
    except Exception as e:
        print(f"HTTP error: {e}")
        return None

    # Check response status
    if response.status_code != 200:
        print(f"Request failed: {response.status_code} – {response.text[:200]}")
        return None
    
    # Verify content type
    if not response.headers.get("Content-Type", "").startswith("application/x-netcdf"):
        print("Unexpected content type:", response.headers.get("Content-Type"))
        return None

    # Write NetCDF file
    try:
        with open(filename, "wb") as f:
            f.write(response.content)
    except Exception as e:
        print(f"Failed to write {filename}: {e}")
        return None
    
    return filename


def get_predictions(lat, lon, date, filename="output.nc", api_url=None):
    """
    Synchronous wrapper for fetch_predictions.
    
    This function automatically handles both synchronous and asynchronous contexts.
    If called from within an async context, it schedules the task and waits for completion.
    Otherwise, it runs the async function in a new event loop.
    
    Args:
        lat: Latitude values (single value, list, or array)
        lon: Longitude values (single value, list, or array)
        date: Date values in YYYY-MM-DD format (single value, list, or array)
        filename: Output NetCDF filename
        api_url: API endpoint URL (defaults to DEFAULT_API)
    
    Returns:
        str: Output filename on success, None on failure
    """
    # Preprocess inputs (handles various input types)
    lat, lon, date = preprocess_inputs(lat, lon, date)
    print(f"Fetching predictions for {len(lat)} points...")
    
    # Check if we're in an async context
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Notebook/async host: schedule task and block until done
        return asyncio.run_coroutine_threadsafe(
            fetch_predictions(lat, lon, date, filename, api_url=api_url),
            loop
        ).result()
    else:
        # Standard sync context: run in new event loop
        return asyncio.run(fetch_predictions(lat, lon, date, filename, api_url=api_url))


def get_predictions_batch(lat, lon, date, batch_size=1000, filename_prefix="output", 
                         api_url=None, merge_output=True):
    """
    Process large datasets in batches to avoid memory issues.
    
    This function automatically splits large datasets into smaller batches,
    processes each batch separately, and optionally merges the results
    into a single output file.
    
    Args:
        lat: Latitude values (list or array)
        lon: Longitude values (list or array)
        date: Date values in YYYY-MM-DD format (list or array)
        batch_size: Number of points to process in each batch (default: 1000)
        filename_prefix: Prefix for batch filenames (default: "output")
        api_url: API endpoint URL (defaults to DEFAULT_API)
        merge_output: Whether to merge batch files into a single output (default: True)
    
    Returns:
        str or list: Single merged filename if merge_output=True and successful,
                    list of batch filenames otherwise
    """
    # Preprocess inputs
    lat, lon, date = preprocess_inputs(lat, lon, date)
    total_points = len(lat)
    print(f"Processing {total_points} points in batches of {batch_size}")
    
    successful_files = []
    
    # Process each batch
    for i in range(0, total_points, batch_size):
        end_idx = min(i + batch_size, total_points)
        batch_lat = lat[i:end_idx]
        batch_lon = lon[i:end_idx]
        batch_date = date[i:end_idx]
        
        batch_filename = f"{filename_prefix}_batch_{i//batch_size + 1:03d}.nc"
        print(f"Processing batch {i//batch_size + 1} ({i+1}-{end_idx} of {total_points})")
        
        try:
            result = get_predictions(batch_lat, batch_lon, batch_date, 
                                   filename=batch_filename, api_url=api_url)
            if result:
                successful_files.append(result)
                print(f"Batch {i//batch_size + 1} completed successfully: {result}")
            else:
                print(f"Batch {i//batch_size + 1} failed")
        except Exception as e:
            print(f"Batch {i//batch_size + 1} failed with error: {e}")
    
    print(f"Completed {len(successful_files)} out of {(total_points + batch_size - 1) // batch_size} batches")
    
    # Handle output based on merge setting and results
    if merge_output and len(successful_files) > 1:
        # Determine final output filename
        if filename_prefix.endswith('.nc'):
            final_output = filename_prefix
        else:
            final_output = f"{filename_prefix}.nc"
        
        print(f"Auto-merging {len(successful_files)} batch files into {final_output}")
        merged_file = merge_netcdf_files(successful_files, final_output)
        
        if merged_file:
            print(f"Successfully created merged output: {merged_file}")
            return merged_file
        else:
            print("Failed to merge files, returning batch file list")
            return successful_files
    elif len(successful_files) == 1:
        # Only one batch, return the single file
        return successful_files[0]
    else:
        # No successful batches or merge not requested
        return successful_files


def merge_netcdf_files(file_list, output_filename):
    """
    Merge multiple NetCDF files into a single file.
    
    This function concatenates multiple NetCDF files along the profile_number
    dimension and reindexes the profile numbers to be sequential.
    
    Args:
        file_list: List of NetCDF filenames to merge
        output_filename: Output filename for merged file
    
    Returns:
        str: Output filename on success, None on failure
    
    Note:
        Requires xarray to be installed. All input files must have the same
        structure and variables.
    """
    try:
        import xarray as xr
    except ImportError:
        print("xarray is required for merging NetCDF files. Install with: pip install xarray")
        return None
    
    if not file_list:
        print("No files to merge")
        return None
    
    print(f"Merging {len(file_list)} NetCDF files into {output_filename}")
    
    datasets = []
    try:
        # Open all datasets
        for i, filename in enumerate(file_list):
            print(f"  Loading file {i+1}/{len(file_list)}: {os.path.basename(filename)}")
            ds = xr.open_dataset(filename)
            datasets.append(ds)
        
        # Concatenate along profile_number dimension
        print("  Concatenating datasets...")
        merged_ds = xr.concat(datasets, dim='profile_number')
        
        # Reindex profile_number to be sequential
        merged_ds = merged_ds.assign_coords(profile_number=np.arange(len(merged_ds.profile_number)))
        
        # Write merged dataset
        print(f"  Writing merged dataset to {output_filename}")
        # Ensure global attributes on merged output
        merged_ds = apply_netcdf_global_attributes(merged_ds)
        merged_ds.to_netcdf(output_filename)
        
        print(f"Successfully merged {len(file_list)} files into {output_filename}")
        return output_filename
        
    except Exception as e:
        print(f"Error merging files: {e}")
        return None
    finally:
        # Clean up - close all datasets
        for ds in datasets:
            try:
                ds.close()
            except:
                pass


# Example usage
if __name__ == "__main__":
    # Simple example with a few points
    print("=== NeSPReSO Client Example ===")
    
    # Example coordinates and dates
    latitudes = [25.0, 26.0, 27.0]
    longitudes = [-83.0, -84.0, -85.0]
    date_strings = ["2016-12-31", "2016-12-30", "2016-12-29"]
    output_file = "example_output.nc"
    
    print(f"Processing {len(latitudes)} points...")
    print(f"Latitude range: {min(latitudes):.1f} to {max(latitudes):.1f}")
    print(f"Longitude range: {min(longitudes):.1f} to {max(longitudes):.1f}")
    print(f"Date range: {min(date_strings)} to {max(date_strings)}")
    
    # Use batch processing for large datasets
    if len(latitudes) > 100:
        print("Large dataset detected, using batch processing...")
        result = get_predictions_batch(latitudes, longitudes, date_strings, 
                                     batch_size=100, 
                                     filename_prefix="example_batch",
                                     merge_output=True)
    else:
        result = get_predictions(latitudes, longitudes, date_strings, filename=output_file)
    
    print(f"\nResult: {result}")
    
    # Handle the result
    if isinstance(result, list):
        if len(result) == 1:
            output_file = result[0]
            print(f"Single output file: {output_file}")
        else:
            print(f"Multiple batch files created: {len(result)} files")
            output_file = None
    else:
        output_file = result
        print(f"Output file: {output_file}")
    
    # Try to read and display file information
    if output_file and os.path.exists(output_file):
        try:
            import xarray as xr
            with xr.open_dataset(output_file) as ds:
                print(f"\nFile successfully created:")
                print(f"  Temperature shape: {ds.Temperature.shape}")
                print(f"  Salinity shape: {ds.Salinity.shape}")
                print(f"  File size: {os.path.getsize(output_file)} bytes")
        except ImportError:
            print("xarray not available for file inspection")
        except Exception as e:
            print(f"Error reading output file: {e}")
    else:
        print("Failed to get predictions - output file not created")