#!/usr/bin/env python3
"""
NeSPReSO Grid Endpoint Client

A Python client for the NeSPReSO grid endpoint that allows you to query all 
predefined grid points for a single date, with optional BBOX filtering.

Features:
- Query full grid or filtered regions using BBOX
- Process multiple dates efficiently
- Automatic output directory management
- Comprehensive error handling and reporting
- Progress tracking for batch operations

Example:
    from grid_client import query_grid, query_multiple_dates
    
    # Query full grid for a date
    result = query_grid("2016-12-31")
    
    # Query with BBOX filtering
    gulf_bbox = [-95.0, 18.0, -80.0, 31.0]
    result = query_grid("2016-12-31", bbox=gulf_bbox)
    
    # Process multiple dates
    dates = ["2016-12-29", "2016-12-30", "2016-12-31"]
    summary = query_multiple_dates(dates, bbox=gulf_bbox)
"""

import requests
import json
from datetime import datetime, timedelta
import os

# Configuration
GRID_OUTPUT_DIR = os.path.join("uses", "grid")
DEFAULT_API_URL = "http://localhost:5000/v1/profile/grid"
DEFAULT_TIMEOUT = 600  # 10 minutes


def ensure_grid_output_dir():
    """Create the grid output directory if it doesn't exist."""
    if not os.path.exists(GRID_OUTPUT_DIR):
        os.makedirs(GRID_OUTPUT_DIR, exist_ok=True)
        print(f"Created output directory: {GRID_OUTPUT_DIR}")


def query_grid(date_str, bbox=None, api_url=DEFAULT_API_URL):
    """
    Query the grid endpoint for a specific date, optionally with BBOX filtering.
    
    Args:
        date_str (str): Date in YYYY-MM-DD format
        bbox (list, optional): Bounding box [lon_min, lat_min, lon_max, lat_max]
        api_url (str): Base URL of the API
    
    Returns:
        dict: Response information with success status and details
    """
    ensure_grid_output_dir()

    # Prepare the request
    request_data = {"date": date_str}
    if bbox is not None:
        request_data["bbox"] = bbox
        print(f"Querying grid for date: {date_str} with BBOX: {bbox}")
    else:
        print(f"Querying grid for date: {date_str} (full grid)")

    print(f"API endpoint: {api_url}")

    try:
        # Make the request
        response = requests.post(api_url, json=request_data, timeout=DEFAULT_TIMEOUT)

        if response.status_code == 200:
            # Success - save the NetCDF file
            if bbox:
                bbox_str = f"_bbox_{bbox[0]:.2f}_{bbox[1]:.2f}_{bbox[2]:.2f}_{bbox[3]:.2f}"
                output_filename = f"nespreso_grid_{date_str}{bbox_str}.nc"
            else:
                output_filename = f"nespreso_grid_{date_str}.nc"

            output_path = os.path.join(GRID_OUTPUT_DIR, output_filename)
            with open(output_path, 'wb') as f:
                f.write(response.content)

            print(f"✅ Grid query successful!")
            print(f"Output saved to: {output_path}")
            print(f"File size: {len(response.content)} bytes")

            return {
                "success": True,
                "filename": output_path,
                "size_bytes": len(response.content),
                "status_code": response.status_code
            }

        else:
            # Error response
            print(f"❌ Grid query failed with status {response.status_code}")
            try:
                error_data = response.json()
                print(f"Error details: {json.dumps(error_data, indent=2)}")
            except:
                print(f"Error response: {response.text}")

            return {
                "success": False,
                "status_code": response.status_code,
                "error": response.text
            }

    except requests.exceptions.Timeout:
        print("❌ Request timed out (10 minutes)")
        return {"success": False, "error": "Request timed out"}
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return {"success": False, "error": str(e)}
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return {"success": False, "error": str(e)}


def query_multiple_dates(date_list, bbox=None, api_url=DEFAULT_API_URL):
    """
    Query the grid endpoint for multiple dates, optionally with BBOX filtering.
    
    Args:
        date_list (list): List of date strings in YYYY-MM-DD format
        bbox (list, optional): Bounding box [lon_min, lat_min, lon_max, lat_max]
        api_url (str): Base URL of the API
    
    Returns:
        dict: Summary of results with success/failure counts
    """
    bbox_info = f" with BBOX {bbox}" if bbox else ""
    print(f"Querying grid for {len(date_list)} dates{bbox_info}...")

    results = []
    successful = 0
    failed = 0

    for i, date_str in enumerate(date_list, 1):
        print(f"\n[{i}/{len(date_list)}] Processing date: {date_str}")

        result = query_grid(date_str, bbox, api_url)
        results.append(result)

        if result["success"]:
            successful += 1
        else:
            failed += 1

    # Summary
    print(f"\n=== Summary ===")
    print(f"Total dates: {len(date_list)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

    return {
        "total": len(date_list),
        "successful": successful,
        "failed": failed,
        "results": results
    }


def generate_date_range(start_date, end_date):
    """
    Generate a list of dates between start_date and end_date.
    
    Args:
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
    
    Returns:
        list: List of date strings
    """
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    date_list = []
    current = start
    while current <= end:
        date_list.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)

    return date_list


def get_common_bbox_regions():
    """
    Get commonly used BBOX regions for the Gulf of Mexico.
    
    Returns:
        dict: Dictionary of named BBOX regions
    """
    return {
        "full_gulf": [-97.0, 18.0, -82.0, 31.0],
        "western_gulf": [-97.0, 20.0, -90.0, 29.0],
        "eastern_gulf": [-90.0, 20.0, -82.0, 29.0],
        "northern_gulf": [-97.0, 25.0, -82.0, 29.0],
        "southern_gulf": [-97.0, 18.0, -82.0, 25.0],
        "florida_straits": [-82.0, 24.0, -79.0, 26.0],
        "yucatan_channel": [-87.0, 20.0, -84.0, 22.0]
    }


if __name__ == "__main__":
    print("=== NeSPReSO Grid Client Examples ===\n")
    
    # Get common BBOX regions
    bbox_regions = get_common_bbox_regions()
    
    # Example 1: Full Grid Query
    print("=== Example 1: Full Grid Query ===")
    print("Querying full grid for 2016-12-31")
    try:
        result = query_grid("2016-12-31")
        if result["success"]:
            print("✅ Full grid query completed successfully")
        else:
            print("❌ Full grid query failed")
    except Exception as e:
        print(f"❌ Full grid query failed: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # Example 2: BBOX Filtered Query
    print("=== Example 2: BBOX Filtered Query ===")
    print("Querying Gulf of Mexico region for 2017-01-01")
    gulf_bbox = bbox_regions["western_gulf"]
    try:
        result = query_grid("2017-01-01", bbox=gulf_bbox)
        if result["success"]:
            print("✅ BBOX filtered query completed successfully")
        else:
            print("❌ BBOX filtered query failed")
    except Exception as e:
        print(f"❌ BBOX filtered query failed: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # Example 3: Multiple Dates
    print("=== Example 3: Multiple Dates Query ===")
    print("Querying multiple dates in December 2016")
    dates = ["2016-12-29", "2016-12-30", "2016-12-31"]
    try:
        summary = query_multiple_dates(dates)
        if summary["successful"] > 0:
            print("✅ Multiple dates query completed successfully")
        else:
            print("❌ All multiple dates queries failed")
    except Exception as e:
        print(f"❌ Multiple dates query failed: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # Example 4: Date Range Generation
    print("=== Example 4: Date Range Generation ===")
    print("Generating date range for December 2016")
    try:
        date_range = generate_date_range("2016-12-01", "2016-12-31")
        print(f"Generated {len(date_range)} dates: {date_range[0]} to {date_range[-1]}")
        print("✅ Date range generation successful")
    except Exception as e:
        print(f"❌ Date range generation failed: {e}")
    
    print("\n" + "="*60)
    print("SUMMARY:")
    print("="*60)
    print("✅ Full grid queries for complete spatial coverage")
    print("✅ BBOX filtering for regional analysis")
    print("✅ Multiple date processing for time series")
    print("✅ Automatic output directory management")
    print("✅ Comprehensive error handling and reporting")
    print("✅ Common BBOX regions for Gulf of Mexico")
