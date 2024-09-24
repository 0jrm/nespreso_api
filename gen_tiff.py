import os
import numpy as np
import xarray as xr
import rasterio
from rasterio.transform import from_bounds

def export_to_geotiff(data_array, output_file, lat_min, lat_max, lon_min, lon_max, nodata_value=-9999):
    # Replace NaN with a specific nodata value (-9999)
    data_array = np.where(np.isnan(data_array), nodata_value, data_array)
    
    # Ensure data_array is of type float32
    data_array = data_array.astype(np.float32)
    
    # Flip the data array vertically to correct the upside-down issue
    data_array = np.flipud(data_array)
    
    # Define the transform and metadata
    transform = from_bounds(lon_min, lat_min, lon_max, lat_max, data_array.shape[1], data_array.shape[0])
    profile = {
        'driver': 'GTiff',
        'height': data_array.shape[0],
        'width': data_array.shape[1],
        'count': 1,  # Single band (ADT data)
        'dtype': 'float32',  # Match the data_array dtype
        'crs': 'EPSG:4326',  # WGS84 CRS
        'transform': transform,
        'nodata': nodata_value  # Set nodata value for transparency
    }

    # Write to GeoTIFF
    with rasterio.open(output_file, 'w', **profile) as dst:
        dst.write(data_array, 1)

# Function to process the data for a specific date
def process_and_export_by_date(selected_date, output_folder='/path/to/output/folder', input_folder='/unity/f1/ozavala/DATA/GOFFISH/AVISO/GoM/'):
    # Extract year and month from selected_date
    selected_year, selected_month = selected_date.split('-')[:2]
    
    # Construct the expected filename based on the year and month
    filename = f"{selected_year}-{selected_month}.nc"
    file_path = os.path.join(input_folder, filename)
    
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    # Load the dataset
    data = xr.load_dataset(file_path)
    
    # Latitude and longitude range for the bounding box
    lat_min, lat_max = 17, 32  
    lon_min, lon_max = -100, -80 
    
    # Select data for the specific date and bounding box
    subset = data.sel(
        time=selected_date,
        latitude=slice(lat_min, lat_max),
        longitude=slice(lon_min, lon_max)
    )

    # Check if 'adt' variable exists in the dataset and the subset has valid data
    if 'adt' in subset and not subset['adt'].isnull().all():
        adt_data = subset['adt'].squeeze().values  # Extract the 'adt' data
        
        # Generate the output filename and export the data to GeoTIFF
        output_file = os.path.join(output_folder, f"{filename.replace('.nc', '')}_{selected_date}.tiff")
        print(f"Data array shape: {adt_data.shape}")
        export_to_geotiff(adt_data, output_file, lat_min, lat_max, lon_min, lon_max)
        print(f"Exported {output_file}")
    else:
        print(f"No valid 'adt' data available for {selected_date} in {file_path}")

# Example usage

import datetime

# Define the start and end dates for May 2021
start_date = datetime.date(2021, 5, 1)
end_date = datetime.date(2021, 5, 31)

# Output folder
output_folder = '/unity/g2/jmiranda/nespreso_api/tiff'

# Loop through all dates in May 2021
current_date = start_date
while current_date <= end_date:
    date_str = current_date.strftime('%Y-%m-%d')
    print(f"Processing date: {date_str}")
    process_and_export_by_date(date_str, output_folder)
    current_date += datetime.timedelta(days=1)

print("Processing complete.")
