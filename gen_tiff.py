import os
import numpy as np
import xarray as xr
import rioxarray # Using rioxarray to export to GeoTIFF instead of raw rasterio

def export_to_geotiff(dataset:xr.Dataset, output_file:str, nodata_value:int=-9999):
    # Assign the CRS
    dataset.rio.write_crs("epsg:4326", inplace=True)

    # Write to GeoTIFF
    dataset.rio.to_raster(
        output_file, 
        driver='GTiff', 
        dtype='float32', 
        nodata=nodata_value)

# Function to process the data for a specific date
def process_and_export_by_date(selected_date:str, output_folder:str='./', input_folder:str='/unity/f1/ozavala/DATA/GOFFISH/AVISO/GoM/'):
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
        adt_data = subset['adt']
        # Generate the output filename and export the data to GeoTIFF
        output_file = os.path.join(output_folder, f"{filename.replace('.nc', '')}_{selected_date}.tiff")
        print(f"Data array shape: {adt_data.shape}")
        export_to_geotiff(adt_data, output_file)
        print(f"Exported {output_file}")
    else:
        print(f"No valid 'adt' data available for {selected_date} in {file_path}")

# Example usage

import datetime

# Define the start and end dates for May 2021
start_date = datetime.date(2021, 5, 1)
end_date = datetime.date(2021, 5, 31)

# Output folder
output_folder = '../tiff_test'

# Loop through all dates in May 2021
current_date = start_date
while current_date <= end_date:
    date_str = current_date.strftime('%Y-%m-%d')
    print(f"Processing date: {date_str}")
    process_and_export_by_date(date_str, output_folder)
    current_date += datetime.timedelta(days=1)

print("Processing complete.")
