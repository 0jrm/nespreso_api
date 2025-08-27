# NeSPReSO Usage Guide

A quick reference guide for common NeSPReSO API usage patterns.

## Quick Examples

### 1. Single Point Prediction

```python
from profile_client import get_predictions

# Get temperature/salinity profile for one location
result = get_predictions(
    lat=[25.0],           # Latitude
    lon=[-83.0],          # Longitude  
    date=["2016-12-31"],  # Date
    filename="my_profile.nc"
)

print(f"Profile saved to: {result}")
```

### 2. Multiple Points

```python
# Get profiles for multiple locations
latitudes = [25.0, 26.0, 27.0]
longitudes = [-83.0, -84.0, -85.0]
dates = ["2016-12-31", "2016-12-30", "2016-12-29"]

result = get_predictions(latitudes, longitudes, dates, filename="multiple_profiles.nc")
```

### 3. Large Dataset (Automatic Batching)

```python
from profile_client import get_predictions_batch

# Process thousands of points automatically
result = get_predictions_batch(
    latitudes, longitudes, dates,
    batch_size=100,           # Process 100 points at a time
    filename_prefix="dataset", # Output: dataset_batch_001.nc, dataset_batch_002.nc, etc.
    merge_output=True          # Automatically merge into single file
)

# Result will be either a single filename or list of batch files
if isinstance(result, str):
    print(f"All data merged into: {result}")
else:
    print(f"Batch files created: {result}")
```

### 4. Complete Grid Coverage

```python
from grid_client import query_grid

# Get all grid points for a date (complete Gulf of Mexico coverage)
result = query_grid("2016-12-31")

if result["success"]:
    print(f"Grid data saved to: {result['filename']}")
    print(f"File size: {result['size_bytes']} bytes")
```

### 5. Regional Grid (BBOX Filtering)

```python
from grid_client import query_grid, get_common_bbox_regions

# Get predefined regions
bbox_regions = get_common_bbox_regions()

# Query specific region
western_gulf = bbox_regions["western_gulf"]  # [-97.0, 20.0, -90.0, 29.0]
result = query_grid("2016-12-31", bbox=western_gulf)

# Or define custom region
custom_bbox = [-95.0, 18.0, -80.0, 31.0]  # [lon_min, lat_min, lon_max, lat_max]
result = query_grid("2016-12-31", bbox=custom_bbox)
```

### 6. Multiple Dates

```python
from grid_client import query_multiple_dates, generate_date_range

# Generate date sequence
dates = generate_date_range("2016-12-01", "2016-12-31")

# Process all dates
summary = query_multiple_dates(dates, bbox=western_gulf)
print(f"Successfully processed {summary['successful']} out of {summary['total']} dates")
```

## Common BBOX Regions

```python
from grid_client import get_common_bbox_regions

bbox_regions = get_common_bbox_regions()

# Available regions:
# full_gulf: [-97.0, 18.0, -82.0, 31.0]      # Complete Gulf of Mexico
# western_gulf: [-97.0, 20.0, -90.0, 29.0]    # Western Gulf
# eastern_gulf: [-90.0, 20.0, -82.0, 29.0]    # Eastern Gulf  
# northern_gulf: [-97.0, 25.0, -82.0, 29.0]   # Northern Gulf
# southern_gulf: [-97.0, 18.0, -82.0, 25.0]   # Southern Gulf
# florida_straits: [-82.0, 24.0, -79.0, 26.0] # Florida Straits
# yucatan_channel: [-87.0, 20.0, -84.0, 22.0] # Yucatan Channel
```

## File Outputs

### Profile Endpoint Files
- **Format**: NetCDF with `profile_number` dimension
- **Variables**: Temperature, Salinity, SSS, SST, AVISO, depth, lat, lon, time
- **Use case**: Individual point analysis, specific location studies

### Grid Endpoint Files  
- **Format**: NetCDF with `depth`, `lat`, `lon` dimensions
- **Variables**: Temperature(depth, lat, lon), Salinity(depth, lat, lon), SSS(lat, lon), SST(lat, lon), AVISO(lat, lon)
- **Use case**: Spatial analysis, visualization, GIS integration

## Performance Tips

- **Small datasets** (< 100 points): Use `get_predictions()`
- **Large datasets** (> 100 points): Use `get_predictions_batch()`
- **Regional analysis**: Use BBOX filtering to reduce processing time
- **Multiple dates**: Process sequentially to avoid overwhelming the server
- **Optimal batch size**: 100-1000 points (adjust based on server capacity)

## Error Handling

```python
# Check for success
if result["success"]:
    print(f"File saved: {result['filename']}")
else:
    print(f"Error: {result['error']}")
    print(f"Status code: {result['status_code']}")

# For profile predictions
if result is None:
    print("Prediction failed")
else:
    print(f"Success: {result}")
```

## Common Issues

1. **Connection errors**: Check if API server is running
2. **Timeout errors**: Large requests may take 5-15 minutes
3. **Input validation**: Ensure coordinates are valid (lat: -90 to 90, lon: -180 to 180)
4. **Date format**: Use YYYY-MM-DD format
5. **File permissions**: Ensure output directory is writable

## Quick Test

```bash
# Test profile client
python profile_client.py

# Test grid client  
python grid_client.py
```
