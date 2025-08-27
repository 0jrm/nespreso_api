# NeSPReSO Grid Endpoint

This document describes the new grid query endpoint that allows you to query all predefined grid points for a single date, with optional BBOX filtering.

## Overview

The grid endpoint (`/v1/profile/grid`) provides a way to query the NeSPReSO model for all grid points defined in the `nespreso_grid_and_mask.pkl` file. This is useful for generating complete spatial coverage of the Gulf of Mexico region for a specific date. The endpoint now supports optional BBOX filtering to limit the spatial extent of the query.

## Endpoint Details

- **URL**: `/v1/profile/grid`
- **Method**: `POST`
- **Content-Type**: `application/json`

## Request Format

### Basic Request (Full Grid)
```json
{
  "date": "2023-01-15"
}
```

### Request with BBOX Filtering
```json
{
  "date": "2023-01-15",
  "bbox": [-95.0, 20.0, -85.0, 28.0]
}
```

### Parameters

- `date` (string, required): Date in YYYY-MM-DD format
- `bbox` (array, optional): Bounding box [lon_min, lat_min, lon_max, lat_max]
  - `lon_min`: Minimum longitude (-180 to 180)
  - `lat_min`: Minimum latitude (-90 to 90)
  - `lon_max`: Maximum longitude (-180 to 180)
  - `lat_max`: Maximum latitude (-90 to 90)

## Response

The endpoint returns a NetCDF file containing:
- Temperature profiles for all grid points
- Salinity profiles for all grid points
- Surface data (SSS, SST, AVISO) for all grid points
- **LAT, LON, and TIME as dimensions** for better visualization
- Coordinates and metadata

### Dataset Structure

The NetCDF files now have a proper gridded structure with the dimensions you requested:
- **Dimensions**: `depth`, `lat`, `lon`
- **Coordinates**: 
  - `depth`: Depth levels (0-1800m)
  - `lat`: Latitude coordinates (sorted, unique values)
  - `lon`: Longitude coordinates (sorted, unique values)
  - `time`: Single time value for the entire dataset
- **Variables**:
  - `Temperature(depth, lat, lon)`: Temperature profiles on 3D grid
  - `Salinity(depth, lat, lon)`: Salinity profiles on 3D grid
  - `SSS(lat, lon)`: Sea Surface Salinity on 2D grid
  - `SST(lat, lon)`: Sea Surface Temperature on 2D grid
  - `AVISO(lat, lon)`: AVISO data on 2D grid

**Grid Information:**
- **Full Gulf of Mexico**: 41×59 grid (1,018 valid points)
- **Typical BBOX regions**: 20×30 to 40×50 grids (200-800 valid points)
- **Grid spacing**: 0.25° resolution
- **Coordinate ranges**: Lat 19.0° to 29.0°, Lon -97.0° to -82.5°

### Response Headers

- `Content-Type`: `application/x-netcdf`
- `Content-Disposition`: `attachment; filename=NeSPReSO_grid_YYYY-MM-DD.nc` or `attachment; filename=NeSPReSO_grid_YYYY-MM-DD_bbox_lon_min_lat_min_lon_max_lat_max.nc`
- `MODEL_SHA`: Model version identifier
- `STATS_SHA`: Statistics version identifier
- `SAT_SNAPSHOT`: Satellite data version identifier

## Grid Data

The grid coordinates are loaded from `nespreso_grid_and_mask.pkl` which contains:
- `lon_grid`: Longitude coordinates for all grid points
- `lat_grid`: Latitude coordinates for all grid points  
- `inside_mask`: Boolean mask indicating which points are inside the region of interest

Only points where `inside_mask` is `True` are processed. If a BBOX is provided, only points within the bounding box are included.

## BBOX Filtering

The BBOX parameter allows you to limit the spatial extent of your query:

### BBOX Format
```json
"bbox": [lon_min, lat_min, lon_max, lat_max]
```

### Example BBOX Values
- **Full Gulf of Mexico**: No BBOX parameter (default)
- **Western Gulf**: `[-97.0, 20.0, -90.0, 29.0]`
- **Eastern Gulf**: `[-90.0, 20.0, -82.0, 29.0]`
- **Northern Gulf**: `[-97.0, 25.0, -82.0, 29.0]`
- **Southern Gulf**: `[-97.0, 18.0, -82.0, 25.0]`

### BBOX Validation
- Longitude values must be between -180 and 180
- Latitude values must be between -90 and 90
- `lon_min` must be less than `lon_max`
- `lat_min` must be less than `lat_max`

## Usage Examples

### Python Client

```python
import requests

# Query full grid for a specific date
response = requests.post(
    "http://localhost:5000/v1/profile/grid",
    json={"date": "2023-01-15"}
)

# Query grid with BBOX filtering
response_bbox = requests.post(
    "http://localhost:5000/v1/profile/grid",
    json={
        "date": "2023-01-15",
        "bbox": [-95.0, 20.0, -85.0, 28.0]
    }
)

if response.status_code == 200:
    # Save the NetCDF file
    with open("grid_output.nc", "wb") as f:
        f.write(response.content)
    print("Grid query successful!")
else:
    print(f"Error: {response.text}")
```

### Using the Provided Client

```python
from grid_client_example import query_grid, query_multiple_dates

# Single date query - full grid
result = query_grid("2023-01-15")

# Single date query - with BBOX
gulf_bbox = [-95.0, 20.0, -85.0, 28.0]
result_bbox = query_grid("2023-01-15", bbox=gulf_bbox)

# Multiple dates with BBOX
dates = ["2023-01-15", "2023-01-16", "2023-01-17"]
summary = query_multiple_dates(dates, bbox=gulf_bbox)
```

## Error Handling

The endpoint returns appropriate HTTP status codes:

- `200`: Success
- `400`: Bad request (invalid date format or BBOX)
- `413`: Request too large (grid exceeds MAX_PROFILES limit)
- `503`: Service unavailable (satellite data or grid data unavailable)
- `500`: Internal server error

### BBOX Validation Errors

Common BBOX validation errors:
- `BBOX must have exactly 4 values`: Incorrect number of coordinates
- `Longitude values must be between -180 and 180`: Invalid longitude range
- `Latitude values must be between -90 and 90`: Invalid latitude range
- `lon_min must be less than lon_max`: Invalid longitude order
- `lat_min must be less than lat_max`: Invalid latitude order

## Configuration

The endpoint respects the `MAX_PROFILES` configuration setting. If the grid (or filtered grid with BBOX) has more points than this limit, the request will be rejected with a 413 status.

To increase the limit, set the environment variable:
```bash
export NESPRESO_MAX_PROFILES=10000
```

## Performance Considerations

- Grid queries can be computationally intensive due to the large number of points
- BBOX filtering reduces the number of points processed, improving performance
- Processing time scales with the number of grid points
- The endpoint includes a 10-minute timeout for client requests
- Consider using BBOX filtering for large grids to improve performance

## Testing

Use the provided test script to verify the endpoint:

```bash
python test_grid_endpoint.py
```

This will test:
- Grid data loading
- Basic grid endpoint functionality
- BBOX filtering functionality
- BBOX validation with invalid inputs

## File Structure

```
nespreso_api/
├── services/api/app.py          # Main API with grid endpoint and BBOX support
├── nespreso_grid_and_mask.pkl  # Grid coordinates and mask
├── test_grid_endpoint.py       # Test script with BBOX testing
├── grid_client_example.py      # Client usage examples with BBOX
└── GRID_ENDPOINT_README.md     # This documentation
```

## Differences from Profile Endpoint

| Feature | Profile Endpoint | Grid Endpoint |
|---------|------------------|---------------|
| Input | Custom lat/lon/date lists | Predefined grid + single date + optional BBOX |
| Use Case | Specific locations | Complete spatial coverage (with optional filtering) |
| Request Size | Variable | Fixed (all grid points or BBOX subset) |
| Output | Profiles for requested points | Profiles for all grid points (or BBOX subset) |
| Dataset Structure | profile_number dimension | **depth, lat, lon dimensions + time coordinate** |
| Grid Format | Unstructured profiles | **Regular 2D/3D grid with proper coordinates** |

## Visualization Benefits

The new dataset structure with **depth, lat, lon as dimensions** provides several benefits:

1. **Proper NetCDF structure**: Standard gridded format that visualization tools expect
2. **Easy 2D plotting**: Direct access to lat/lon grids for surface maps
3. **3D visualization**: Depth dimension allows for vertical profile plots and cross-sections
4. **GIS integration**: Perfect compatibility with QGIS, ArcGIS, and other GIS software
5. **Statistical analysis**: Easy to perform spatial and temporal statistics on gridded data
6. **Interpolation**: Gridded data can be easily interpolated to other resolutions
7. **Standard tools**: Works seamlessly with matplotlib, cartopy, xarray plotting functions
8. **Data subsetting**: Easy to extract specific regions using xarray's `.sel()` method

## Troubleshooting

### Common Issues

1. **Grid file not found**: Ensure `nespreso_grid_and_mask.pkl` exists in the correct location
2. **BBOX validation errors**: Check BBOX format and coordinate ranges
3. **Timeout errors**: Grid queries may take several minutes for large grids
4. **Memory issues**: Large grids may require increased server memory limits
5. **Satellite data unavailable**: Check satellite data sources and availability

### Debug Information

The endpoint logs detailed information about:
- Grid loading and BBOX filtering
- Satellite data processing
- Model inference progress
- Output generation

Check the server logs for debugging information.
