# NeSPReSO API and Client Examples

## Overview

NeSPReSO is a scientific service for generating **synthetic temperature and salinity profiles** in the ocean, given latitude, longitude, and date. It is designed for oceanographers, data scientists, and operational users who need fast, reliable, and physically consistent profile predictions.

This repository provides:
- A **modular Flask-based API** with two main endpoints:
  - `/v1/profile` - For individual point predictions
  - `/v1/profile/grid` - For complete grid coverage with optional filtering
- **Python clients** for easy integration
- **Comprehensive examples** and documentation
- OpenAPI documentation, CI/CD, and Prometheus observability

---

## Quick Start

### 1. Install Dependencies

```bash
# Using conda
conda create -n nespreso_api -c conda-forge --file requirements.txt python=3.10
conda activate nespreso_api

# Using pip
pip install -r requirements.txt

# For client usage only (lighter installation)
pip install -r requirements_clients.txt
```

### 2. Start the API Server

```bash
gunicorn -w 2 -c config/gunicorn.conf.py 'wsgi:app'
```

The API will be available at:
- **Profile endpoint**: `http://localhost:5000/v1/profile`
- **Grid endpoint**: `http://localhost:5000/v1/profile/grid`
- **Metrics**: `http://localhost:5000/metrics`

### 3. Use the Python Clients

```python
# For individual point predictions
from profile_client import get_predictions, get_predictions_batch

# Single prediction
result = get_predictions([25.0], [-83.0], ["2016-12-31"])

# Batch processing for large datasets
result = get_predictions_batch(latitudes, longitudes, dates, batch_size=100)

# For grid queries
from grid_client import query_grid, query_multiple_dates

# Full grid query
result = query_grid("2016-12-31")

# Regional query with BBOX
gulf_bbox = [-95.0, 18.0, -80.0, 31.0]
result = query_grid("2016-12-31", bbox=gulf_bbox)
```

---

## API Endpoints

### 1. Profile Endpoint (`/v1/profile`)

**Purpose**: Generate predictions for specific latitude/longitude coordinates and dates.

**Method**: POST  
**Request Format**:
```json
{
  "lat": [25.0, 26.0, 27.0],
  "lon": [-83.0, -84.0, -85.0],
  "date": ["2016-12-31", "2016-12-30", "2016-12-29"]
}
```

**Response**: NetCDF file containing temperature and salinity profiles

**Features**:
- Supports single points or arrays of coordinates
- Automatic input preprocessing
- Batch processing for large datasets
- File merging capabilities

### 2. Grid Endpoint (`/v1/profile/grid`)

**Purpose**: Query all predefined grid points for complete spatial coverage.

**Method**: POST  
**Request Format**:
```json
{
  "date": "2016-12-31",
  "bbox": [-95.0, 18.0, -80.0, 31.0]  // Optional: [lon_min, lat_min, lon_max, lat_max]
}
```

**Response**: NetCDF file with gridded data (depth × lat × lon dimensions)

**Features**:
- Complete Gulf of Mexico coverage (41×59 grid, ~1,018 points)
- Optional BBOX filtering for regional analysis
- Proper NetCDF structure for visualization
- 0.25° resolution

---

## Python Clients

### 1. Main Client (`profile_client.py`)

The main client provides functions for individual point predictions:

#### `get_predictions(lat, lon, date, filename="output.nc", api_url=None)`
- **Purpose**: Get predictions for specific coordinates
- **Input**: Latitude, longitude, and date arrays
- **Output**: NetCDF filename or None on failure
- **Features**: Automatic async/sync handling, input preprocessing

#### `get_predictions_batch(lat, lon, date, batch_size=1000, filename_prefix="output", api_url=None, merge_output=True)`
- **Purpose**: Process large datasets in batches
- **Input**: Large coordinate arrays
- **Output**: Single merged file or list of batch files
- **Features**: Automatic batching, progress tracking, file merging

#### `merge_netcdf_files(file_list, output_filename)`
- **Purpose**: Merge multiple NetCDF batch files
- **Input**: List of NetCDF filenames
- **Output**: Single merged NetCDF file
- **Features**: Profile concatenation, coordinate reindexing

### 2. Grid Client (`grid_client.py`)

The grid client provides functions for complete grid coverage:

#### `query_grid(date_str, bbox=None, api_url=DEFAULT_API_URL)`
- **Purpose**: Query grid for a specific date
- **Input**: Date string and optional BBOX
- **Output**: Dictionary with success status and file details
- **Features**: Automatic output directory creation, error handling

#### `query_multiple_dates(date_list, bbox=None, api_url=DEFAULT_API_URL)`
- **Purpose**: Process multiple dates efficiently
- **Input**: List of date strings and optional BBOX
- **Output**: Summary of results with success/failure counts
- **Features**: Progress tracking, comprehensive reporting

#### `generate_date_range(start_date, end_date)`
- **Purpose**: Generate date sequences
- **Input**: Start and end dates in YYYY-MM-DD format
- **Output**: List of date strings
- **Features**: Automatic date incrementing

#### `get_common_bbox_regions()`
- **Purpose**: Get predefined BBOX regions for Gulf of Mexico
- **Output**: Dictionary of named regions
- **Regions**: full_gulf, western_gulf, eastern_gulf, northern_gulf, southern_gulf, florida_straits, yucatan_channel

---

## Usage Examples

### Basic Profile Prediction

```python
from profile_client import get_predictions

# Single point
result = get_predictions([25.0], [-83.0], ["2016-12-31"], filename="single_point.nc")

# Multiple points
latitudes = [25.0, 26.0, 27.0]
longitudes = [-83.0, -84.0, -85.0]
dates = ["2016-12-31", "2016-12-30", "2016-12-29"]

result = get_predictions(latitudes, longitudes, dates, filename="multiple_points.nc")
```

### Large Dataset Processing

```python
from profile_client import get_predictions_batch

# Process large dataset with automatic batching
result = get_predictions_batch(
    latitudes, longitudes, dates,
    batch_size=100,  # Process 100 points at a time
    filename_prefix="large_dataset",
    merge_output=True  # Automatically merge batch files
)

if isinstance(result, str):
    print(f"Single merged file: {result}")
else:
    print(f"Multiple batch files: {result}")
```

### Grid Queries

```python
from grid_client import query_grid, get_common_bbox_regions

# Get common BBOX regions
bbox_regions = get_common_bbox_regions()

# Full grid query
result = query_grid("2016-12-31")

# Regional query
gulf_bbox = bbox_regions["western_gulf"]
result = query_grid("2016-12-31", bbox=gulf_bbox)

# Check results
if result["success"]:
    print(f"File saved: {result['filename']}")
    print(f"File size: {result['size_bytes']} bytes")
else:
    print(f"Error: {result['error']}")
```

### Multiple Date Processing

```python
from grid_client import query_multiple_dates, generate_date_range

# Generate date range
dates = generate_date_range("2016-12-01", "2016-12-31")

# Process all dates with BBOX filtering
gulf_bbox = [-95.0, 18.0, -80.0, 31.0]
summary = query_multiple_dates(dates, bbox=gulf_bbox)

print(f"Processed {summary['total']} dates")
print(f"Successful: {summary['successful']}")
print(f"Failed: {summary['failed']}")
```

---

## Output Files

### Profile Endpoint Output
- **Format**: NetCDF with `profile_number` dimension
- **Variables**: Temperature, Salinity, SSS, SST, AVISO, depth, lat, lon, time
- **Structure**: Unstructured profiles for requested coordinates

### Grid Endpoint Output
- **Format**: NetCDF with `depth`, `lat`, `lon` dimensions
- **Variables**: Temperature(depth, lat, lon), Salinity(depth, lat, lon), SSS(lat, lon), SST(lat, lon), AVISO(lat, lon)
- **Structure**: Regular 3D grid with proper coordinates
- **Benefits**: Easy visualization, GIS integration, statistical analysis

---

## Configuration

### Environment Variables
```bash
# Increase maximum profiles limit
export NESPRESO_MAX_PROFILES=10000

# Set custom API URL
export NESPRESO_API_URL=http://your-server:5000/v1/profile
```

### Client Configuration
```python
# Custom API endpoint
from profile_client import get_predictions
result = get_predictions(lat, lon, date, api_url="http://custom-server:5000/v1/profile")

# Custom timeout
from grid_client_example import query_grid
result = query_grid("2016-12-31", api_url="http://custom-server:5000/v1/profile/grid")
```

---

## Error Handling

### Common Issues and Solutions

1. **Connection Errors**
   - Check if the API server is running
   - Verify the API URL and port
   - Check network connectivity

2. **Timeout Errors**
   - Large requests may take several minutes
   - Increase timeout settings if needed
   - Use BBOX filtering to reduce request size

3. **Input Validation Errors**
   - Ensure coordinates are within valid ranges (lat: -90 to 90, lon: -180 to 180)
   - Check date format (YYYY-MM-DD)
   - Verify all input arrays have the same length

4. **File Writing Errors**
   - Check write permissions in output directory
   - Ensure sufficient disk space
   - Verify output directory exists

### Error Response Format
```python
{
    "success": False,
    "status_code": 400,
    "error": "Invalid input parameters"
}
```

---

## Performance Considerations

### Batch Processing
- **Small datasets** (< 100 points): Use `get_predictions()`
- **Large datasets** (> 100 points): Use `get_predictions_batch()`
- **Optimal batch size**: 100-1000 points (depends on server capacity)

### Grid Queries
- **Full grid**: ~1,018 points, may take 5-15 minutes
- **BBOX filtering**: Reduces processing time significantly
- **Multiple dates**: Process sequentially to avoid overwhelming the server

### Memory Management
- Large NetCDF files can be memory-intensive
- Use BBOX filtering for regional analysis
- Consider processing dates separately for very large time ranges

---

## Testing

### Run the Examples
```bash
# Test profile client
python profile_client.py

# Test grid client
python grid_client.py
```

### Run API Tests
```bash
PYTHONPATH=nespreso_api:nespreso_api/eoas-pyutils pytest
```

---

## Troubleshooting

### PYTHONPATH Issues
Always run with the correct PYTHONPATH:
```bash
PYTHONPATH=nespreso_api:nespreso_api/eoas-pyutils python your_script.py
```

### Import Errors
Ensure all dependencies are installed:
```bash
pip install httpx xarray numpy requests
```

### API Errors
- Check server logs for detailed error information
- Verify satellite data availability for requested dates
- Check if the request exceeds MAX_PROFILES limit

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

### Development Guidelines
- All new endpoints must have OpenAPI documentation
- Include property-based tests for output validation
- Follow the existing code style and documentation format

---

## License

This project is licensed under the MIT License.

---

## Support

For questions and support:
- Check the documentation in the `docs/` directory
- Review the example files for usage patterns
- Check server logs for detailed error information
- Open an issue on the GitHub repository
