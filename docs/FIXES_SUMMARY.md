# NeSPReSO API Fixes Summary

## Issues Identified and Fixed

### 1. NetCDF Encoding Issue
**Problem**: The scipy backend for xarray doesn't support `zlib` and `complevel` compression parameters, causing the fallback to fail.

**Fix**: Modified `_write_netcdf_bytes()` in `app.py` to remove encoding parameters when falling back to scipy backend.

**Location**: `nespreso_api/services/api/app.py` lines 75-76

### 2. Satellite Data Loading Errors
**Problem**: "No SSS data for subset (4)" error and poor error handling when satellite data loading fails.

**Fixes Applied**:
- Added detailed debug logging in `load_satellite_data()` function
- Improved error handling with specific error messages
- Added validation to ensure at least some valid data is available
- Better handling of missing satellite data files
- **NEW**: Added climatological fallback values when satellite data is completely missing
- **NEW**: Improved file pattern matching for SST files (handles both "_subset" and non-"_subset" files)
- **NEW**: Enhanced SSS file discovery with wildcard fallback patterns

**Location**: `nespreso_api/services/accessor/sat.py` throughout the file

### 3. Large Dataset Memory Issues
**Problem**: Processing 345 points at once causes memory issues and timeouts.

**Fixes Applied**:
- Added batch processing capability in `nespreso_client.py`
- Automatic detection of large datasets (>100 points)
- Configurable batch sizes (default: 50 points per batch)
- Better error reporting for large requests

**Location**: `nespreso_api/nespreso_client.py` - new `get_predictions_batch()` function

### 4. MATLAB Date Format Handling
**Problem**: MATLAB dates were not being converted properly to ISO format strings.

**Fix**: Added proper MATLAB datenum to Python datetime conversion in the client.

**Location**: `nespreso_api/nespreso_client.py` lines 75-80

### 5. Input Validation and Error Reporting
**Problem**: Poor error messages when validation fails.

**Fixes Applied**:
- Better validation of satellite data dimensions
- More informative error messages
- Coordinate validation (lat/lon ranges)
- Profile count validation

**Location**: `nespreso_api/services/api/app.py` and `nespreso_client.py`

### 6. Satellite Data File Discovery
**Problem**: SST and SSS file patterns were too rigid and didn't handle variations in filenames.

**Fixes Applied**:
- **SST**: Added wildcard pattern matching with preference for "_subset" files
- **SSS**: Added wildcard fallback patterns for different version numbers
- **File existence checks**: Improved pattern matching for all satellite data types
- **Debug logging**: Added detailed logging for file discovery process

**Location**: `nespreso_api/services/accessor/sat.py` - `get_sst_ghrsst_by_date()` and `get_sss_by_date()` functions

### 7. Climatological Fallbacks
**Problem**: When satellite data is completely missing, the system would fail completely.

**Fix**: Added climatological fallback values for the Gulf of Mexico region:
- SSS: 36.0 (typical Gulf salinity)
- SST: 25.0°C (typical Gulf temperature)
- SSH: 0.0 (anomaly centered around 0)

This allows the model to run even when satellite data is unavailable, using reasonable default values.

**Location**: `nespreso_api/services/accessor/sat.py` - both one-to-one and cross-product modes

## How to Use the Fixes

### For Small Datasets (<100 points)
Use the original `get_predictions()` function:

```python
result = get_predictions(latitudes, longitudes, dates, filename="output.nc")
```

### For Large Datasets (≥100 points)
The client automatically detects large datasets and uses batch processing:

```python
# This will automatically use batch processing for 345 points
result = get_predictions_batch(latitudes, longitudes, dates, 
                              batch_size=50, 
                              filename_prefix="output")
```

### Manual Batch Processing
You can also manually control batch processing:

```python
result = get_predictions_batch(latitudes, longitudes, dates, 
                              batch_size=25,  # Smaller batches
                              filename_prefix="my_data")
```

## Testing the Fixes

### 1. Test Script
Run the test script to verify the fixes work:

```bash
cd nespreso_api
python test_fixes.py
```

### 2. Satellite Data Check
Check if your satellite data files are accessible:

```bash
cd nespreso_api
python check_satellite_data.py
```

This will help identify if the issue is with file paths, missing files, or data loading.

## Expected Behavior After Fixes

1. **Better Error Messages**: You'll see specific error messages instead of generic 500 errors
2. **Automatic Batch Processing**: Large datasets will be processed in smaller chunks
3. **Improved Debugging**: More detailed logging to help diagnose issues
4. **Graceful Degradation**: The system will handle missing satellite data more gracefully
5. **Proper Date Handling**: MATLAB dates will be converted correctly
6. **Robust File Discovery**: Better handling of SST/SSS file naming variations
7. **Climatological Fallbacks**: Model can run even with missing satellite data
8. **Detailed Logging**: Comprehensive debug information for troubleshooting

## Configuration

The system respects the `CFG.MAX_PROFILES` setting. If you need to process more than this limit, use batch processing.

## Monitoring and Debugging

Check the API logs for detailed debug information:
- Satellite data loading status
- File discovery and pattern matching
- Interpolation success/failure
- Memory usage and processing times
- Batch processing progress
- Climatological fallback usage

## Troubleshooting

### If you still get NaN values:
1. Run `check_satellite_data.py` to verify file availability
2. Check the debug logs for file discovery issues
3. Verify the satellite data paths in your configuration
4. Look for climatological fallback warnings in the logs

### If SST files aren't found:
1. Check if files have "_subset" suffix or not
2. Verify the file naming pattern matches your data
3. Check file permissions and accessibility

### If SSS files aren't found:
1. Check for different version numbers (v06.0, v05.0, etc.)
2. Verify the day-of-year calculation
3. Check if files are in the expected directory structure

## Next Steps

1. **Test with the small dataset** first using `test_fixes.py`
2. **Check satellite data availability** using `check_satellite_data.py`
3. **Run your full dataset** - it will automatically use batch processing
4. **Monitor the logs** for detailed progress information
5. **Adjust batch sizes** if needed based on your system's memory constraints
