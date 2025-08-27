# Batch Failure and NetCDF Merging Fixes

## Issues Identified

### 1. First Batch Failure (Worker Timeout)
**Problem**: The first batch was failing with a worker timeout after 16 minutes when processing AVISO data.

**Root Cause**: 
- AVISO data loading was hanging indefinitely when accessing `sub.adt.values`
- No timeout protection for satellite data loading operations
- Gunicorn worker timeout was too short (default 30 seconds)

**Error Traceback**:
```
File "/unity/g2/jmiranda/nespreso_api/services/accessor/sat.py", line 71, in get_aviso_by_date
  adt = np.asarray(sub.adt.values, dtype=np.float32)  # 2D (lat, lon)
```

### 2. NetCDF Merging Issue
**Problem**: The client expected a single output file but received multiple batch files, causing confusion.

**Root Cause**:
- Batch processing created multiple `.nc` files
- No automatic merging mechanism
- Client couldn't handle multiple file outputs

## Fixes Implemented

### 1. Timeout Protection for Satellite Data Loading

#### A. Added Timeout Decorator
```python
@timeout_decorator(300)  # 5 minutes timeout
def get_aviso_by_date(aviso_folder: str, c_date: datetime, bbox=None):
```

#### B. Enhanced Error Handling
- Added specific handling for `TimeoutError` exceptions
- Fallback mechanisms for data loading failures
- Better error messages with date and file information

#### C. Improved AVISO Loading
- Added try-catch blocks around file operations
- Alternative data access methods (`to_numpy()` fallback)
- Better error reporting for file loading issues

### 2. Gunicorn Configuration Updates

#### A. Increased Timeouts
- **Worker timeout**: 1800 seconds (30 minutes) instead of default 30 seconds
- **Keep-alive**: 2 seconds for better connection handling
- **Graceful timeout**: 30 seconds for clean worker shutdown

#### B. Performance Optimizations
- **Shared memory**: Use `/dev/shm` for worker temporary files
- **Max requests**: Restart workers after 1000 requests to prevent memory leaks
- **Preload**: Disabled to reduce memory usage

### 3. NetCDF File Merging

#### A. Automatic Merging Function
```python
def merge_netcdf_files(file_list, output_filename):
    """Merge multiple NetCDF files into a single file."""
```

#### B. Batch Processing Enhancement
- **Auto-merge**: Automatically merges batch files when `merge_output=True`
- **Smart filename handling**: Determines final output filename automatically
- **Error handling**: Graceful fallback if merging fails

#### C. Client Output Handling
- **Unified interface**: Returns single merged file or handles multiple files gracefully
- **File validation**: Checks file existence before attempting to read
- **Clear messaging**: Informs user about merge status

## How to Use the Fixes

### 1. Start Server with Proper Configuration

#### Option A: Use the startup script
```bash
cd nespreso_api
./start_server.sh
```

#### Option B: Manual gunicorn with config
```bash
cd nespreso_api
gunicorn -c gunicorn.conf.py 'wsgi:app'
```

#### Option C: Command line with timeout
```bash
cd nespreso_api
gunicorn -w 2 -b 0.0.0.0:5000 --timeout=1800 'wsgi:app'
```

### 2. Client Usage

#### Automatic Merging (Default)
```python
# This will automatically merge all batch files into a single output
result = get_predictions_batch(latitudes, longitudes, dates, 
                              filename_prefix="output",
                              merge_output=True)
```

#### Manual Control
```python
# Get batch files without merging
result = get_predictions_batch(latitudes, longitudes, dates, 
                              filename_prefix="output",
                              merge_output=False)

# Manual merge if needed
if isinstance(result, list) and len(result) > 1:
    merged_file = merge_netcdf_files(result, "final_output.nc")
```

## Expected Behavior After Fixes

### 1. No More Worker Timeouts
- **First batch**: Should complete successfully within 5 minutes
- **Large datasets**: Can process up to 30 minutes per request
- **Better error handling**: Clear messages if data loading fails

### 2. Automatic File Management
- **Batch processing**: Creates individual batch files during processing
- **Auto-merge**: Automatically combines all batches into single output
- **Clean output**: Single `.nc` file with all 345 profiles

### 3. Improved Debugging
- **Timeout warnings**: Clear indication when operations take too long
- **File information**: Shows which files are being loaded
- **Error details**: Specific error messages for debugging

## Monitoring and Troubleshooting

### 1. Check Server Logs
```bash
tail -f nespreso_api/wsgi.log
```

### 2. Monitor Timeouts
Look for these messages in the logs:
- `DEBUG[aviso]: Successfully loaded ADT data, shape: ...`
- `DEBUG[sat]: SSS bounds check - X/Y coordinates within data bounds`
- `WARNING - All interpolated values are NaN for date YYYY-MM-DD`

### 3. Verify File Creation
```bash
ls -la nespreso_api/uses/Idalia_profiles_Aug2Sep2023*
```

## Configuration Options

### 1. Timeout Settings
- **Satellite data timeout**: 5 minutes per file (configurable in decorator)
- **Worker timeout**: 30 minutes (configurable in gunicorn.conf.py)
- **Request timeout**: 30 minutes (configurable in gunicorn.conf.py)

### 2. Batch Processing
- **Batch size**: 50 points per batch (configurable)
- **Auto-merge**: Enabled by default (configurable)
- **File cleanup**: Batch files retained for debugging

## Next Steps

1. **Restart the server** using the new configuration
2. **Test the first batch** to ensure it no longer times out
3. **Run the full dataset** to verify automatic merging works
4. **Monitor the logs** for any remaining issues
5. **Verify output** - should get a single merged NetCDF file

## Files Modified

- `services/accessor/sat.py` - Added timeout protection and better error handling
- `nespreso_client.py` - Added NetCDF merging and improved batch processing
- `wsgi.py` - Updated configuration
- `gunicorn.conf.py` - New configuration file with proper timeouts
- `start_server.sh` - New startup script with proper configuration
