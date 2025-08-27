# File Naming Changes - NeSPReSO API

## Overview

The NeSPReSO API client files have been renamed to better reflect their functionality and purpose. This document outlines all the changes made and provides guidance for updating existing code.

## File Renaming Summary

| Old Name | New Name | Purpose |
|-----------|----------|---------|
| `nespreso_client.py` | `profile_client.py` | Individual point predictions and batch processing |
| `grid_client_example.py` | `grid_client.py` | Grid coverage queries with BBOX filtering |
| `client_requirements.txt` | `requirements_clients.txt` | Client-only dependencies |

## Changes Made

### 1. File Renaming
- ✅ `nespreso_client.py` → `profile_client.py`
- ✅ `grid_client_example.py` → `grid_client.py`
- ✅ `client_requirements.txt` → `requirements_clients.txt`

### 2. Documentation Updates
- ✅ `README.md` - Updated all import statements and file references
- ✅ `USAGE_GUIDE.md` - Updated all import statements and file references
- ✅ `profile_client.py` - Updated header comments and example imports
- ✅ `grid_client.py` - Updated header comments and example imports
- ✅ `requirements_clients.txt` - Updated installation instructions

### 3. Import Statement Updates
All documentation now uses the new import statements:

```python
# OLD (no longer works)
from nespreso_client import get_predictions, get_predictions_batch
from grid_client_example import query_grid, query_multiple_dates

# NEW (use these)
from profile_client import get_predictions, get_predictions_batch
from grid_client import query_grid, query_multiple_dates
```

## How to Update Your Code

### If you have existing scripts using the old names:

1. **Update import statements:**
   ```python
   # Change this:
   from nespreso_client import get_predictions
   
   # To this:
   from profile_client import get_predictions
   ```

2. **Update file execution commands:**
   ```bash
   # Change this:
   python nespreso_client.py
   
   # To this:
   python profile_client.py
   ```

3. **Update requirements installation:**
   ```bash
   # Change this:
   pip install -r client_requirements.txt
   
   # To this:
   pip install -r requirements_clients.txt
   ```

## New File Structure

```
nespreso_api/
├── profile_client.py           # Individual point predictions
├── grid_client.py              # Grid coverage queries
├── requirements_clients.txt     # Client dependencies
├── README.md                   # Comprehensive documentation
├── USAGE_GUIDE.md             # Quick reference guide
└── ... (other files)
```

## Functionality Remains the Same

**Important**: All functionality remains exactly the same. Only the file names and import statements have changed.

- `get_predictions()` function works identically
- `get_predictions_batch()` function works identically  
- `query_grid()` function works identically
- `query_multiple_dates()` function works identically
- All parameters, return values, and behavior unchanged

## Benefits of New Naming

1. **Clearer Purpose**: `profile_client.py` clearly indicates it's for profile predictions
2. **Simpler Names**: `grid_client.py` is cleaner than `grid_client_example.py`
3. **Better Organization**: `requirements_clients.txt` clearly distinguishes from main requirements
4. **Easier Discovery**: Users can immediately understand what each file does

## Testing the Changes

After updating your code, test that everything still works:

```bash
# Test profile client
python profile_client.py

# Test grid client
python grid_client.py
```

## Support

If you encounter any issues after updating to the new file names:

1. Check that all import statements use the new names
2. Verify the files exist in your directory
3. Check the updated documentation for correct usage
4. Open an issue if problems persist

## Migration Checklist

- [ ] Update `from nespreso_client import` to `from profile_client import`
- [ ] Update `from grid_client_example import` to `from grid_client import`
- [ ] Update any script execution commands
- [ ] Update requirements installation commands
- [ ] Test that all functionality works as expected
