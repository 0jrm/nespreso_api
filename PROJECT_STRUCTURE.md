# NeSPReSO API Project Structure

## Overview

The NeSPReSO API project has been reorganized to improve clarity and maintainability. This document outlines the new folder structure and what each directory contains.

## Root Directory Structure

```
nespreso_api/
├── 📁 config/                    # Configuration files and scripts
├── 📁 data/                      # Data files and resources
├── 📁 docs/                      # Documentation and guides
├── 📁 logs/                      # Log files
├── 📁 notebooks/                 # Jupyter notebooks
├── 📁 scripts/                   # Utility scripts and tools
├── 📁 services/                  # Core API services
├── 📁 tests/                     # Test files
├── 📁 uses/                      # User examples and outputs
├── 📄 profile_client.py          # Profile prediction client
├── 📄 grid_client.py             # Grid coverage client
├── 📄 requirements.txt           # Main project dependencies
├── 📄 requirements_clients.txt   # Client-only dependencies
├── 📄 wsgi.py                    # WSGI entry point
└── 📄 pyproject.toml            # Project configuration
```

## Directory Details

### 📁 `config/` - Configuration Files
Contains all configuration files, server scripts, and deployment configurations:
- `gunicorn_config.py` - Gunicorn configuration
- `gunicorn.conf.py` - Alternative Gunicorn config
- `ozavala_custom_wsgi.conf` - Custom WSGI configuration
- `start_server.sh` - Server startup script
- `restart_server.sh` - Server restart script

### 📁 `data/` - Data Files
Contains data files and resources used by the API:
- `nespreso_grid_and_mask.pkl` - Grid coordinates and mask data
- `satellite_dates_report.json` - Satellite data availability report

### 📁 `docs/` - Documentation
Comprehensive documentation for the project:
- `README.md` - Main project documentation
- `USAGE_GUIDE.md` - Quick usage reference
- `NAMING_CHANGES.md` - File renaming guide
- `GRID_ENDPOINT_README.md` - Grid endpoint documentation
- `ARCHIVE_BUILDER_README.md` - Archive builder documentation
- `BATCH_FAILURE_FIXES.md` - Batch processing fixes
- `FIXES_SUMMARY.md` - Summary of fixes
- `PICKLE_COMPATIBILITY_FIX.md` - Pickle compatibility notes
- `UNLIMITED_CHANGES.md` - Unlimited changes documentation
- `environment.txt` - Environment setup guide
- `old_readme.md` - Previous readme (archived)
- `api.yaml` - OpenAPI specification
- `structure.md` - Project structure documentation

### 📁 `logs/` - Log Files
Contains log files from various processes:
- `archive_builder.log` - Archive builder logs
- `nespreso_archive_builder.log` - NeSPReSO archive builder logs

### 📁 `notebooks/` - Jupyter Notebooks
Contains Jupyter notebooks for analysis and development:
- `gird_coordinates.ipynb` - Grid coordinates analysis

### 📁 `scripts/` - Utility Scripts
Contains utility scripts and tools:
- `bbox_examples.py` - Bounding box examples
- `build_nespreso_archive.py` - Archive building script
- `check_satellite_data.py` - Satellite data validation
- `debug_nan_issues.py` - NaN debugging tools
- `scan_satellite_dates.py` - Satellite date scanning

### 📁 `services/` - Core API Services
Contains the main API service modules:
- `api/` - API endpoints and handlers
- `kernel/` - Core model handling
- `accessor/` - Data access layer
- `utils/` - Utility functions
- `config.py` - Service configuration

### 📁 `tests/` - Test Files
Contains test files and test configuration:
- Test modules for various components
- Test configuration files

### 📁 `uses/` - User Examples
Contains user examples and output files:
- Example outputs and user-generated content

## Client Files

### 📄 `profile_client.py`
Main client for individual point predictions and batch processing:
- `get_predictions()` - Single prediction function
- `get_predictions_batch()` - Batch processing function
- `merge_netcdf_files()` - File merging utility

### 📄 `grid_client.py`
Client for grid coverage queries with BBOX filtering:
- `query_grid()` - Grid query function
- `query_multiple_dates()` - Multiple date processing
- `generate_date_range()` - Date range generation
- `get_common_bbox_regions()` - Predefined BBOX regions

## Benefits of New Structure

1. **Clear Organization**: Each directory has a specific purpose
2. **Easy Navigation**: Users can quickly find what they need
3. **Maintainability**: Related files are grouped together
4. **Scalability**: Easy to add new files in appropriate locations
5. **Professional Appearance**: Clean, organized project structure

## File Locations

### For Users
- **Documentation**: `docs/README.md` and `docs/USAGE_GUIDE.md`
- **Client Examples**: `profile_client.py` and `grid_client.py`
- **Dependencies**: `requirements_clients.txt`

### For Developers
- **API Code**: `services/` directory
- **Configuration**: `config/` directory
- **Scripts**: `scripts/` directory
- **Tests**: `tests/` directory

### For Data Scientists
- **Notebooks**: `notebooks/` directory
- **Data**: `data/` directory
- **Examples**: `uses/` directory

## Migration Notes

- All functionality remains the same
- Only file locations have changed
- Import statements in Python code remain unchanged
- Documentation has been updated to reflect new structure
