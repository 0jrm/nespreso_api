# NeSPReSO Data Archive Builder

This directory contains scripts to build a comprehensive NeSPReSO data archive by finding all dates with complete satellite data and processing them through the grid endpoint.

## 🎯 Overview

The archive builder consists of two main scripts:

1. **`scan_satellite_dates.py`** - Scans satellite directories to find available dates
2. **`build_nespreso_archive.py`** - Builds the complete NeSPReSO archive

## 📊 Satellite Data Requirements

The NeSPReSO model requires data from three satellite sources:
- **SST (Sea Surface Temperature)**: OISST data from `/Net/work/ozavala/DATA/GOFFISH/SST/OISST/`
- **SSS (Sea Surface Salinity)**: SMAP data from `/Net/work/ozavala/DATA/GOFFISH/SSS/SMAP_Global/`
- **AVISO (Sea Surface Height)**: SSH data from `/unity/f1/ozavala/DATA/GOFFISH/AVISO/GoM/`

A date is considered "complete" only when **ALL THREE** sources have data available.

## 🚀 Quick Start

### Step 1: Scan for Available Dates

First, scan the satellite directories to see what data is available:

```bash
# Basic scan
python scan_satellite_dates.py

# Detailed scan with verbose output
python scan_satellite_dates.py --detailed

# Save report to custom file
python scan_satellite_dates.py --output my_report.json
```

This will:
- Scan all satellite data directories
- Identify dates with complete data
- Generate a comprehensive report
- Save results to `satellite_dates_report.json`

### Step 2: Build the Archive

Once you know what dates are available, build the NeSPReSO archive:

```bash
# Start fresh archive building
python build_nespreso_archive.py

# Resume from checkpoint (if interrupted)
python build_nespreso_archive.py
```

## 📋 Script Details

### `scan_satellite_dates.py`

**Purpose**: Scans satellite directories without requiring the API to be running.

**Features**:
- Fast directory scanning
- Date extraction from filenames
- Complete data identification
- Comprehensive reporting
- JSON output for further processing

**Usage**:
```bash
python scan_satellite_dates.py [--output OUTPUT_FILE] [--detailed]
```

**Output**: `satellite_dates_report.json` with:
- Available dates by source and year
- Complete dates (all three sources)
- Statistics and summaries
- Date ranges for each source

### `build_nespreso_archive.py`

**Purpose**: Builds the complete NeSPReSO archive by processing each valid date.

**Features**:
- Automatic satellite data validation
- NeSPReSO API integration
- Progress tracking with checkpoints
- Pause/resume functionality
- Error handling and retries
- Multi-threaded processing
- Comprehensive logging

**Requirements**:
- NeSPReSO API must be running on `http://localhost:5000`
- Satellite data directories must be accessible
- Sufficient disk space for output files

## ⏸️ Pause and Resume

The archive builder automatically saves progress to `archive_checkpoint.json`. You can:

**Pause**: Press `Ctrl+C` - the script will save progress and exit gracefully
**Resume**: Run the script again - it will detect the checkpoint and offer to resume

**Checkpoint Data**:
- Date processing status
- Error messages and retry counts
- Output file locations
- Processing timestamps

## 📁 Output Structure

```
nespreso_archive/
├── nespreso_grid_2023-06-15.nc
├── nespreso_grid_2023-06-16.nc
├── nespreso_grid_2023-06-17.nc
├── ...
├── archive_summary.json
└── archive_checkpoint.json
```

**Files**:
- `nespreso_grid_YYYY-MM-DD.nc`: NeSPReSO grid data for each date
- `archive_summary.json`: Processing summary and statistics
- `archive_checkpoint.json`: Progress tracking for resume functionality

## 🔧 Configuration

Edit the `ArchiveConfig` class in `build_nespreso_archive.py` to customize:

```python
@dataclass
class ArchiveConfig:
    sst_root: str = "/Net/work/ozavala/DATA/GOFFISH/SST/OISST/"
    sss_root: str = "/Net/work/ozavala/DATA/GOFFISH/SSS/SMAP/"
    aviso_root: str = "/Net/work/ozavala/DATA/GOFFISH/AVISO/"
    api_url: str = "http://localhost:5000/v1/profile/grid"
    output_dir: str = "./nespreso_archive"
    checkpoint_file: str = "./archive_checkpoint.json"
    max_workers: int = 4          # Number of parallel workers
    retry_attempts: int = 3       # Retry failed requests
    retry_delay: int = 5          # Delay between retries (seconds)
```

## 📊 Monitoring and Logging

**Log File**: `nespreso_archive_builder.log`
**Console Output**: Real-time progress updates
**Checkpoint**: Automatic saves every 10 processed dates

**Log Levels**:
- `INFO`: General progress and status
- `WARNING`: Non-critical issues
- `ERROR`: Processing failures and errors

## 🚨 Error Handling

The archive builder handles various error scenarios:

**API Errors**: Network issues, timeouts, server errors
**Data Errors**: Missing satellite files, incomplete data
**System Errors**: Disk space, permissions, interruptions

**Retry Logic**:
- Failed requests are retried up to 3 times
- Exponential backoff between retries
- Failed dates are logged for manual review

## 📈 Performance

**Processing Speed**: Depends on:
- Number of available dates
- API response time
- Network conditions
- System resources

**Typical Performance**:
- ~100-500 dates per hour (depending on complexity)
- Multi-threaded processing (4 workers by default)
- Automatic checkpointing every 10 dates

## 🔍 Troubleshooting

### Common Issues

**1. No Complete Dates Found**
```bash
# Check if satellite directories exist
ls /Net/work/ozavala/DATA/GOFFISH/SST/OISST/
ls /Net/work/ozavala/DATA/GOFFISH/SSS/SMAP/
ls /Net/work/ozavala/DATA/GOFFISH/AVISO/
```

**2. API Connection Failed**
```bash
# Check if NeSPReSO API is running
curl http://localhost:5000/health
```

**3. Permission Denied**
```bash
# Check directory permissions
ls -la /Net/work/ozavala/DATA/GOFFISH/
```

**4. Disk Space Issues**
```bash
# Check available disk space
df -h .
```

### Debug Mode

Enable detailed logging:
```python
logging.basicConfig(level=logging.DEBUG)
```

## 📚 Example Workflow

```bash
# 1. Scan for available dates
python scan_satellite_dates.py --detailed

# 2. Review the report
cat satellite_dates_report.json

# 3. Start archive building
python build_nespreso_archive.py

# 4. Monitor progress
tail -f nespreso_archive_builder.log

# 5. Check results
ls -la nespreso_archive/
cat nespreso_archive/archive_summary.json
```

## 🎯 Best Practices

1. **Start with scanning** to understand data availability
2. **Test with a few dates** before running the full archive
3. **Monitor disk space** - each NetCDF file is ~100-500MB
4. **Use checkpoints** for long-running processes
5. **Review error logs** for failed dates
6. **Backup checkpoint files** before major changes

## 📞 Support

For issues or questions:
1. Check the log files for error details
2. Verify satellite data availability
3. Ensure API is running and accessible
4. Review configuration settings

## 🔄 Future Enhancements

Planned improvements:
- Web-based progress monitoring
- Email notifications for completion
- Integration with data management systems
- Automated quality checks
- Compression and archiving options
