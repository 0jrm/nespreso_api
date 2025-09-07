#!/usr/bin/env python3
"""
NeSPReSO Data Archive Builder

This script finds all dates with complete satellite data (SST, SSS, AVISO) and
builds a comprehensive NeSPReSO data archive. It can be paused and resumed easily.

Features:
- Uses the corrected scanner from scan_satellite_dates.py
- Identifies dates with complete data across all three sources
- Builds NeSPReSO profiles for each valid date
- Progress tracking with checkpoint files
- Easy pause/resume functionality
- Comprehensive logging
- Error handling and retry logic

Usage examples:
- Build the archive interactively (will prompt to resume if a checkpoint exists):
- Start fresh even if a checkpoint exists (answer "n" when prompted):
  python build_nespreso_archive.py

- Show progress without running any jobs:
  python build_nespreso_archive.py --status

Notes:
- AVISO availability is merged from two roots: `aviso_root` before 2024-11-01 and
  `new_aviso_root` on/after 2024-11-01.
- To adjust paths, API URL, or worker count, edit the `ArchiveConfig` defaults
  below or instantiate `ArchiveConfig` with custom arguments.
"""

import os
import json
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import requests
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import signal
import sys

# Import our corrected scanner functions
from scan_satellite_dates import (
    scan_all_satellite_directories,
    find_complete_dates
)

# Configure logging to output to both file and terminal
def setup_logging():
    """Setup logging to output to both file and terminal"""
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Create file handler
    file_handler = logging.FileHandler('nespreso_archive_builder.log')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Get logger and add handlers
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Add our handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Setup logging
logger = setup_logging()

@dataclass
class ArchiveConfig:
    """Configuration for the archive builder"""
    sst_root: str = "/Net/work/ozavala/DATA/GOFFISH/SST/OISST/"
    sss_root: str = "/Net/work/ozavala/DATA/GOFFISH/SSS/SMAP_Global/"
    aviso_root: str = "/Net/work/ozavala/DATA/GOFFISH/AVISO/GoM/"
    new_aviso_root: str = "/Net/work/ozavala/DATA/GOFFISH/AVISO/GoM/CMEMS_GLOBAL_PHY_ANFC/"
    api_url: str = "http://146.201.220.56:5000/v1/profile/grid"
    output_dir: str = "/Net/work/ozavala/DATA/SubSurfaceFields/NeSPReSO"
    checkpoint_file: str = "/Net/work/ozavala/DATA/SubSurfaceFields/NeSPReSO/archive_checkpoint.json"
    max_workers: int = 4
    retry_attempts: int = 3
    retry_delay: int = 5
    
    def __post_init__(self):
        # Create output directory if it doesn't exist
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

@dataclass
class DateStatus:
    """Status of a specific date in the archive building process"""
    date: str
    sst_available: bool = False
    
    sss_available: bool = False
    aviso_available: bool = False
    complete: bool = False
    processed: bool = False
    error: Optional[str] = None
    retry_count: int = 0
    last_attempt: Optional[str] = None
    output_file: Optional[str] = None

class ArchiveBuilder:
    """Main class for building the NeSPReSO data archive"""
    
    def __init__(self, config: ArchiveConfig):
        self.config = config
        self.checkpoint_data = self._load_checkpoint()
        self.running = True
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info(f"Archive Builder initialized with config: {asdict(config)}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
        self._save_checkpoint()
        sys.exit(0)
    
    def _load_checkpoint(self) -> Dict:
        """Load checkpoint data if it exists"""
        if os.path.exists(self.config.checkpoint_file):
            try:
                with open(self.config.checkpoint_file, 'r') as f:
                    data = json.load(f)
                    logger.info(f"Loaded checkpoint with {len(data)} dates")
                    return data
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")
                return {}
        return {}
    
    def _save_checkpoint(self):
        """Save current progress to checkpoint file"""
        try:
            with open(self.config.checkpoint_file, 'w') as f:
                json.dump(self.checkpoint_data, f, indent=2)
            logger.info("Checkpoint saved successfully")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def _get_complete_dates(self) -> List[str]:
        """Get dates with complete satellite data using the scanner"""
        logger.info("Getting complete dates using scanner...")
        
        # Use the scanner to find dates for SST/SSS and AVISO (old root)
        available_dates = scan_all_satellite_directories(
            self.config.sst_root,
            self.config.sss_root, 
            self.config.aviso_root
        )

        # Also scan AVISO using the new root, then merge starting 2024-11-01
        try:
            available_dates_new_aviso = scan_all_satellite_directories(
                self.config.sst_root,
                self.config.sss_root,
                self.config.new_aviso_root
            ).get('aviso', {})
        except Exception:
            available_dates_new_aviso = {}

        # Merge AVISO availability: use old root before 2024-11, new root from 2024-11 onward
        old_aviso_by_year = available_dates.get('aviso', {}) or {}
        merged_aviso: Dict[str, Set[str]] = {}
        years_union = set(old_aviso_by_year.keys()) | set(available_dates_new_aviso.keys())

        for year in years_union:
            try:
                year_int = int(year)
            except ValueError:
                continue

            old_set = old_aviso_by_year.get(year, set()) or set()
            new_set = available_dates_new_aviso.get(year, set()) or set()

            if year_int < 2024:
                merged_set = set(old_set)
            elif year_int > 2024:
                merged_set = set(new_set) if new_set else set(old_set)
            else:  # year == 2024
                # Keep months Jan-Oct from old, Nov-Dec from new
                merged_set = set()
                for adate in old_set:
                    if len(adate) == 8 and adate.endswith('01'):
                        try:
                            if int(adate[4:6]) < 11:
                                merged_set.add(adate)
                        except ValueError:
                            continue
                for adate in new_set:
                    if len(adate) == 8 and adate.endswith('01'):
                        try:
                            if int(adate[4:6]) >= 11:
                                merged_set.add(adate)
                        except ValueError:
                            continue

            if merged_set:
                merged_aviso[year] = merged_set

        available_dates['aviso'] = merged_aviso
        
        # Determine the earliest year where all three sources have data
        try:
            sst_years = sorted([int(y) for y in available_dates.get('sst', {}).keys()])
            sss_years = sorted([int(y) for y in available_dates.get('sss', {}).keys()])
            aviso_years = sorted([int(y) for y in available_dates.get('aviso', {}).keys()])
        except Exception:
            sst_years, sss_years, aviso_years = [], [], []

        min_sst_year = sst_years[0] if sst_years else None
        min_sss_year = sss_years[0] if sss_years else None
        min_aviso_year = aviso_years[0] if aviso_years else None

        candidate_years = [y for y in [min_sst_year, min_sss_year, min_aviso_year] if y is not None]
        start_year = max(candidate_years) if candidate_years else None

        if start_year is None:
            logger.error("Could not determine a common start year across SST/SSS/AVISO.")
            return []

        logger.info(
            f"Computed common start year across sources: SST={min_sst_year}, "
            f"SSS={min_sss_year}, AVISO={min_aviso_year} -> start_year={start_year}"
        )

        # Also compute strict daily intersections across sources
        # Build per-year intersections of SST dates with SSS DOY-converted dates and AVISO months
        intersected_dates: List[str] = []

        # Union of all candidate years from any source
        candidate_years_all_sources = set()
        candidate_years_all_sources.update(available_dates.get('sst', {}).keys())
        candidate_years_all_sources.update(available_dates.get('sss', {}).keys())
        candidate_years_all_sources.update(available_dates.get('aviso', {}).keys())

        for year in sorted(candidate_years_all_sources):
            try:
                year_int = int(year)
            except ValueError:
                logger.warning(f"Unexpected year key format in available_dates: {year}")
                continue

            if year_int < start_year:
                continue

            sst_dates: Set[str] = available_dates.get('sst', {}).get(year, set())
            sss_doys: Set[str] = available_dates.get('sss', {}).get(year, set())
            aviso_dates: Set[str] = available_dates.get('aviso', {}).get(year, set())

            if not sst_dates or not sss_doys or not aviso_dates:
                continue

            # Convert SSS YYYYDOY entries to YYYYMMDD
            sss_dates_converted: Set[str] = set()
            for doy_str in sss_doys:
                try:
                    if len(doy_str) == 7 and doy_str[:4].isdigit() and doy_str[4:].isdigit():
                        year_i = int(doy_str[:4])
                        doy = int(doy_str[4:])
                        date_obj = datetime(year_i, 1, 1) + timedelta(days=doy - 1)
                        sss_dates_converted.add(date_obj.strftime("%Y%m%d"))
                except Exception:
                    continue

            # Extract available AVISO months for the year
            aviso_months: Set[str] = set()
            for adate in aviso_dates:
                if len(adate) == 8 and adate.endswith('01'):
                    aviso_months.add(adate[4:6])

            # Daily intersection requiring SST day present, SSS day present, and AVISO month present
            for sst_day in sst_dates:
                if len(sst_day) != 8 or not sst_day.isdigit():
                    continue
                month = sst_day[4:6]
                if sst_day in sss_dates_converted and month in aviso_months:
                    intersected_dates.append(sst_day)

        # Remove duplicates and sort
        intersected_dates = sorted(list(set(intersected_dates)))

        logger.info(f"Found {len(intersected_dates)} dates with complete satellite data (daily intersection)")
        return intersected_dates
    
    def _initialize_date_statuses(self, complete_dates: List[str]) -> Dict[str, DateStatus]:
        """Initialize or update date statuses for all complete dates"""
        logger.info("Initializing date statuses...")
        
        date_statuses = {}
        
        for date_str in complete_dates:
            if date_str not in self.checkpoint_data:
                # New date, create status
                date_status = DateStatus(
                    date=date_str,
                    sst_available=True,
                    sss_available=True,
                    aviso_available=True,
                    complete=True,
                    processed=False
                )
                # Store in checkpoint as dict for persistence
                self.checkpoint_data[date_str] = asdict(date_status)
                # Store in return dict as DateStatus object for processing
                date_statuses[date_str] = date_status
            else:
                # Existing date, ensure status is up to date
                status_data = self.checkpoint_data[date_str]
                status_data['sst_available'] = True
                status_data['sss_available'] = True
                status_data['aviso_available'] = True
                status_data['complete'] = True
                
                # Convert back to DateStatus object for processing
                date_status = DateStatus(
                    date=date_str,
                    sst_available=status_data.get('sst_available', False),
                    sss_available=status_data.get('sss_available', False),
                    aviso_available=status_data.get('aviso_available', False),
                    complete=status_data.get('complete', False),
                    processed=status_data.get('processed', False),
                    error=status_data.get('error'),
                    retry_count=status_data.get('retry_count', 0),
                    last_attempt=status_data.get('last_attempt'),
                    output_file=status_data.get('output_file')
                )
                date_statuses[date_str] = date_status
        
        self._save_checkpoint()
        return date_statuses
    
    def _query_nespreso_api(self, date_str: str) -> Tuple[bool, Optional[str]]:
        """Query the NeSPReSO API for a specific date"""
        try:
            # Convert date string to YYYY-MM-DD format
            date_obj = datetime.strptime(date_str, "%Y%m%d")
            formatted_date = date_obj.strftime("%Y-%m-%d")
            
            # Prepare request
            payload = {"date": formatted_date}
            headers = {"Content-Type": "application/json"}
            
            logger.info(f"Querying NeSPReSO API for date: {formatted_date}")
            
            response = requests.post(
                self.config.api_url,
                json=payload,
                headers=headers,
                timeout=300  # 5 minutes timeout
            )
            
            if response.status_code == 200:
                # Save the NetCDF file
                output_filename = f"nespreso_grid_{formatted_date}.nc"
                output_path = os.path.join(self.config.output_dir, output_filename)
                
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                
                logger.info(f"Successfully processed date {formatted_date}, saved to {output_path}")
                return True, output_path
            else:
                error_msg = f"API request failed with status {response.status_code}: {response.text}"
                logger.error(f"Date {formatted_date}: {error_msg}")
                return False, error_msg
                
        except requests.exceptions.Timeout:
            error_msg = "Request timed out"
            logger.error(f"Date {date_str}: {error_msg}")
            return False, error_msg
        except requests.exceptions.RequestException as e:
            error_msg = f"Request failed: {str(e)}"
            logger.error(f"Date {date_str}: {error_msg}")
            return False, error_msg
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(f"Date {date_str}: {error_msg}")
            return False, error_msg
    
    def _process_date(self, date_str: str) -> bool:
        """Process a single date and update its status"""
        if not self.running:
            return False
        
        # Get status from checkpoint data (it's stored as dict)
        if date_str not in self.checkpoint_data:
            logger.error(f"Date {date_str} not found in checkpoint data")
            return False
            
        status = self.checkpoint_data[date_str]
        
        # Skip if already processed successfully
        if status.get('processed') and status.get('output_file'):
            logger.info(f"Date {date_str} already processed, skipping")
            return True
        
        # Update attempt info
        status['last_attempt'] = datetime.now().isoformat()
        status['retry_count'] = status.get('retry_count', 0) + 1
        
        logger.info(f"Processing date {date_str} (attempt {status['retry_count']})")
        
        # Query the API
        success, result = self._query_nespreso_api(date_str)
        
        if success:
            # Update status for successful processing
            status['processed'] = True
            status['error'] = None
            status['output_file'] = result
            logger.info(f"Date {date_str} processed successfully")
            return True
        else:
            # Update status for failed processing
            status['error'] = result
            if status['retry_count'] >= self.config.retry_attempts:
                logger.error(f"Date {date_str} failed after {status['retry_count']} attempts")
            else:
                logger.warning(f"Date {date_str} failed, will retry (attempt {status['retry_count']}/{self.config.retry_attempts})")
            return False
    
    def _process_dates_batch(self, date_statuses: Dict[str, DateStatus]) -> None:
        """Process dates in batches with progress tracking"""
        logger.info("Starting batch processing of dates...")
        
        # Get unprocessed dates
        unprocessed_dates = [
            date_str for date_str, status in date_statuses.items()
            if not status.processed or not status.output_file
        ]
        
        if not unprocessed_dates:
            logger.info("All dates have been processed successfully!")
            return
        
        logger.info(f"Found {len(unprocessed_dates)} dates to process")
        
        # Add safety check for very large numbers
        if len(unprocessed_dates) > 1000:
            logger.warning(f"Large number of dates to process ({len(unprocessed_dates)}). This may take a long time.")
            logger.info("Consider processing in smaller batches or using Ctrl+C to pause and resume later.")
        
        # Process dates with thread pool
        try:
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                # Submit all tasks
                logger.info(f"Submitting {len(unprocessed_dates)} tasks to thread pool...")
                future_to_date = {
                    executor.submit(self._process_date, date_str): date_str
                    for date_str in unprocessed_dates
                }
                
                logger.info("All tasks submitted. Processing results...")
                
                # Process completed tasks
                completed = 0
                failed = 0
                for future in as_completed(future_to_date):
                    if not self.running:
                        logger.info("Shutdown requested, stopping processing...")
                        break
                    
                    date_str = future_to_date[future]
                    try:
                        success = future.result()
                        if success:
                            completed += 1
                        else:
                            failed += 1
                        
                        # Save checkpoint periodically
                        if completed % 10 == 0:
                            self._save_checkpoint()
                            logger.info(f"Progress: {completed}/{len(unprocessed_dates)} dates completed, {failed} failed")
                            
                    except Exception as e:
                        failed += 1
                        logger.error(f"Date {date_str} processing failed with exception: {e}")
                        if date_str in self.checkpoint_data:
                            self.checkpoint_data[date_str]['error'] = str(e)
                        else:
                            logger.error(f"Date {date_str} not found in checkpoint data")
                
                # Final checkpoint save
                self._save_checkpoint()
                logger.info(f"Batch processing completed. {completed}/{len(unprocessed_dates)} dates processed successfully, {failed} failed")
                
        except Exception as e:
            logger.error(f"Thread pool processing failed: {e}")
            self._save_checkpoint()
            raise
    
    def build_archive(self):
        """Main method to build the complete archive"""
        logger.info("Starting NeSPReSO archive building process...")
        
        try:
            # Step 1: Get complete dates using the scanner
            complete_dates = self._get_complete_dates()
            
            if not complete_dates:
                logger.error("No dates found with complete satellite data!")
                return
            
            # Step 2: Initialize date statuses
            date_statuses = self._initialize_date_statuses(complete_dates)
            
            # Step 3: Process dates
            self._process_dates_batch(date_statuses)
            
            # Step 4: Generate summary report
            self._generate_summary_report()
            
            logger.info("Archive building process completed successfully!")
            
        except KeyboardInterrupt:
            logger.info("Process interrupted by user")
            self._save_checkpoint()
        except Exception as e:
            logger.error(f"Archive building failed: {e}")
            self._save_checkpoint()
            raise
    
    def _generate_summary_report(self):
        """Generate a summary report of the archive building process"""
        logger.info("Generating summary report...")
        
        total_dates = len(self.checkpoint_data)
        processed_dates = sum(1 for status in self.checkpoint_data.values() if status['processed'])
        failed_dates = sum(1 for status in self.checkpoint_data.values() if status['error'])
        
        report = {
            "summary": {
                "total_dates": total_dates,
                "processed_dates": processed_dates,
                "failed_dates": failed_dates,
                "success_rate": f"{(processed_dates/total_dates)*100:.1f}%" if total_dates > 0 else "0%"
            },
            "processed_dates": [
                {
                    "date": date_str,
                    "output_file": status['output_file'],
                    "last_attempt": status['last_attempt']
                }
                for date_str, status in self.checkpoint_data.items()
                if status['processed']
            ],
            "failed_dates": [
                {
                    "date": date_str,
                    "error": status['error'],
                    "retry_count": status['retry_count'],
                    "last_attempt": status['last_attempt']
                }
                for date_str, status in self.checkpoint_data.items()
                if status['error']
            ],
            "timestamp": datetime.now().isoformat()
        }
        
        # Save report
        report_path = os.path.join(self.config.output_dir, "archive_summary.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Summary report saved to {report_path}")
        logger.info(f"Archive Summary: {processed_dates}/{total_dates} dates processed successfully ({report['summary']['success_rate']})")
    
    def resume_archive(self):
        """Resume archive building from checkpoint"""
        logger.info("Resuming archive building from checkpoint...")
        
        if not self.checkpoint_data:
            logger.info("No checkpoint found, starting fresh...")
            self.build_archive()
            return
        
        # Convert checkpoint data back to DateStatus objects for processing
        logger.info("Converting checkpoint data to date statuses...")
        date_statuses = {}
        for date_str, status_data in self.checkpoint_data.items():
            # Create DateStatus object from checkpoint data
            date_status = DateStatus(
                date=date_str,
                sst_available=status_data.get('sst_available', False),
                sss_available=status_data.get('sss_available', False),
                aviso_available=status_data.get('aviso_available', False),
                complete=status_data.get('complete', False),
                processed=status_data.get('processed', False),
                error=status_data.get('error'),
                retry_count=status_data.get('retry_count', 0),
                last_attempt=status_data.get('last_attempt'),
                output_file=status_data.get('output_file')
            )
            date_statuses[date_str] = date_status
        
        logger.info(f"Converted {len(date_statuses)} date statuses from checkpoint")
        
        # Check if there are any unprocessed dates
        unprocessed_count = sum(1 for status in date_statuses.values() 
                              if not status.processed or not status.output_file)
        
        if unprocessed_count == 0:
            logger.info("All dates in checkpoint have been processed successfully!")
            self._generate_summary_report()
            return
        
        logger.info(f"Found {unprocessed_count} unprocessed dates to continue with...")
        
        # Continue processing
        self._process_dates_batch(date_statuses)
        self._generate_summary_report()

def main():
    """Main entry point"""
    # Configuration
    config = ArchiveConfig()
    
    # Create archive builder
    builder = ArchiveBuilder(config)
    
    # Check if resuming
    if os.path.exists(config.checkpoint_file):
        logger.info("Checkpoint file found. Do you want to resume? (y/n): ")
        response = input().lower().strip()
        if response in ['y', 'yes']:
            logger.info("Resuming archive building...")
            builder.resume_archive()
        else:
            logger.info("Starting fresh archive building...")
            builder.build_archive()
    else:
        logger.info("No checkpoint found, starting fresh archive building...")
        builder.build_archive()

if __name__ == "__main__":
    import sys
    
    # Check if user wants to just check status
    if len(sys.argv) > 1 and sys.argv[1] == "--status":
        # Just check status without running
        config = ArchiveConfig()
        builder = ArchiveBuilder(config)
        
        if not builder.checkpoint_data:
            print("No checkpoint found - no archive building has been started yet.")
        else:
            total = len(builder.checkpoint_data)
            processed = sum(1 for status in builder.checkpoint_data.values() if status.get('processed'))
            failed = sum(1 for status in builder.checkpoint_data.values() if status.get('error'))
            remaining = total - processed - failed
            
            print(f"Archive Status:")
            print(f"  Total dates: {total}")
            print(f"  Processed: {processed}")
            print(f"  Failed: {failed}")
            print(f"  Remaining: {remaining}")
            if total > 0:
                progress = (processed / total) * 100
                print(f"  Progress: {progress:.1f}%")
    else:
        main()
