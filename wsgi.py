#!/usr/bin/env python3
import sys
import os
import logging

# Configure logging
logging.basicConfig(
    filename='/var/www/virtualhosts/nespreso.coaps.fsu.edu/nespreso_api/wsgi.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)

logging.info("================== Initializing Nespreso API ===========================")

# Add the project directory to the sys.path
project_home = '/var/www/virtualhosts/nespreso.coaps.fsu.edu/nespreso_api'
if project_home not in sys.path:
    sys.path.insert(0, project_home)
    logging.info(f"Added {project_home} to sys.path")

try:
    from app import app as application
    logging.info("WSGI application loaded successfully.")
except Exception as e:
    logging.exception("Failed to load WSGI application.")
    raise

logging.info("!!!!!!!!!! Done initializing WSGI application !!!!!!!!!!!!!!!!!")
