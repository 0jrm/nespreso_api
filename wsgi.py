#!/usr/bin/env python3
# PYTHONPATH=nespreso_api:nespreso_api/eoas-pyutils gunicorn -w 2 -b 0.0.0.0:5000 'nespreso_api.wsgi:app'
import sys
import os
import logging
from services.api.app import create_app

# Configure logging
logging.basicConfig(
    filename='wsgi.log',
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
    app = create_app()
    logging.info("WSGI application loaded successfully.")
except Exception as e:
    logging.exception("Failed to load WSGI application.")
    raise

logging.info("!!!!!!!!!! Done initializing WSGI application !!!!!!!!!!!!!!!!!")

if __name__ == "__main__":
    # app.run(host="0.0.0.0", port=5000, debug=True) #gunicorn
    app.run(...) #apache
