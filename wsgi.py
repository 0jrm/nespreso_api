#!/usr/bin/env python3
"""
WSGI entry point for NeSPReSO API
usage:
gunicorn -w 2 -c config/gunicorn.conf.py 'wsgi:app'
"""

from services.api.app import create_app

app = create_app()

# Configure gunicorn settings
app.config['TIMEOUT'] = 1800  # 30 minutes timeout
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max request size

if __name__ == "__main__":
    app.run(host="localhost", port=5000, debug=False)
