#!/bin/bash
# Startup script for NeSPReSO API with proper timeout configuration

echo "Starting NeSPReSO API server..."

# Kill any existing processes
pkill -f "gunicorn.*wsgi:app" || true

# Start server with proper configuration
gunicorn -c gunicorn.conf.py 'wsgi:app' \
    --log-file=wsgi.log \
    --log-level=info \
    --timeout=1800 \
    --keep-alive=2 \
    --max-requests=1000 \
    --max-requests-jitter=50 \
    --worker-tmp-dir=/dev/shm

echo "Server started. Check wsgi.log for details."
