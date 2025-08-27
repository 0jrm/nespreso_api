#!/bin/bash

echo "Stopping existing gunicorn processes..."
pkill -f "gunicorn.*wsgi:app"

echo "Waiting for processes to stop..."
sleep 2

echo "Starting gunicorn with unlimited configuration..."
gunicorn -c gunicorn_config.py -w 2 -b 0.0.0.0:5000 'wsgi:app' &

echo "Server started with:"
echo "- No batch size limits"
echo "- 30 minute timeout"
echo "- Memory optimization enabled"
echo "- Server running on http://0.0.0.0:5000"

echo "Use 'tail -f wsgi.log' to monitor logs"
