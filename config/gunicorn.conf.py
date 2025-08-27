#!/usr/bin/env python3
"""
Gunicorn configuration file for NeSPReSO API
"""

# Server socket
bind = "0.0.0.0:5000"
backlog = 2048

# Worker processes
workers = 2
worker_class = "sync"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 50
preload_app = False

# Timeouts
timeout = 1800  # 30 minutes - increase this for large datasets
keepalive = 2
graceful_timeout = 30
worker_tmp_dir = "/dev/shm"  # Use shared memory for better performance

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = "nespreso_api"

# Server mechanics
daemon = False
pidfile = None
umask = 0
user = None
group = None
tmp_upload_dir = None

# SSL (if needed)
# keyfile = None
# certfile = None

# Memory management
max_requests = 1000
max_requests_jitter = 50
preload_app = False

# Restart workers after this many requests, to help prevent memory leaks
max_requests = 1000
max_requests_jitter = 50
