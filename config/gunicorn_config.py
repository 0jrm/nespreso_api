# gunicorn_config.py
import os

bind = os.getenv("GUNICORN_BIND", "0.0.0.0:5000")
workers = int(os.getenv("GUNICORN_WORKERS", "4"))

# Long timeout tolerated, but keep graceful semantics explicit.
timeout = int(os.getenv("GUNICORN_TIMEOUT", "1800"))
graceful_timeout = int(os.getenv("GUNICORN_GRACEFUL_TIMEOUT", "60"))

max_requests = int(os.getenv("GUNICORN_MAX_REQUESTS", "1000"))
max_requests_jitter = int(os.getenv("GUNICORN_MAX_REQUESTS_JITTER", "100"))

preload_app = os.getenv("GUNICORN_PRELOAD_APP", "false").lower() == "true"
loglevel = os.getenv("GUNICORN_LOG_LEVEL", "info")
worker_class = os.getenv("GUNICORN_WORKER_CLASS", "sync")

# Hardening
limit_request_line = int(os.getenv("GUNICORN_LIMIT_REQUEST_LINE", "8190"))
limit_request_fields = int(os.getenv("GUNICORN_LIMIT_REQUEST_FIELDS", "100"))
limit_request_field_size = int(os.getenv("GUNICORN_LIMIT_REQUEST_FIELD_SIZE", "8190"))
