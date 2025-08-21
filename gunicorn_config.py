import multiprocessing
import os

# ---------------------------------------------------------------------------
# Gunicorn runtime configuration
# ---------------------------------------------------------------------------
# This file is *discovered automatically* by Gunicorn when passed via the `-c`
# flag, e.g.:
#     gunicorn -c nespreso_api/gunicorn_config.py "nespreso_api.wsgi:app"
# or by setting the environment variable:
#     export GUNICORN_CMD_ARGS="-c nespreso_api/gunicorn_config.py"
# ---------------------------------------------------------------------------

# Bind interface/port – keep in sync with Docker/Helm charts if present.
bind = os.getenv("GUNICORN_BIND", "0.0.0.0:5000")

# Workers:2×CPU + 1 is the usual rule-of-thumb, but it's crashing ozavala
# workers = int(os.getenv("GUNICORN_WORKERS", (multiprocessing.cpu_count() * 2) + 1))
workers = 4

# >>>>>  CRITICAL  <<<<<
# The default Gunicorn timeout is 30 s.  The first request to the /v1/profile
# endpoint may trigger a *cold download* of satellite tiles (via earthaccess &
# Copernicus Marine), which can easily exceed that threshold.  When the
# timeout is hit Gunicorn *kills* the worker and the client sees
#   httpx.RemoteProtocolError: Server disconnected without sending a response
# Raising the timeout gives the worker a fighting chance to finish.
# ---------------------------------------------------------------------------

timeout = int(os.getenv("GUNICORN_TIMEOUT", 1800))  # 30 minutes for unlimited batch processing

# Memory and performance settings for large batch processing
max_requests = int(os.getenv("GUNICORN_MAX_REQUESTS", 1000))
max_requests_jitter = int(os.getenv("GUNICORN_MAX_REQUESTS_JITTER", 100))
preload_app = os.getenv("GUNICORN_PRELOAD_APP", "true").lower() == "true"

# Log-level defaults to *info* but is configurable for debug sessions.
loglevel = os.getenv("GUNICORN_LOG_LEVEL", "info")

# Opt-in: sync workers are fine for our IO-heavy workload, but can be tweaked.
worker_class = os.getenv("GUNICORN_WORKER_CLASS", "sync") 
