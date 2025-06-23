# NeSPReSO API and Client

## Overview

This project is a **modular Flask-based** service for generating NeSPReSO synthetic temperature and salinity profiles for specified latitude, longitude, and date inputs. The codebase is now organized into clear layers (kernel, accessor, API), with robust CI/CD, property-based tests, and observability (Prometheus metrics).

## Quickstart: New API Usage

### Running the Flask Server (New Modular Service)

```bash
PYTHONPATH=nespreso_api:nespreso_api/eoas-pyutils conda run -n nespreso python nespreso_api/wsgi.py
```

This will run the Flask app on port `5000`. The main API endpoint is now:

```
POST http://localhost:5000/v1/profile
```

**Request JSON:**
```json
{
  "lat": [25.0, 26.0],
  "lon": [-90.0, -91.0],
  "date": ["2022-01-01", "2022-01-02"]
}
```
**Response:** NetCDF file with predicted profiles.

### OpenAPI Documentation

The API is documented in `docs/api.yaml` (OpenAPI 3.0). Every new endpoint must have a stub here.

### Prometheus Metrics

A `/metrics` endpoint is available for Prometheus scraping (latency, HTTP codes, etc).

### Testing and CI

- Linting, type-checking, and property-based tests are run in CI.
- A smoke test spins up the Flask app and hits `/v1/profile`.
- Property-based tests ensure NetCDF outputs are monotonic and physically valid.

## Project Structure

- `services/kernel/handler.py`: Loads TorchScript model, exposes `infer`, and provides PCA objects.
- `services/accessor/sat.py`: Loads satellite data, prepares model inputs, LRU-cached, circuit-breaker protected.
- `services/api/app.py`: Flask app factory, `/v1/profile` endpoint, Pydantic validation, in-memory NetCDF, logging, metrics, batch guard.
- `wsgi.py`: Entrypoint for running the modular Flask app.
- `docs/api.yaml`: OpenAPI documentation for all endpoints.
- `requirements.txt`: All dependencies, including Prometheus, tenacity, and Hypothesis.
- `.github/workflows/ci.yml`: CI pipeline with artefact upload and smoke test.

## Legacy API (Deprecated)

The old `/predict` endpoint in `nespreso_flask.py` is now deprecated. Use `/v1/profile` for all new integrations.

## Example: Calling the New API

```python
import requests
payload = {
    "lat": [25.0, 26.0],
    "lon": [-90.0, -91.0],
    "date": ["2022-01-01", "2022-01-02"]
}
r = requests.post("http://localhost:5000/v1/profile", json=payload)
with open("output.nc", "wb") as f:
    f.write(r.content)
```

## Developer Notes

- All new endpoints must have OpenAPI stubs and property-based tests for output invariants.
- The codebase is ready for production deployment, with observability, resilience, and CI/CD best practices.
- See `.cursor/todo.md` for phase progress and remaining tasks.

## License

This project is licensed under the MIT License.
