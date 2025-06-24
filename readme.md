# NeSPReSO API and Client

## Overview

NeSPReSO is a scientific service for generating **synthetic temperature and salinity profiles** in the ocean, given latitude, longitude, and date. It is designed for oceanographers, data scientists, and operational users who need fast, reliable, and physically consistent profile predictions.

This repository provides:
- A **modular Flask-based API** (with `/v1/profile` as the main endpoint)
- A Python client (`nespreso_client.py`) for easy integration
- OpenAPI documentation, CI/CD, property-based tests, and Prometheus observability

---

## Quickstart

### 1. Install Dependencies

**Recommended:** Use the provided conda environment for full reproducibility:

```bash
conda env create -n nespreso -f requirements.yml
conda activate nespreso
```

If you only have `requirements.txt`:
```bash
conda create -n nespreso python=3.10
conda activate nespreso
pip install -r requirements.txt
```

### 2. Run the API Server (Development)

```bash
PYTHONPATH=nespreso_api:nespreso_api/eoas-pyutils conda run -n nespreso python nespreso_api/wsgi.py
```
- The API will be available at `http://localhost:5000/v1/profile`
- Prometheus metrics: `http://localhost:5000/metrics`

### 3. Make a Prediction (Python Example)

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

### 4. Use the Python Client

```python
from nespreso_client import get_predictions
result = get_predictions([25.0, 26.0], [-90.0, -91.0], ["2022-01-01", "2022-01-02"], filename="output.nc")
print("NetCDF file saved as:", result)
```
- The client supports lists, numpy arrays, pandas Series, and xarray DataArrays as input.

---

## API Usage

### Main Endpoint: `/v1/profile`
- **Method:** POST
- **Request JSON:**
  ```json
  {
    "lat": [25.0, 26.0],
    "lon": [-90.0, -91.0],
    "date": ["2022-01-01", "2022-01-02"]
  }
  ```
- **Response:** NetCDF file (binary)
- **OpenAPI docs:** See [`docs/api.yaml`](docs/api.yaml)
- **Prometheus metrics:** `/metrics`

#### Example with `curl`:
```bash
curl -X POST http://localhost:5000/v1/profile \
  -H "Content-Type: application/json" \
  -d '{"lat": [25.0], "lon": [-90.0], "date": ["2022-01-01"]}' \
  --output output.nc
```

#### NetCDF Output
- The output file contains variables: `Temperature`, `Salinity`, `SSS`, `SST`, `AVISO`, `time`, `lat`, `lon`, and `depth`.
- You can open it with `xarray`, `netCDF4`, or Panoply.

---

## Project Structure

- `services/kernel/handler.py` — Loads TorchScript model, exposes `infer`, provides PCA objects
- `services/accessor/sat.py` — Loads satellite data, prepares model inputs, LRU-cached, circuit-breaker protected
- `services/api/app.py` — Flask app factory, `/v1/profile` endpoint, Pydantic validation, in-memory NetCDF, logging, metrics, batch guard
- `wsgi.py` — Entrypoint for running the modular Flask app
- `nespreso_client.py` — Python client for the API
- `docs/api.yaml` — OpenAPI documentation for all endpoints
- `requirements.txt` / `requirements.yml` — All dependencies
- `.github/workflows/ci.yml` — CI pipeline with artefact upload and smoke test

---

## Advanced/Production Deployment

- **WSGI/Gunicorn:** You can run the app with Gunicorn for production:
  ```bash
  PYTHONPATH=nespreso_api:nespreso_api/eoas-pyutils gunicorn -w 2 -b 0.0.0.0:5000 'nespreso_api.wsgi:app'
  ```
- **Apache/WSGI:** See `wsgi.py` and your Apache config for integration.
- **Prometheus:** Scrape `/metrics` for latency and error monitoring.
- **Resource limits, security, and Helm chart:** See `.cursor/todo.md` for planned productionization steps.

---

## Testing and CI

- **Run all tests:**
  ```bash
  conda run -n nespreso pytest
  ```
- **CI:** Linting, type-checking, property-based tests, and a live smoke test are run on every push.
- **Property-based tests:** Ensure NetCDF outputs are monotonic and physically valid.

---

## Troubleshooting & FAQ

- **PYTHONPATH errors:** Always run with `PYTHONPATH=nespreso_api:nespreso_api/eoas-pyutils`.
- **Model or data file not found:** Check paths in `services/kernel/handler.py` and `services/accessor/sat.py`.
- **Permission errors:** Ensure your user (or Apache) can read all model/data files.
- **API returns 503:** Satellite data source may be temporarily unavailable (circuit-breaker protection).
- **Batch size errors:** The API limits requests to 32 profiles per call.

---

## Contributing & Development

- All new endpoints must have an OpenAPI stub in `docs/api.yaml` and a property-based test for output invariants.
- See `.cursor/todo.md` for phase progress and remaining tasks.
- PRs and issues are welcome!

---

## License

This project is licensed under the MIT License.
