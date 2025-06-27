# Project Structure

This document describes the organization of the NeSPReSO repository (API and client), explaining the purpose of each major folder and module. 

**Repository Root** (`nespreso_api/` repo):
- **`nespreso_api/`** – Main Python package for the NeSPReSO API. This contains all backend code.
  - **`services/`** – Application logic organized by domain:
    - **`api/`** – The Flask API layer.
      - `app.py` – Flask application factory and route definitions (e.g. the `/v1/profile` endpoint). It validates requests (using Pydantic), orchestrates calls to the accessor and kernel, and constructs the NetCDF response.
      - `metrics.py` – Prometheus metrics endpoint and request timing middleware. Defines the `/metrics` route and histograms/counters for monitoring.
      - *Tests:* `test_api.py` (smoke test for the profile endpoint), `test_property_netcdf.py` (property-based tests on the NetCDF output), `test_equivalence.py` (ensures `/predict` and `/v1/profile` outputs are identical).
    - **`accessor/`** – Satellite data access and input preparation.
      - `sat.py` – Functions to load satellite data (SSS, SST, SSH) for given coordinates and dates. Uses cached remote data sources and interpolates values. Also includes `prepare_inputs(...)` to transform inputs (time, lat, lon, etc.) into the feature tensor expected by the model.
      - *Tests:* `test_sat.py` (Hypothesis tests to ensure `prepare_inputs` output shape and absence of NaNs).
    - **`kernel/`** – ML model handling (the "core" prediction engine).
      - `handler.py` – Loads the TorchScript model and PCA objects, and provides `infer(batch)` to perform a prediction on an input tensor. Also provides `get_pca_objects()` to retrieve the PCA transformers needed to post-process model output. This layer is kept framework-specific (PyTorch) and isolated from Flask or data concerns.
      - *Tests:* `test_handler.py` (would test model inference determinism or CPU/GPU parity – not fully shown in this branch).
    - **`utils.py`** – Utility functions (used by both the API and the Python client). For example, `preprocess_inputs(...)` which converts various input formats (list, NumPy array, pandas Series, etc.) into the standardized list-of-floats format for lat, lon, date. The client uses this to accept flexible input types.
  - **`models/`** – Model files and related artifacts.
    - `ocean_tensorscript.pt` – The TorchScript version of the trained neural network model for profile prediction. This is a serialized PyTorch model that the API loads at runtime. (The model predicts principal components which are then transformed back to physical values using PCA.)
    - `pca_stats.pkl` – Contains the PCA objects (`pca_temp`, `pca_sal`) and `input_params` needed for post-processing model output. This file is loaded directly by the API and replaces the previous use of a separate checkpoint file.
  - **`data/`** – (If present) Data files or caches. In this project, large external datasets (satellite data grids) are not bundled here, but this directory could be used for any packaged reference data. For example, a future `stats.yaml` (with mean/std for inputs) or cached tile files might reside here. Currently, the accessor uses in-memory caching via `functools.lru_cache` instead.
  - **`wsgi.py`** – Entrypoint for the Flask application. It creates the app using `create_app()` from `services/api/app.py`. This file is used by WSGI servers (Gunicorn, Apache mod_wsgi, etc.) to serve the application. It also sets up basic logging on startup. You can run `python nespreso_api/wsgi.py` for development (it will default to running on `0.0.0.0:5000` in debug mode if invoked directly).
- **`nespresso_client.py`** – Python client for the API. This is a convenience module that allows users to fetch predictions without dealing with HTTP manually. It provides:
  - `get_predictions(lat, lon, date, filename="output.nc", api_url=None)` – a synchronous function that posts to the API and saves the returned NetCDF to a file.
  - Internally it uses `httpx` and `asyncio` (see `fetch_predictions`), and it accepts flexible input types (leveraging `services/utils.py`).
  - This client is intended for researchers or applications that want to integrate NeSPReSO predictions easily. It defaults to `api_url="http://0.0.0.0:5000/v1/profile"` but can be pointed to a remote host.
- **`docs/`** – Documentation files.
  - `api.yaml` – OpenAPI 3.0 specification for the API endpoints. It currently documents the `/v1/profile` POST endpoint, including expected input schema and responses.
  - *(Proposed)* `structure.md` – (this document) Overview of the repository structure and components.
- **`nespresso-ui/`** – Front-end web application (React). This directory contains a Create React App project that provides a UI for drawing regions on a map and requesting profiles.
  - *Key files:* `src/App.js` contains the logic to collect user input (points/lines/areas and dates) and submit a request to the API. It expects the API at the `/predict` endpoint (which is the legacy endpoint; this may be updated to `/v1/profile`). The UI then initiates a download of the NetCDF file and displays basic stats from response headers.
  - This UI is mainly for demonstration and user-friendly access to NeSPReSO. It can be built (`npm run build`) and served as a static app. (Deployment of the UI is separate from the Flask API, though they can be hosted together under the same domain if configured.)
- **`deployment/`** – (Planned) Deployment configuration files.
  - This folder is intended for Dockerfiles, Kubernetes manifests, or Helm charts. For example, a `Chart.yaml` and templates would live here for deploying on a cluster. In the current branch, this is a placeholder – the infrastructure-as-code is still under development.
- **`.github/workflows/`** – CI/CD pipeline definitions.
  - `ci.yml` – GitHub Actions workflow for continuous integration. It installs the environment, then runs code quality checks (lint, format, type-check) and the test suite, and finally does a simple live test of the API. On success, it also uploads artifacts (the model and a sample output) for record-keeping. This ensures that every commit to main (or PR) is validated.
- **`old_files-nespreso_api/`** – Archived legacy code (for reference only).
  - This directory contains previous iterations of the code (monolithic Flask app, training scripts, etc.). Notably, `nespreso_flask.py` here was the old Flask application with a `/predict` route, and other utilities like `singleFileModel_SAT.py`. The new modular design has superseded this, but the files are kept for history. They are **not used** in the current deployment.
- **Other files at root:**
  - `README.md` – Comprehensive overview, quickstart instructions, usage examples, and troubleshooting tips for the project.
  - `requirements.txt` – Python package requirements for development (mostly linters and test tools). *(For runtime, see environment YAML.)*
  - `requirements.yml` (or `environment.yml`) – Conda environment specification listing all dependencies (Python and system libs) needed to run the API. Using this ensures a reproducible setup.
  - `.gitmodules` – Defines the `eoas_pyutils` submodule. `eoas_pyutils/` is an external utility library (from COAPS/FSU) that is included for potential data-handling helpers. In this project, it's mostly an **external dependency** – you typically shouldn't need to modify it. Make sure to initialize and update submodules if you clone the repo (`git submodule update --init`).
  - `.cursor/` – Configuration for the Cursor editor and project rules (for developers). Contains markdown files outlining project phases and conventions (not required for deployment).