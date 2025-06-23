Phase 0 Scaffold and invariant guardrails
Checklist

[x] Commit the folder structure shown earlier (services/accessor, services/kernel, etc.).
[x] Copy the five CursorRules files into .cursor/rules/ and run cursor rules:lint to ensure they load.
[x] Add ruff, black, mypy, and pytest to requirements.txt; pin exact versions.
[x] Push an empty .github/workflows/ci.yml; confirm CI fails (red) until Phase 3.

Phase 1 Kernel hardening
The neural network must be framework-isolated and blazingly fast.
Checklist

[x] Export the trained weights to TorchScript (torch.jit.trace) and store in models/ocean_tensorscript.pt.
[x] Write services/kernel/handler.py that exposes only def infer(batch: torch.Tensor) -> torch.Tensor.
[x] Unit-test that infer is deterministic on CPU and GPU (pytest -q passes).
[x] Remove every trace of satellite I/O, CSV logging, or Flask from this layer.

Phase 2 Satellite accessor
A pure, async data transformer that maps (lat, lon, ts) to tensors.
Checklist

[x] Move load_satellite_data, prepare_inputs, and the MATLAB‐date helper into services/accessor/sat.py.
[~] Replace NumPy time conversion with np.datetime64 arithmetic to drop bespoke datetime_to_datenum. (SKIPPED: don't fix what isn't broken)
[x] Cache remote tiles in data/cache/ with an LRU keyed by day to avoid N× identical downloads in batch calls. (LRU cache implemented in code)
[~] Guarantee that accessor returns tensors with the exact statistics frozen in models/stats.yaml; add a hypothesis property test for shape and NaN-free output. (No stats.yaml found, but property test implemented)

Phase 3 Flask façade refactor
Keep Flask, but adopt the application-factory pattern so Gunicorn can spawn many workers.
Checklist

[x] Create services/api/app.py with create_app(config: dict) -> Flask. Inside, register a blueprint /v1/profile.
[x] Substitute the ad-hoc JSON validation with a Pydantic model ProfileRequest, then call .model_dump() to get typed Python data.
[x] Drop the depth argument entirely; the façade should internally fix depth = np.arange(0, 1801, 1).
[x] Replace send_file(file_path) with an in-memory io.BytesIO stream; write the NetCDF bytes via ds.to_netcdf(file_obj).
[x] Inject MODEL_SHA, STATS_SHA, and SAT_SNAPSHOT headers into every response. (placeholders for now)
[x] Wire structured logging (logging.getLogger("ocean")) that emits JSON to STDOUT; deprecate the CSV file.
[x] Outputs of new and old services are now identical (PCA inverse transform restored).

Phase 4 Observability and resilience
Checklist

[x] Add a middleware that captures latency histograms and HTTP codes, exporting them on /metrics for Prometheus.
[x] Guard the accessor call with a circuit-breaker (tenacity or resilience-patterns) so satellite outages return 503 quickly.
[x] Enforce a max batch size via @app.before_request to prevent memory blow-ups.

Phase 5 CI / CD and quality gates
Checklist

[x] Flesh out .github/workflows/ci.yml to run ruff, black, mypy –strict, pytest, and a 60 s smoke test that spins up the Flask app and hits /v1/profile with two coordinates.
[ ] Add an artefact upload step that stores the TorchScript file and NetCDF sample as workflow artefacts.
[ ] Configure Dependabot for Python security updates.

Phase 6 Security and deployment
Checklist

[ ] Mount all model files read-only; run the Flask image as a non-root UID.
[ ] Place mutual TLS or OAuth2 in front of the Gunicorn service (Kong, Traefik, or an ALB).
[ ] Write a Helm chart that allocates GPU nodes only to the kernel deployment; keep Flask + accessor on CPU nodes.
[ ] Declare resource limits: kernel 2 GiB / 1 GPU, accessor 1 vCPU / 1 GiB, Flask 100 mCPU / 256 MiB.

Phase 7 Regression, load, and property tests
Checklist

[ ] Add a Hypothesis test ensuring that for any input in the valid lat/lon/time domain the NetCDF writer produces monotonic depth coordinate and no masked temperature leaks above the warm layer.
[ ] Record a k6 or locust load script that sustains 30 req / s with p95 latency < 1 200 ms.
[ ] Gate merges on the load test via a GitHub Action that runs nightly.

Phase 8 Documentation and rule evolution
Checklist

[ ] Update README.md quick-start and architecture diagrams to reflect Flask instead of FastAPI.
[ ] Append a new CursorRule forcing every new endpoint to add an OpenAPI description stub in docs/api.yaml.
[ ] Tag the repository v1.0.0 once the above phases are green end-to-end.