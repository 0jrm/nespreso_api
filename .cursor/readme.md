# Ocean-Profiles-API

**Goal.**  Map `(lat, lon, time)` ➞ vertical profiles of temperature (°C) and salinity (psu) using a frozen neural estimator that ingests co-located satellite SST, SSH, SSS, winds, and bathymetry.

## Architecture

1. **Accessor (CPU)**  
   Flask async service.  Fetches remote‐sensed tiles, interpolates them onto requested coordinates, normalises with training stats, assembles an `n×d` tensor.

2. **Kernel (GPU)**  
   TorchServe (or Triton) with scripted model; predicts an `n×p` tensor of T/S.  No I/O or auth.

3. **Façade (stateless)**  
   Thin gateway—authN/Z, rate-limit, streams output NetCDF chunks.

See `docs/design.md` for UML and latency budget.

### Quick start

```bash
make dev        # build accessor+kernel with hot-reload
make test       # run pytest + mypy
make deploy     # render Helm chart and push OCI images
