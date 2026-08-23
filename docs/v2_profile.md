# v2 DA profile cells

SAT `POST /v1_profile` and `POST /v1_profile/grid` remain the Ozavala TorchScript cell. `NESPRESO_MODEL_PATH` does not load v2 `.pth` files. The served DA cells are the keys in `SERVED_MODELS`.

## Served cells

`SERVED_MODELS` is `A_CRPS`, `HeaveFast`, and `ops`. `R_KIND` is `dai_sigma_o_after_H` for every served cell. The NetCDF holds μ Temperature and Salinity. It does not hold CRPS-head σ.

| key | role | input | decode |
| --- | --- | --- | --- |
| `A_CRPS` | frozen ingest and OSSE xb | 9-d SAT | `pca_inverse`, 32 PCs |
| `HeaveFast` | heave challenger | 11-d SAT and ENSO | `heave_residual_fast` |
| `ops` | LC-only second challenger | 30-d SAT, ENSO, and 19 operators | `heave_residual_fast` |

Depth is the v2 1 m `PRES` vector from the cell cache.

### A_CRPS

`A_CRPS` uses the SAT 9-d path from `prepare_inputs` with `_SAT_PARAMS`. Decode is `DECODE_PCA` (`pca_inverse`). Allowed `seed` values are `A_CRPS_SEEDS` (`42`, `43`, `44`). The default seed is `42`. Ingest R is the three-seed mean of Dai σ_o after H. `sigma_o_seed` is `mean`. The request `seed` selects the checkpoint. It does not select a per-seed R table.

### HeaveFast

`HeaveFast` splices ONI and RONI onto the SAT 9-d vector. Input width is 11. Decode is `DECODE_HEAVE` (`heave_residual_fast`). The allowed `seed` values are no query `seed` and `42`.

### ops

`ops` concatenates the 11-d SAT and ENSO vector with 19 operator columns from `sample_ops_or_503`. Decode is `DECODE_HEAVE` (`heave_residual_fast`). The allowed `seed` values are no query `seed` and `42`. Operators come from the SAT archive at request time. The training cube is not used. `sample_ops_or_503` reads MUR, SMAP, and AVISO planes. It does not open `gom_cube.zarr`. It does not zero-fill. A missing plane is HTTP 503.

`OPS_FEATURE_SPEC` operators are the following.

| op | channels | scales or window |
| --- | --- | --- |
| `grad` | `sst`, `sss`, `ssh` | `local`, `1.0deg` |
| `laplacian` | `ssh` | `1.0deg` |
| `tendency` | `sst`, `ssh` | `window_days` 7 |
| `geo_uv` | `ssh` | `local`, `1.0deg` |

`OPS_HALO_DEG` is `4.0`.

## SAT routes

The Ozavala SAT cell is unchanged.

| route | cell |
| --- | --- |
| `POST /v1_profile` | Ozavala TorchScript |
| `POST /v1_profile/grid` | Ozavala TorchScript |

Those routes do not stamp `model` on the NetCDF. They do not read `SERVED_MODELS`.

## Routes

The profile blueprint uses `url_prefix="/v1_profile"`. `register_v2_routes` adds `/<model>` and `/<model>/grid`.

| key | profile | grid |
| --- | --- | --- |
| `A_CRPS` | `POST /v1_profile/A_CRPS` | `POST /v1_profile/A_CRPS/grid` |
| `HeaveFast` | `POST /v1_profile/HeaveFast` | `POST /v1_profile/HeaveFast/grid` |
| `ops` | `POST /v1_profile/ops` | `POST /v1_profile/ops/grid` |

## Request body

Profile routes share `ProfileRequest` with SAT `POST /v1_profile`. Grid routes share `GridRequest` with SAT `POST /v1_profile/grid`.

`ProfileRequest` fields are the following.

| field | type | constraint |
| --- | --- | --- |
| `lat` | list of float | non-empty, finite, in `[-90, 90]` |
| `lon` | list of float | non-empty, finite, in `[-180, 180]` |
| `date` | list of string | non-empty, `YYYY-MM-DD` |

`lat`, `lon`, and `date` have equal length.

`GridRequest` fields are the following.

| field | type | constraint |
| --- | --- | --- |
| `date` | string | `YYYY-MM-DD` |
| `bbox` | list of four floats, optional | `[lon_min, lat_min, lon_max, lat_max]` |
| `resolution` | positive finite float, optional | degrees |

## Query seed

`seed` is a query parameter. `resolve_seed` validates it.

| key | allowed `seed` | default when omitted |
| --- | --- | --- |
| `A_CRPS` | `42`, `43`, `44` | `42` |
| `HeaveFast` | none, or `42` | `42` |
| `ops` | none, or `42` | `42` |

A non-integer `seed` is HTTP 400. A seed outside the table is HTTP 400.

## Response NetCDF

`Content-Type` is `application/x-netcdf`. Profile filename is `NeSPReSO_{model}_{dates[0]}_to_{dates[-1]}.nc`. Grid filename is `NeSPReSO_{model}_grid_{date}` plus optional `_bbox_...` and `_res_...` suffixes.

`_netcdf_response` refuses to write `sigma` or `err` data variables.

### Profile variables

| name | dims |
| --- | --- |
| `Temperature` | `depth`, `profile_number` |
| `Salinity` | `depth`, `profile_number` |
| `SSS` | `profile_number` |
| `SST` | `profile_number` |
| `AVISO` | `profile_number` |
| `time_iso` | `profile_number` |

Coordinates are `profile_number`, `depth`, `time`, `lat`, and `lon`. `depth` is the cache `PRES` vector at 1 m.

### Grid variables

| name | dims |
| --- | --- |
| `Temperature` | `depth`, `lat`, `lon` |
| `Salinity` | `depth`, `lat`, `lon` |
| `SSS` | `lat`, `lon` |
| `SST` | `lat`, `lon` |
| `AVISO` | `lat`, `lon` |
| `time_iso` | scalar string |

Coordinates are `depth`, `lat`, `lon`, and `time`.

### Global attributes

`_cell_attrs` stamps the following.

| attr | source |
| --- | --- |
| `model` | `V2Decode.model` |
| `seed` | `V2Decode.seed` as a string |
| `checkpoint` | `V2Decode.checkpoint` |
| `cache_hash` | `V2Decode.cache_hash` |
| `decode` | `V2Decode.decode` (`pca_inverse` or `heave_residual_fast`) |
| `r_kind` | `R_KIND` (`dai_sigma_o_after_H`) |
| `cache_kind` | `V2Decode.cache_kind` when set |

### sigma_o sidecar

`_attach_sigma_o` writes Dai σ_o after H on coordinate `hycom_k`. This is ingest R. It is not CRPS-head σ. If `reports/sigma_o_hycom.csv` is missing, the sidecar is omitted and the response is still 200.

| name | dims | notes |
| --- | --- | --- |
| `hycom_k` | `hycom_k` | layer index |
| `sigma_o_zmid` | `hycom_k` | layer mid-depth, m |
| `sigma_o_T` | `hycom_k` | units `degree_C`, long_name `Dai sigma_o T after H` |
| `sigma_o_S` | `hycom_k` | units `psu`, long_name `Dai sigma_o S after H` |
| `sigma_o_T_lc` | `hycom_k` | present when LC rows exist |
| `sigma_o_S_lc` | `hycom_k` | present when LC rows exist |
| `sigma_o_T_complement` | `hycom_k` | present when complement rows exist |
| `sigma_o_S_complement` | `hycom_k` | present when complement rows exist |

Sidecar attributes are the following.

| attr | value |
| --- | --- |
| `sigma_o_regime` | `all` |
| `sigma_o_seed` | `mean` for `A_CRPS`, `42` for `HeaveFast` and `ops` |
| `sigma_o_floor_T` | `0.05` |
| `sigma_o_floor_S` | `0.02` |

## Response headers

| header | value |
| --- | --- |
| `Content-Type` | `application/x-netcdf` |
| `Content-Disposition` | `attachment; filename=...` |
| `MODEL_SHA` | `V2Decode.checkpoint_stem` |
| `STATS_SHA` | `V2Decode.cache_hash`, or `unknown` |
| `SAT_SNAPSHOT` | `unknown` |

## Errors

| status | cause |
| --- | --- |
| 400 | `ProfileRequest` or `GridRequest` validation failure, unequal `lat`, `lon`, and `date` lengths, non-integer `seed`, `seed` not allowed for the cell, empty grid |
| 404 | path `{model}` not in `SERVED_MODELS`. Body includes `served`. |
| 413 | profile or grid point count exceeds `NESPRESO_MAX_PROFILES` when that cap is nonzero |
| 503 | `V2UnavailableError` from `load_sat_or_503` or `sample_ops_or_503` (missing SSS, SST, or SSH, a missing ops plane, a non-finite operator sample, or an input-width mismatch) |
| 500 | uncaught exception |

A 404 body is `{"error": "Unknown model ...", "served": ["A_CRPS", "HeaveFast", "ops"]}`.

## Environment variables

| name | `Config` field | default |
| --- | --- | --- |
| `NESPRESO_V2_ROOT` | `V2_ROOT` | `/unity/g2/jmiranda/SubsurfaceFields/Data/ISAS20_ARGO/ISAS20_project` |
| `NESPRESO_V2_CODE_HOME` | `V2_CODE_HOME` | `{NESPRESO_V2_ROOT}/NeSPReSO2_onTemplate` |
| `NESPRESO_V2_SPEC_PATH` | `V2_SPEC_PATH` | `{NESPRESO_V2_ROOT}/reports/heave_da_serve_spec.json` |
| `NESPRESO_MODEL_PATH` | `MODEL_PATH` | SAT TorchScript under `models/` |
| `NESPRESO_MAX_PROFILES` | `MAX_PROFILES` | `0` (no cap) |
| `NESPRESO_SSS_ROOT` | `SSS_ROOT` | SMAP archive |
| `NESPRESO_SST_ROOT` | `SST_ROOT` | MUR or OISST archive |
| `NESPRESO_AVISO_ROOT` | `AVISO_ROOT` | AVISO archive |
| `NESPRESO_AVISO_NEW_ROOT` | `AVISO_NEW_ROOT` | later AVISO archive |

`NESPRESO_MODEL_PATH` is the SAT TorchScript file. It is not a v2 checkpoint.

## Not served

The following keys are not in `SERVED_MODELS`. `POST /v1_profile/{key}` returns 404.

| key |
| --- |
| `conv3` |
| `bathy` |
| `wind` |
| Heave `s42d` |
| `Latent` |
| `Direct` |

OpenAPI for these routes is `docs/api.yaml`.
