"""HTTP tests for frozen v2 DA profile routes. Heavy I/O is mocked."""

from __future__ import annotations

import io
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from flask.testing import FlaskClient
from hypothesis import HealthCheck, given, settings, strategies as st

from services.kernel.v2_cells import V2Decode


def _fake_decode(model: str, seed: int, n: int) -> V2Decode:
    depth = np.arange(1801, dtype=np.float32)
    return V2Decode(
        temperature=np.zeros((1801, n), dtype=np.float32),
        salinity=np.zeros((1801, n), dtype=np.float32),
        depth=depth,
        mld=None,
        d26=None,
        model=model,
        seed=seed,
        checkpoint="model_best.pth",
        checkpoint_stem="p5_A_CRPS_v2_s42_s2",
        cache_hash="3adcff404b0b",
        decode="pca_inverse" if model == "A_CRPS" else "heave_residual_fast",
        cache_kind="heave_ops" if model == "ops" else None,
    )


@pytest.fixture()
def v2_client(app, monkeypatch) -> FlaskClient:  # type: ignore[no-untyped-def]
    def fake_run(
        model: str, seed: int, times: list[datetime], lat: np.ndarray, lon: np.ndarray
    ):
        n = int(lat.shape[0])
        zeros = np.zeros(n, dtype=np.float32)
        return _fake_decode(model, seed, n), zeros, zeros, zeros

    def fake_stamp(ds: xr.Dataset, decoded: V2Decode) -> xr.Dataset:
        ds.attrs.update(
            {
                "model": decoded.model,
                "seed": str(decoded.seed),
                "checkpoint": decoded.checkpoint,
                "cache_hash": decoded.cache_hash,
                "decode": decoded.decode,
                "r_kind": "dai_sigma_o_after_H",
            }
        )
        return ds

    monkeypatch.setattr("services.api.v2_profile._run_cell", fake_run)
    monkeypatch.setattr("services.api.v2_profile._stamp_cell", fake_stamp)
    with app.test_client() as c:
        yield c


def test_unknown_model_404(v2_client: FlaskClient) -> None:
    payload = {"lat": [25.0], "lon": [-90.0], "date": ["2020-01-01"]}
    resp = v2_client.post("/v1_profile/conv3", json=payload)
    assert resp.status_code == 404


def test_legacy_v1_profile_still_served(client: FlaskClient) -> None:
    payload = {
        "lat": [25.0, 26.0],
        "lon": [-90.0, -91.0],
        "date": ["2022-01-01", "2022-01-02"],
    }
    resp = client.post("/v1_profile", json=payload)
    assert resp.status_code == 200, resp.json
    assert resp.headers["Content-Type"].startswith("application/x-netcdf")
    with tempfile.NamedTemporaryFile(suffix=".nc") as f:
        f.write(resp.data)
        f.flush()
        ds = xr.open_dataset(f.name)
        assert "model" not in ds.attrs


def test_v2_profile_netcdf_has_ts_depth_and_model(v2_client: FlaskClient) -> None:
    payload = {"lat": [25.0], "lon": [-90.0], "date": ["2020-01-01"]}
    resp = v2_client.post("/v1_profile/A_CRPS", json=payload)
    assert resp.status_code == 200, resp.json
    assert resp.headers["MODEL_SHA"] != "unknown"
    with tempfile.NamedTemporaryFile(suffix=".nc") as f:
        f.write(resp.data)
        f.flush()
        ds = xr.open_dataset(f.name)
        assert "Temperature" in ds.data_vars
        assert "Salinity" in ds.data_vars
        assert "depth" in ds.coords
        assert ds.attrs.get("model") == "A_CRPS"
        assert ds.attrs.get("r_kind") == "dai_sigma_o_after_H"
        assert "sigma" not in ds.data_vars
        assert "err" not in ds.data_vars


def test_a_crps_bad_seed_400(v2_client: FlaskClient) -> None:
    payload = {"lat": [25.0], "lon": [-90.0], "date": ["2020-01-01"]}
    resp = v2_client.post("/v1_profile/A_CRPS?seed=99", json=payload)
    assert resp.status_code == 400


def test_missing_sat_503(app, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from services.common.v2_spec import V2UnavailableError

    def boom(*_a, **_k):  # type: ignore[no-untyped-def]
        raise V2UnavailableError("Satellite data unavailable: missing")

    monkeypatch.setattr("services.api.v2_profile._run_cell", boom)
    with app.test_client() as c:
        resp = c.post(
            "/v1_profile/HeaveFast",
            json={"lat": [25.0], "lon": [-90.0], "date": ["2020-01-01"]},
        )
    assert resp.status_code == 503


@st.composite
def _v2_payload(draw: st.DrawFn) -> dict[str, list[object]]:
    n = draw(st.integers(min_value=1, max_value=3))
    lat = draw(
        st.lists(st.floats(min_value=18.0, max_value=31.0), min_size=n, max_size=n)
    )
    lon = draw(
        st.lists(st.floats(min_value=-98.0, max_value=-81.0), min_size=n, max_size=n)
    )
    date = draw(
        st.lists(
            st.dates(
                min_value=np.datetime64("2016-01-01").astype(object),
                max_value=np.datetime64("2021-12-31").astype(object),
            ),
            min_size=n,
            max_size=n,
        )
    )
    return {
        "lat": lat,
        "lon": lon,
        "date": [d.strftime("%Y-%m-%d") for d in date],
    }


@given(_v2_payload())
@settings(
    max_examples=10,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_v2_netcdf_property_model_attr(
    v2_client: FlaskClient, payload: dict[str, list[object]]
) -> None:
    resp = v2_client.post("/v1_profile/A_CRPS", json=payload)
    assert resp.status_code == 200, resp.json
    ds = xr.open_dataset(io.BytesIO(resp.data))
    try:
        assert "Temperature" in ds
        assert "Salinity" in ds
        depth = ds["depth"].values
        assert np.all(np.diff(depth) > 0)
        assert ds.attrs.get("model") == "A_CRPS"
    finally:
        ds.close()


def test_sat_batch_single_profile_is_n1() -> None:
    from services.accessor.v2_inputs import _sat_batch

    x = _sat_batch(
        [datetime(2016, 12, 31)],
        np.array([25.0]),
        np.array([-83.0]),
        np.array([36.0]),
        np.array([300.0]),
        np.array([0.1]),
    )
    assert x.shape == (1, 9)
    assert np.isfinite(x).all()


def test_ops_columns_come_from_sat_planes_not_cube(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    calls: list[str] = []
    lats = np.linspace(23.0, 27.0, 41)
    lons = np.linspace(-85.0, -81.0, 41)
    lat_g, lon_g = np.meshgrid(lats, lons, indexing="ij")
    sst0 = (20.0 + 0.4 * (lon_g + 83.0)).astype(np.float32)
    sss0 = (36.0 + 0.15 * (lat_g - 25.0)).astype(np.float32)
    ssh0 = (0.25 * (lat_g - 25.0) + 0.1 * (lon_g + 83.0)).astype(np.float32)

    def fake_load(channel: str, day: datetime, _bbox):  # type: ignore[no-untyped-def]
        calls.append(f"{channel}:{day:%Y-%m-%d}")
        drift = 0.02 * float((day - datetime(2016, 12, 25)).days)
        if channel == "sst":
            return lats, lons, sst0 + np.float32(drift)
        if channel == "sss":
            return lats, lons, sss0
        if channel == "ssh":
            return lats, lons, ssh0 + np.float32(0.5 * drift)
        raise AssertionError(channel)

    monkeypatch.setattr("services.accessor.v2_inputs._load_channel_plane", fake_load)
    from services.accessor.v2_inputs import sample_ops_or_503

    vals = sample_ops_or_503(
        [datetime(2016, 12, 31)], np.array([25.0]), np.array([-83.0])
    )
    assert vals.shape == (1, 19)
    assert np.isfinite(vals).all()
    assert not np.allclose(vals, 0.0)
    assert any(c.startswith("sst:") for c in calls)
    assert any(c.startswith("sss:") for c in calls)
    assert any(c.startswith("ssh:") for c in calls)
    assert any("2016-12-25" in c for c in calls)


def test_ops_missing_sat_plane_is_503_not_zero(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from services.accessor.v2_inputs import sample_ops_or_503
    from services.common.v2_spec import V2UnavailableError

    def missing(channel: str, day: datetime, _bbox):  # type: ignore[no-untyped-def]
        if channel == "sst" and day == datetime(2016, 12, 31) - timedelta(days=3):
            raise V2UnavailableError("sst hole")
        lats = np.linspace(23.0, 27.0, 21)
        lons = np.linspace(-85.0, -81.0, 21)
        lat_g, lon_g = np.meshgrid(lats, lons, indexing="ij")
        field = (lat_g + lon_g).astype(np.float32)
        return lats, lons, field

    monkeypatch.setattr("services.accessor.v2_inputs._load_channel_plane", missing)
    with pytest.raises(V2UnavailableError, match="sst hole"):
        sample_ops_or_503([datetime(2016, 12, 31)], np.array([25.0]), np.array([-83.0]))


def test_sigma_o_a_crps_is_three_seed_mean() -> None:
    csv_path = Path(
        "/unity/g2/jmiranda/SubsurfaceFields/Data/ISAS20_ARGO/ISAS20_project/reports/sigma_o_hycom.csv"
    )
    if not csv_path.is_file():
        pytest.skip("sigma_o csv not on this host")
    from services.common.v2_spec import load_sigma_o

    table = load_sigma_o(csv_path, "A_CRPS", 42)
    assert table.regime == "all"
    assert table.seed_label == "mean"
    assert table.zmid_m.shape[0] >= 33
    assert not np.isfinite(table.sigma_t[33])
    assert table.sigma_t_lc is not None
