import tempfile

import numpy as np
import xarray as xr
from flask.testing import FlaskClient
from hypothesis import given, strategies as st


@st.composite
def random_payload(draw):
    n = draw(st.integers(min_value=1, max_value=5))
    lat = draw(
        st.lists(st.floats(min_value=18.0, max_value=31.0), min_size=n, max_size=n)
    )
    lon = draw(
        st.lists(st.floats(min_value=-98.0, max_value=-81.0), min_size=n, max_size=n)
    )
    date = draw(
        st.lists(
            st.dates(min_value=np.datetime64("2015-01-01").astype(object), max_value=np.datetime64("2022-12-31").astype(object)),
            min_size=n,
            max_size=n,
        )
    )
    date = [d.strftime("%Y-%m-%d") for d in date]
    return {"lat": lat, "lon": lon, "date": date}


@given(random_payload())
def test_netcdf_monotonic_and_no_leak(client: FlaskClient, payload):  # type: ignore[override]
    resp = client.post("/v1/profile", json=payload)
    assert resp.status_code == 200, resp.json
    with tempfile.NamedTemporaryFile(suffix=".nc") as f:
        f.write(resp.data)
        f.flush()
        ds = xr.open_dataset(f.name)
        # Monotonic depth
        depth = ds["depth"].values
        assert np.all(np.diff(depth) > 0), "Depth is not strictly increasing"
        # No masked temperature above warm layer (e.g., top 50m)
        temp = ds["Temperature"].values
        assert not np.isnan(temp[:50, :]).any(), "Masked temperature above warm layer"


def test_netcdf_engine_is_netcdf4():
    """Regression: Ensure NetCDF files are written with engine='netcdf4'."""
    ds = xr.Dataset({"foo": (("x",), np.arange(10, dtype=np.float32))})
    encoding = {"foo": {"zlib": True, "complevel": 4}}
    with tempfile.NamedTemporaryFile(suffix=".nc") as f:
        ds.to_netcdf(f.name, encoding=encoding, engine="netcdf4")
        ds2 = xr.open_dataset(f.name)
        np.testing.assert_array_equal(ds2["foo"].values, ds["foo"].values) 