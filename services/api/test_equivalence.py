import tempfile

import numpy as np
import xarray as xr
from flask.testing import FlaskClient


def test_equivalence(client: FlaskClient):  # type: ignore[override]
    payload = {
        "lat": [25.0, 26.0],
        "lon": [-90.0, -91.0],
        "date": ["2022-01-01", "2022-01-02"],
    }

    # Old (legacy) endpoint
    resp_old = client.post("/predict", json=payload)
    assert resp_old.status_code == 200, resp_old.json

    # New endpoint
    resp_new = client.post("/v1/profile", json=payload)
    assert resp_new.status_code == 200, resp_new.json

    # Compare NetCDF outputs -------------------------------------------------
    with tempfile.NamedTemporaryFile(suffix=".nc") as f_old, tempfile.NamedTemporaryFile(
        suffix=".nc"
    ) as f_new:
        f_old.write(resp_old.data)
        f_old.flush()
        f_new.write(resp_new.data)
        f_new.flush()

        ds_old = xr.open_dataset(f_old.name)
        ds_new = xr.open_dataset(f_new.name)

        # All variables + coords must match exactly (they are zeros in mocks)
        for var in ds_old.data_vars:
            np.testing.assert_allclose(
                ds_old[var].values, ds_new[var].values, err_msg=f"Mismatch in {var}"
            )
        for var in ds_old.coords:
            np.testing.assert_allclose(
                ds_old[var].values,
                ds_new[var].values,
                err_msg=f"Mismatch in coord {var}",
            ) 