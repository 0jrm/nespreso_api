import requests
import tempfile
import xarray as xr
import numpy as np

def test_equivalence():
    payload = {
        "lat": [25.0, 26.0],
        "lon": [-90.0, -91.0],
        "date": ["2022-01-01", "2022-01-02"]
    }
    # Old service
    r_old = requests.post("http://localhost:5000/predict", json=payload)
    assert r_old.status_code == 200, r_old.text
    # New service
    r_new = requests.post("http://localhost:5000/v1/profile", json=payload)
    assert r_new.status_code == 200, r_new.text
    # Save to temp files
    with tempfile.NamedTemporaryFile(suffix=".nc") as f_old, tempfile.NamedTemporaryFile(suffix=".nc") as f_new:
        f_old.write(r_old.content)
        f_old.flush()
        f_new.write(r_new.content)
        f_new.flush()
        ds_old = xr.open_dataset(f_old.name)
        ds_new = xr.open_dataset(f_new.name)
        # Compare all variables
        for var in ds_old.data_vars:
            np.testing.assert_allclose(ds_old[var].values, ds_new[var].values, err_msg=f"Mismatch in {var}")
        for var in ds_old.coords:
            np.testing.assert_allclose(ds_old[var].values, ds_new[var].values, err_msg=f"Mismatch in coord {var}") 