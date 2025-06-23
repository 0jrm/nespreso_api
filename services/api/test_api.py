import requests
import numpy as np

def test_profile_endpoint():
    url = "http://localhost:5000/v1/profile"
    payload = {
        "lat": [25.0, 26.0],
        "lon": [-90.0, -91.0],
        "date": ["2022-01-01", "2022-01-02"]
    }
    r = requests.post(url, json=payload)
    assert r.status_code == 200, r.text
    assert r.headers["Content-Type"].startswith("application/x-netcdf")
    assert r.content[:3] == b'CDF', "Not a NetCDF file (missing magic bytes)" 