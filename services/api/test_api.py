from flask.testing import FlaskClient


def test_profile_endpoint(client: FlaskClient):  # type: ignore[override]
    payload = {
        "lat": [25.0, 26.0],
        "lon": [-90.0, -91.0],
        "date": ["2022-01-01", "2022-01-02"],
    }
    resp = client.post("/v1/profile", json=payload)
    assert resp.status_code == 200, resp.json
    assert resp.headers["Content-Type"].startswith("application/x-netcdf")
    assert resp.data[:3] == b"CDF", "Not a NetCDF file (missing magic bytes)" 