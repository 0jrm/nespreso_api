import pytest
import numpy as np
import torch
from flask import Flask

from services.api.app import create_app

# ---------------------------------------------------------------------------
# Lightweight stubs to eliminate heavyweight external dependencies
# ---------------------------------------------------------------------------

def _mock_load_satellite_data(times, lat, lon):
    n = len(times)
    return (np.zeros(n), np.zeros(n), np.zeros(n))


def _mock_infer(batch: torch.Tensor) -> torch.Tensor:  # noqa: D401
    n = batch.shape[0]
    return torch.zeros((n, 30))


class _DummyPCA:
    def inverse_transform(self, x):  # noqa: D401 – simple stub
        return np.zeros_like(x)


def _mock_get_pca_objects():
    dummy = _DummyPCA()
    return dummy, dummy, None


# ---------------------------------------------------------------------------
# Pytest fixtures (global)
# ---------------------------------------------------------------------------

@pytest.fixture()
def app(monkeypatch) -> Flask:  # type: ignore[override]
    """Flask app with heavy I/O patched out for ultra-fast tests."""
    monkeypatch.setattr(
        "services.accessor.sat.load_satellite_data", _mock_load_satellite_data
    )
    monkeypatch.setattr("services.kernel.handler.infer", _mock_infer)
    monkeypatch.setattr(
        "services.kernel.handler.get_pca_objects", _mock_get_pca_objects
    )
    return create_app()


@pytest.fixture()
def client(app: Flask):  # noqa: D401
    with app.test_client() as c:
        yield c 