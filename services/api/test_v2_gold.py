"""Gold-path T/S match vs ``thermocline_scorecard._load_ckpt_pred``.

Skipped when checkpoints/caches are not on this host. Run with::

    srun --ntasks=1 --cpus-per-task=8 conda run -n nespreso pytest -q services/api/test_v2_gold.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from services.common.v2_spec import ensure_v2_on_path, load_serve_spec
from services.kernel.v2_cells import cache_row_inputs, get_cell, predict_profiles

_SPEC = Path(
    "/unity/g2/jmiranda/SubsurfaceFields/Data/ISAS20_ARGO/ISAS20_project/reports/heave_da_serve_spec.json"
)
pytestmark = pytest.mark.skipif(
    not _SPEC.is_file(), reason="v2 serve spec not on this host"
)


def _test_idx(cache: dict, ckcfg: dict, n_take: int = 5) -> np.ndarray:
    ensure_v2_on_path()
    from base.split_utils import build_split_indices

    n = int(np.asarray(cache["LAT"]).reshape(-1).shape[0])
    dl = ckcfg.get("data_loader", {}).get("args") or {}
    idx = np.asarray(
        build_split_indices(
            n,
            cache["JULD"],
            dl,
            dataset_tag=cache.get("dataset_tag", "argo_v2"),
            v2_src=ckcfg.get("io", {}).get("v2_src"),
        )["test"],
        dtype=int,
    )
    return idx[:n_take]


def _gold_pred(ckpt: Path, cache: dict, ckcfg: dict, idx: np.ndarray):
    ensure_v2_on_path()
    from scripts.thermocline_scorecard import _load_ckpt_pred

    n_z = int(np.asarray(cache["PRES"]).reshape(-1).size)
    bundle = {
        "cache": cache,
        "cfg": ckcfg,
        "idx": idx,
        "T_true": np.zeros((int(idx.size), n_z), dtype=np.float64),
    }
    return _load_ckpt_pred(ckpt, bundle)


def test_a_crps_s42_matches_gold() -> None:
    spec = load_serve_spec()
    cell = get_cell("A_CRPS", 42)
    idx = _test_idx(cell.cache, cell.ckcfg, 5)
    x = cache_row_inputs("A_CRPS", idx, seed=42)
    got = predict_profiles("A_CRPS", x, seed=42)
    gold = _gold_pred(spec.models["A_CRPS"].ckpts[42], cell.cache, cell.ckcfg, idx)
    assert gold is not None
    t_gold, s_gold = gold
    t_rmse = float(np.sqrt(np.nanmean((got.temperature.T - t_gold) ** 2)))
    s_rmse = float(np.sqrt(np.nanmean((got.salinity.T - s_gold) ** 2)))
    assert t_rmse < 1e-5, t_rmse
    assert s_rmse < 1e-5, s_rmse
    assert "sigma" not in (got.decode,)


def test_heavefast_s42_matches_gold() -> None:
    spec = load_serve_spec()
    cell = get_cell("HeaveFast", 42)
    idx = _test_idx(cell.cache, cell.ckcfg, 5)
    x = cache_row_inputs("HeaveFast", idx, seed=42)
    got = predict_profiles("HeaveFast", x, seed=42)
    gold = _gold_pred(spec.models["HeaveFast"].ckpts[42], cell.cache, cell.ckcfg, idx)
    assert gold is not None
    t_gold, s_gold = gold
    t_rmse = float(np.sqrt(np.nanmean((got.temperature.T - t_gold) ** 2)))
    s_rmse = float(np.sqrt(np.nanmean((got.salinity.T - s_gold) ** 2)))
    assert t_rmse < 1e-5, t_rmse
    assert s_rmse < 1e-5, s_rmse
    assert got.d26 is not None
    import torch

    mu = torch.as_tensor(cell.forward_mu(x), dtype=torch.float32)
    t_idx, _s_idx = cell._heave_loss.physical_ts(mu, torch.as_tensor(idx))
    t_idx_rmse = float(
        np.sqrt(np.nanmean((got.temperature.T - t_idx.detach().cpu().numpy()) ** 2))
    )
    _mld, d26_g, _tr, _sr = cell._heave_loss.decode_ts(mu)
    d26_err = float(np.max(np.abs(got.d26 - d26_g.detach().cpu().numpy())))
    assert t_idx_rmse < 1e-5, t_idx_rmse
    assert d26_err < 1e-4, d26_err


def test_ops_s42_matches_gold() -> None:
    spec = load_serve_spec()
    cell = get_cell("ops", 42)
    idx = _test_idx(cell.cache, cell.ckcfg, 5)
    x = cache_row_inputs("ops", idx, seed=42)
    assert x.shape[1] == 30
    got = predict_profiles("ops", x, seed=42)
    gold = _gold_pred(spec.models["ops"].ckpts[42], cell.cache, cell.ckcfg, idx)
    assert gold is not None
    t_gold, s_gold = gold
    t_rmse = float(np.sqrt(np.nanmean((got.temperature.T - t_gold) ** 2)))
    s_rmse = float(np.sqrt(np.nanmean((got.salinity.T - s_gold) ** 2)))
    assert t_rmse < 1e-5, t_rmse
    assert s_rmse < 1e-5, s_rmse
    assert got.d26 is not None
    import torch

    mu = torch.as_tensor(cell.forward_mu(x), dtype=torch.float32)
    _mld, d26_g, _tr, _sr = cell._heave_loss.decode_ts(mu)
    d26_err = float(np.max(np.abs(got.d26 - d26_g.detach().cpu().numpy())))
    assert d26_err < 1e-4, d26_err
