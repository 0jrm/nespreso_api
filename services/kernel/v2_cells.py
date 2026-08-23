"""Frozen v2 cell inference and gold-path decode. No satellite I/O."""

from __future__ import annotations

import logging
import pickle
import threading
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.base import InconsistentVersionWarning

from services.common.v2_spec import (
    DECODE_HEAVE,
    DECODE_PCA,
    V2ModelSpec,
    V2Registry,
    V2UnavailableError,
    ensure_v2_on_path,
    load_serve_spec,
    resolve_seed,
)

logger = logging.getLogger("ocean")

_cell_lock = threading.Lock()
_cells: dict[tuple[str, int], "V2Cell"] = {}
_caches: dict[str, dict[str, Any]] = {}
_heave_losses: dict[str, Any] = {}


@dataclass(frozen=True)
class V2Decode:
    """Physical T/S (and optional heave depths) for one batch."""

    temperature: np.ndarray  # (n_z, n)
    salinity: np.ndarray
    depth: np.ndarray
    mld: np.ndarray | None
    d26: np.ndarray | None
    model: str
    seed: int
    checkpoint: str
    checkpoint_stem: str
    cache_hash: str
    decode: str
    cache_kind: str | None


def get_registry() -> V2Registry:
    """Load the serve spec once."""
    return load_serve_spec()


def _load_pickle(path: Path) -> dict[str, Any]:
    key = str(path.resolve())
    cached = _caches.get(key)
    if cached is not None:
        return cached
    if not path.is_file():
        raise V2UnavailableError(f"v2 cache missing: {path}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InconsistentVersionWarning)
        with path.open("rb") as f:
            payload = pickle.load(f)
    if not isinstance(payload, dict):
        raise V2UnavailableError(f"v2 cache is not a dict: {path}")
    _caches[key] = payload
    logger.info("Loaded v2 cache %s", path)
    return payload


def _ckpt_cfg(state: dict[str, Any], fallback: dict[str, Any]) -> dict[str, Any]:
    raw = state.get("config", fallback)
    if isinstance(raw, dict):
        return raw
    inner = getattr(raw, "config", None) or getattr(raw, "_config", None)
    return inner if isinstance(inner, dict) else fallback


def _load_json(path: Path) -> dict[str, Any]:
    import json

    return json.loads(path.read_text())


def _take_mu(out: np.ndarray, output_dim: int) -> np.ndarray:
    """Serve μ only. Probabilistic heads emit ``[μ, σ]`` along the last axis."""
    if out.ndim != 2:
        raise ValueError(f"expected (N, D) model output, got {out.shape}")
    if out.shape[-1] == 2 * output_dim:
        return out[:, :output_dim]
    if out.shape[-1] == output_dim:
        return out
    raise ValueError(
        f"output width {out.shape[-1]} is not {output_dim} or {2 * output_dim}"
    )


def _build_heave_loss(cache: dict[str, Any], ckcfg: dict[str, Any]) -> Any:
    """Fit residual PCA on the train split of *this* cache (never mix caches)."""
    ensure_v2_on_path()
    from base.split_utils import build_split_indices
    from model.loss import HeaveResidualFastLoss

    n = int(np.asarray(cache["LAT"]).reshape(-1).shape[0])
    dl = ckcfg.get("data_loader", {}).get("args") or {}
    train_idx = build_split_indices(
        n,
        cache["JULD"],
        dl,
        dataset_tag=cache.get("dataset_tag", "argo_v2"),
        v2_src=ckcfg.get("io", {}).get("v2_src"),
    )["train"]
    loss_cfg = ckcfg.get("loss_config") or {"mode": "heave_residual_fast"}
    return HeaveResidualFastLoss(
        outputs=ckcfg["outputs"],
        device=torch.device("cpu"),
        true_profiles=cache["profiles"],
        pres_levels=cache["PRES"],
        lat=cache["LAT"],
        lon=cache["LON"],
        train_idx=train_idx,
        clim_profiles=cache.get("clim_profiles"),
        loss_config=loss_cfg,
    )


def _heave_loss_for(
    spec: V2ModelSpec, cache: dict[str, Any], ckcfg: dict[str, Any]
) -> Any:
    key = str(spec.cache_path.resolve())
    loss = _heave_losses.get(key)
    if loss is None:
        logger.info("Fitting heave residual PCA on %s", spec.cache_path)
        loss = _build_heave_loss(cache, ckcfg)
        _heave_losses[key] = loss
    return loss


def _physical_ts_basin_mean(
    loss: Any, mu: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unwarp with the cache basin-mean prior (identical on every row).

    These caches have no ``clim_profiles``. The gold-path loss broadcasts
    ``nanmean(true_profiles)``. Do not nearest-neighbor Argo rows.
    """
    from model.heave_fast import canon_to_phys, lerp_along_z, phys_to_canon
    from model.warp import torch_ordered_knots

    n = int(mu.shape[0])
    mld, d26, t_res, s_res = loss.decode_ts(mu)
    z = loss.z.reshape(-1)
    phys, canon = torch_ordered_knots(mld, d26, loss.z_bot)
    z_p = canon_to_phys(z, phys, canon)
    t_clim = loss.T_clim[0:1].expand(n, -1)
    s_clim = loss.S_clim[0:1].expand(n, -1)
    t_prior = lerp_along_z(z_p, z, t_clim)
    s_prior = lerp_along_z(z_p, z, s_clim)
    z_c = phys_to_canon(z, phys, canon)
    t = lerp_along_z(z_c, z, t_prior + t_res)
    s = lerp_along_z(z_c, z, s_prior + s_res)
    return t, s, mld, d26


class V2Cell:
    """One (model, seed) runtime: torch module + paired cache decode buffers."""

    def __init__(self, spec: V2ModelSpec, seed: int):
        if seed not in spec.ckpts:
            raise V2UnavailableError(f"{spec.key} has no checkpoint for seed={seed}")
        ckpt_path = spec.ckpts[seed]
        if not ckpt_path.is_file():
            raise V2UnavailableError(f"checkpoint missing: {ckpt_path}")
        ensure_v2_on_path()
        import model.model as module_arch

        fallback = _load_json(spec.config_path) if spec.config_path.is_file() else {}
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if not isinstance(state, dict):
            raise V2UnavailableError(f"unexpected checkpoint payload: {ckpt_path}")
        ckcfg = _ckpt_cfg(state, fallback)
        arch = ckcfg.get("arch") or fallback.get("arch") or {}
        cls_name = str(arch.get("type") or spec.arch_type)
        args = dict(arch.get("args") or {})
        cls = getattr(module_arch, cls_name)
        model = cls(**args)
        model.load_state_dict(state["state_dict"])
        model.eval()
        self.spec = spec
        self.seed = seed
        self.ckpt_path = ckpt_path
        self.ckcfg = ckcfg
        self.model = model
        self.output_dim = int(args.get("output_dim", spec.output_dim))
        self.cache = _load_pickle(spec.cache_path)
        self.depth = np.asarray(self.cache["PRES"], dtype=np.float32).reshape(-1)
        self._heave_loss: Any | None = None
        if spec.decode == DECODE_HEAVE:
            self._heave_loss = _heave_loss_for(spec, self.cache, ckcfg)

    def forward_mu(self, inputs: np.ndarray) -> np.ndarray:
        """Run the cell and return μ of shape ``(N, output_dim)``."""
        x = np.asarray(inputs, dtype=np.float32)
        expected = int(self.spec.input_dim)
        if x.ndim != 2 or x.shape[1] != expected:
            raise ValueError(f"{self.spec.key} expected (N, {expected}), got {x.shape}")
        with torch.no_grad():
            out = self.model(torch.as_tensor(x, dtype=torch.float32)).cpu().numpy()
        return _take_mu(np.asarray(out), self.output_dim)

    def decode(self, mu: np.ndarray) -> V2Decode:
        """Map μ to native-z T/S using the paired cache (never another cell's PCA)."""
        mu = np.asarray(mu, dtype=np.float32)
        if self.spec.decode == DECODE_PCA:
            pca = self.cache["pca_models"]
            n_t = int(self.ckcfg.get("outputs", {}).get("temperature", 16))
            t = pca["temperature"].inverse_transform(mu[:, :n_t])
            s = pca["salinity"].inverse_transform(mu[:, n_t : n_t + n_t])
            mld = None
            d26 = None
        else:
            if self._heave_loss is None:
                raise V2UnavailableError(f"{self.spec.key} heave loss was not built")
            t_t, s_t, mld_t, d26_t = _physical_ts_basin_mean(
                self._heave_loss, torch.as_tensor(mu, dtype=torch.float32)
            )
            t = t_t.detach().cpu().numpy()
            s = s_t.detach().cpu().numpy()
            mld = mld_t.detach().cpu().numpy()
            d26 = d26_t.detach().cpu().numpy()
        parent = self.ckpt_path.parent.name
        return V2Decode(
            temperature=np.asarray(t, dtype=np.float32).T,
            salinity=np.asarray(s, dtype=np.float32).T,
            depth=self.depth.copy(),
            mld=None if mld is None else np.asarray(mld, dtype=np.float32),
            d26=None if d26 is None else np.asarray(d26, dtype=np.float32),
            model=self.spec.key,
            seed=self.seed,
            checkpoint=str(self.ckpt_path),
            checkpoint_stem=parent,
            cache_hash=self.spec.cache_hash,
            decode=self.spec.decode,
            cache_kind=self.spec.cache_kind,
        )


def get_cell(model: str, seed: int | None = None) -> V2Cell:
    """Return a process-wide cached cell for ``(model, seed)``."""
    spec = get_registry().models[model] if model in get_registry().models else None
    if spec is None:
        raise V2UnavailableError(f"model {model!r} is not served")
    resolved = resolve_seed(model, seed)
    key = (model, resolved)
    cell = _cells.get(key)
    if cell is None:
        with _cell_lock:
            cell = _cells.get(key)
            if cell is None:
                cell = V2Cell(spec, resolved)
                _cells[key] = cell
    return cell


def predict_profiles(
    model: str, inputs: np.ndarray, seed: int | None = None
) -> V2Decode:
    """Infer μ and decode to T/S. ``inputs`` is already the cell's feature matrix."""
    cell = get_cell(model, seed)
    mu = cell.forward_mu(inputs)
    return cell.decode(mu)


def cache_row_inputs(
    model: str, idx: np.ndarray, seed: int | None = None
) -> np.ndarray:
    """Gold-path features: cache rows plus ONI/RONI splice (matches ``_model_inputs``)."""
    cell = get_cell(model, seed)
    cache = cell.cache
    x = np.asarray(cache["inputs"][idx], dtype=np.float32)
    if not cell.spec.enso:
        return x
    ensure_v2_on_path()
    from preproc.enso import inject_enso_columns

    ckcfg = cell.ckcfg
    ip = ckcfg.get("input_params") or cache.get("input_params") or {}
    expected = int(cell.spec.input_dim)
    return inject_enso_columns(
        x,
        cache["JULD"][idx],
        dataset_tag=cache.get("dataset_tag", "argo_v2"),
        input_params=ip,
        n_enc_base=6,
        index_dir=str(cell.spec.index_dir),
        expected_dim=expected,
    )
