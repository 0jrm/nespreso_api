"""Pinned registry for frozen NeSPReSO v2 DA profile cells.

Loads ``reports/heave_da_serve_spec.json``. Does not retrain and does not swap
``NESPRESO_MODEL_PATH`` onto these checkpoints.
"""

from __future__ import annotations

import csv
import json
import sys
import types
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from services.config import CFG

SERVED_MODELS: tuple[str, ...] = ("A_CRPS", "HeaveFast", "ops")
A_CRPS_SEEDS: tuple[int, ...] = (42, 43, 44)
R_KIND: str = "dai_sigma_o_after_H"
DECODE_PCA: str = "pca_inverse"
DECODE_HEAVE: str = "heave_residual_fast"


class V2UnavailableError(Exception):
    """Missing satellite or artifact — HTTP 503."""


class V2BadRequestError(Exception):
    """Invalid model/seed/body — HTTP 400."""


@dataclass(frozen=True)
class SigmaOTable:
    """41-layer Dai σ_o after H (not 1 m RMSE, not CRPS-head σ)."""

    zmid_m: np.ndarray
    sigma_t: np.ndarray
    sigma_s: np.ndarray
    regime: str
    seed_label: str
    sigma_t_lc: np.ndarray | None = None
    sigma_s_lc: np.ndarray | None = None
    sigma_t_complement: np.ndarray | None = None
    sigma_s_complement: np.ndarray | None = None


@dataclass(frozen=True)
class V2ModelSpec:
    """One served cell: checkpoint(s) paired with a single cache."""

    key: str
    arch_type: str
    input_dim: int
    output_dim: int
    decode: str
    cache_path: Path
    cache_hash: str
    config_path: Path
    ckpts: Mapping[int, Path]
    default_seed: int
    enso: bool
    n_enc: int
    n_sat: int
    pin_arch_dims: bool
    cache_kind: str | None
    index_dir: Path
    sigma_o_csv: Path
    sigma_o_floors: Mapping[str, float]


@dataclass(frozen=True)
class V2Registry:
    """Resolved paths from the serve spec."""

    repo_root: Path
    code_home: Path
    models: Mapping[str, V2ModelSpec]
    r_kind: str
    lc_box: Mapping[str, list[float]]


def ensure_v2_on_path(code_home: Path | None = None) -> Path:
    """Put ``NeSPReSO2_onTemplate`` on ``sys.path`` without vendoring it.

    Also registers a no-op ``gsw_torch`` if the differentiable GSW package is
    missing. Serving never instantiates ``StericConstraint``; ``model.loss``
    imports it at module load.

    Args:
        code_home: Training-package root. Defaults to ``CFG.V2_CODE_HOME``.

    Returns:
        The directory inserted (or already present) on ``sys.path``.
    """
    home = Path(code_home or CFG.V2_CODE_HOME).resolve()
    s = str(home)
    if s not in sys.path:
        sys.path.insert(0, s)
    if "gsw_torch" not in sys.modules:
        try:
            import gsw_torch  # noqa: F401
        except ImportError:
            sys.modules["gsw_torch"] = types.ModuleType("gsw_torch")
    return home


def _as_repo_path(repo_root: Path, raw: str) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else (repo_root / p)


@lru_cache(maxsize=4)
def load_serve_spec(spec_path: str | None = None) -> V2Registry:
    """Load and resolve the DA serve pin file.

    Args:
        spec_path: JSON path. Defaults to ``CFG.V2_SPEC_PATH``.

    Returns:
        Registry with absolute checkpoint/cache paths.

    Raises:
        FileNotFoundError: Spec file is missing.
        KeyError: A served model is absent from the spec.
    """
    path = Path(spec_path or CFG.V2_SPEC_PATH)
    payload: dict[str, Any] = json.loads(path.read_text())
    repo_root = Path(payload.get("repo_root") or CFG.V2_ROOT).resolve()
    code_home = Path(CFG.V2_CODE_HOME).resolve()
    index_dir = repo_root / str(
        payload.get("enso", {})
        .get("files", {})
        .get("oni", "data/indices/oni.ascii.txt")
    )
    index_dir = index_dir.parent
    sigma_csv = repo_root / str(payload.get("sigma_o_csv", "reports/sigma_o_hycom.csv"))
    floors = payload.get("sigma_o_floors") or {"T_C": 0.05, "S_psu": 0.02}
    models_raw: dict[str, Any] = payload["models"]
    models: dict[str, V2ModelSpec] = {}
    for key in SERVED_MODELS:
        row = models_raw[key]
        if key == "A_CRPS":
            ckpts = {
                int(s): _as_repo_path(repo_root, p) for s, p in row["ckpts"].items()
            }
        else:
            ckpts = {int(row["seed"]): _as_repo_path(repo_root, row["ckpt"])}
        cache_rel = str(row["cache_path"])
        cache_path = _as_repo_path(repo_root, cache_rel)
        if not cache_path.is_file():
            cache_path = _as_repo_path(code_home, cache_rel)
        decode = DECODE_HEAVE if key in ("HeaveFast", "ops") else DECODE_PCA
        models[key] = V2ModelSpec(
            key=key,
            arch_type=str(row["arch_type"]),
            input_dim=int(row["input_dim"]),
            output_dim=int(row["output_dim"]),
            decode=decode,
            cache_path=cache_path,
            cache_hash=str(row.get("cache_hash", "")),
            config_path=_as_repo_path(repo_root, row["config_path"]),
            ckpts=ckpts,
            default_seed=int(row.get("default_seed", row.get("seed", 42))),
            enso=bool(row.get("enso", False)),
            n_enc=int(row["n_enc"]),
            n_sat=int(row["n_sat"]),
            pin_arch_dims=bool(row.get("pin_arch_dims", False)),
            cache_kind=row.get("cache_kind"),
            index_dir=index_dir,
            sigma_o_csv=sigma_csv,
            sigma_o_floors={str(k): float(v) for k, v in dict(floors).items()},
        )
    return V2Registry(
        repo_root=repo_root,
        code_home=code_home,
        models=models,
        r_kind=str(payload.get("r_kind", R_KIND)),
        lc_box=payload.get("lc_box") or {"lat": [24.0, 28.0], "lon": [-88.0, -84.0]},
    )


def resolve_seed(model: str, seed: int | None) -> int:
    """Validate optional ``seed`` query. A_CRPS only; others must omit or use 42.

    Args:
        model: Served model key.
        seed: Requested seed, or None for the cell default.

    Returns:
        Seed to load.

    Raises:
        V2BadRequestError: Model unknown or seed not allowed.
    """
    if model not in SERVED_MODELS:
        raise V2BadRequestError(
            f"Unknown model {model!r}. Served: {list(SERVED_MODELS)}"
        )
    if model == "A_CRPS":
        s = 42 if seed is None else int(seed)
        if s not in A_CRPS_SEEDS:
            raise V2BadRequestError(f"A_CRPS seed must be one of {list(A_CRPS_SEEDS)}")
        return s
    if seed is not None and int(seed) not in (42,):
        raise V2BadRequestError(f"{model} has no seed={seed}; omit seed or use 42")
    return 42


def _float_or_nan(raw: str) -> float:
    if raw is None or str(raw).strip() == "":
        return float("nan")
    return float(raw)


def load_sigma_o(csv_path: Path, model: str, seed: int) -> SigmaOTable:
    """Read Dai σ_o rows. A_CRPS ingest is the 3-seed mean, regime ``all``.

    Args:
        csv_path: ``reports/sigma_o_hycom.csv``.
        model: Served cell name as stored in the CSV.
        seed: Checkpoint seed (ignored for A_CRPS mean ingest R).

    Returns:
        Layer vectors including optional LC / complement.

    Raises:
        FileNotFoundError: CSV missing.
        ValueError: No matching rows.
    """
    del seed  # ingest R for A_CRPS is always the 3-seed mean
    rows: list[dict[str, str]] = []
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty sigma_o csv: {csv_path}")

    def _col(subset: list[dict[str, str]], name: str) -> np.ndarray:
        by_k: dict[int, list[float]] = {}
        for row in subset:
            k = int(row["k"])
            by_k.setdefault(k, []).append(_float_or_nan(row[name]))
        if not by_k:
            return np.array([], dtype=np.float64)
        ks = np.arange(max(by_k) + 1)
        out = np.full(ks.shape, np.nan, dtype=np.float64)
        for k, vals in by_k.items():
            arr = np.asarray(vals, dtype=np.float64)
            out[k] = float(np.nanmean(arr)) if np.isfinite(arr).any() else float("nan")
        return out

    def _subset(
        regime: str, seeds: tuple[str, ...] | None = None
    ) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        for row in rows:
            if row["model"] != model or row["regime"] != regime:
                continue
            if seeds is not None and row["seed"] not in seeds:
                continue
            out.append(row)
        return out

    if model == "A_CRPS":
        seeds = tuple(str(s) for s in A_CRPS_SEEDS)
        seed_label = "mean"
        primary = _subset("all", seeds)
    else:
        seeds = ("42",)
        seed_label = "42"
        primary = _subset("all", seeds)
    if not primary:
        raise ValueError(f"no sigma_o rows for model={model} regime=all")
    zmid = _col(primary, "zmid_m")
    sigma_t = _col(primary, "sigma_T")
    sigma_s = _col(primary, "sigma_S")
    lc = _subset("lc", seeds)
    comp = _subset("complement", seeds)
    return SigmaOTable(
        zmid_m=zmid,
        sigma_t=sigma_t,
        sigma_s=sigma_s,
        regime="all",
        seed_label=seed_label,
        sigma_t_lc=_col(lc, "sigma_T") if lc else None,
        sigma_s_lc=_col(lc, "sigma_S") if lc else None,
        sigma_t_complement=_col(comp, "sigma_T") if comp else None,
        sigma_s_complement=_col(comp, "sigma_S") if comp else None,
    )
