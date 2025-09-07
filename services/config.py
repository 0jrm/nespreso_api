# services/config.py
from __future__ import annotations

import os
from dataclasses import dataclass

def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.lower() in {"1", "true", "yes", "on"}

# Add these configuration options for scikit-learn compatibility
SKLEARN_COMPATIBILITY = {
    "suppress_version_warnings": True,
    "use_compatibility_mode": True,
    "fallback_to_dummy_pca": True,
    "max_retry_attempts": 3
}

@dataclass(frozen=True)
class Config:
    # Security / resource controls (0 means "no cap", for compatibility)
    MAX_PROFILES: int = int(os.getenv("NESPRESO_MAX_PROFILES", "0"))

    # Filesystem roots (must be readable; downloading is decoupled)
    SSS_ROOT: str = os.getenv("NESPRESO_SSS_ROOT", "/Net/work/ozavala/DATA/GOFFISH/SSS/SMAP_Global/")
    SST_ROOT: str = os.getenv("NESPRESO_SST_ROOT", "/Net/work/ozavala/DATA/GOFFISH/SST/OISST/")
    AVISO_ROOT: str = os.getenv("NESPRESO_AVISO_ROOT", "/unity/f1/ozavala/DATA/GOFFISH/AVISO/GoM/")
    AVISO_NEW_ROOT: str = os.getenv("NESPRESO_AVISO_NEW_ROOT", "/Net/work/ozavala/DATA/GOFFISH/AVISO/GoM/CMEMS_GLOBAL_PHY_ANFC/")
    AVISO_SWITCH_DATE: str = os.getenv("NESPRESO_AVISO_SWITCH_DATE", "2024-11-01")
    AVISO_ALT_ROOT: str = os.getenv("NESPRESO_AVISO_ALT_ROOT", "")

    # Numerical constants
    BBOX_PADDING_DEG: float = float(os.getenv("NESPRESO_BBOX_PADDING_DEG", "0.5"))
    EXCLUSION_LAT: float = float(os.getenv("NESPRESO_EXCLUSION_LAT", "30.5"))
    EXCLUSION_LON: float = float(os.getenv("NESPRESO_EXCLUSION_LON", "-79.5"))
    KELVIN_OFFSET: float = 273.15  # do not override

    # Model artifacts (override for deployments)
    MODEL_PATH: str = os.getenv("NESPRESO_MODEL_PATH", os.path.join(os.path.dirname(__file__), "../models/ocean_tensorscript.pt"))
    PCA_PATH: str = os.getenv("NESPRESO_PCA_PATH", os.path.join(os.path.dirname(__file__), "../models/pca_stats.pkl"))

    # Grid/mask artifact
    GRID_MASK_PATH: str = os.getenv("NESPRESO_GRID_MASK_PATH", 
                                    "/unity/g2/jmiranda/nespreso_api/data/nespreso_grid_and_mask.pkl")

    # Logging
    LOG_PAYLOAD_SAMPLES: int = int(os.getenv("NESPRESO_LOG_PAYLOAD_SAMPLES", "0"))  # keep at 0 by default to avoid PII/volume

CFG = Config()
