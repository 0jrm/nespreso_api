# services/kernel/handler.py
from __future__ import annotations

import os, pickle, logging, threading
import torch
from services.config import CFG
import warnings
import pickle
from sklearn.base import InconsistentVersionWarning

# Suppress the specific scikit-learn version warning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

logger = logging.getLogger(__name__)

_model_lock = threading.Lock()
_pca_lock = threading.Lock()
_model = None
_pca_temp = None
_pca_sal = None
_input_params = None

def _get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                path = os.path.abspath(CFG.MODEL_PATH)
                if not os.path.isfile(path):
                    raise FileNotFoundError(f"Model file not found: {path}")
                m = torch.jit.load(path, map_location='cpu')
                m.eval()
                _model = m
    return _model

def infer(batch: torch.Tensor) -> torch.Tensor:
    """
    Run inference on the given batch tensor using the TorchScript model.
    batch: (N, 9) float32
    """
    model = _get_model()
    with torch.no_grad():
        return model(batch)

def safe_pickle_load(file_path):
    """
    Safely load pickle files with scikit-learn version compatibility handling.
    """
    try:
        # First try normal loading
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        if "InconsistentVersionWarning" in str(e) or "version" in str(e).lower():
            print(f"WARNING: Scikit-learn version mismatch detected when loading {file_path}")
            print("Attempting to load with compatibility mode...")
            
            try:
                # Try with a more permissive pickle protocol
                import pickle5 as pickle_compat
                with open(file_path, 'rb') as f:
                    return pickle_compat.load(f)
            except ImportError:
                print("pickle5 not available, trying alternative approach...")
                
                # Try to suppress the warning and load anyway
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        with open(file_path, 'rb') as f:
                            return pickle.load(f)
                    except Exception as e2:
                        print(f"Failed to load even with warning suppression: {e2}")
                        raise e
        else:
            raise e

def get_pca_objects():
    """
    Load PCA objects with pickle compatibility handling.
    Suppresses version warnings and provides fallback options.
    """
    global _pca_temp, _pca_sal, _input_params
    if _pca_temp is None or _pca_sal is None or _input_params is None:
        with _pca_lock:
            if _pca_temp is None or _pca_sal is None or _input_params is None:
                path = os.path.abspath(CFG.PCA_PATH)
                try:
                    # Suppress scikit-learn version warnings during loading
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", InconsistentVersionWarning)
                        
                        # Use safer pickle loading
                        stats = safe_pickle_load(path)
                        _pca_temp = stats['pca_temp']
                        _pca_sal = stats['pca_sal']
                        _input_params = stats.get('input_params') or {
                            "timecos": True, "timesin": True, "latcos": True, "latsin": True,
                            "loncos": True, "lonsin": True, "sat": True, "sst": True, "sss": True, "ssh": True
                        }
                        
                        print(f"DEBUG: PCA objects loaded successfully from {path}")
                        print(f"DEBUG: PCA temp type: {type(_pca_temp)}, PCA sal type: {type(_pca_sal)}")
                        
                except Exception as e:
                    logger.error("Failed to load PCA from %s: %s", path, e)
                    print(f"WARNING: Using dummy PCA fallback due to loading error: {e}")
                    print("This may be due to scikit-learn version incompatibility")
                    
                    # Import sklearn here to avoid circular imports
                    try:
                        from sklearn.decomposition import PCA
                        _pca_temp = PCA(n_components=15)
                        _pca_sal = PCA(n_components=15)
                        _input_params = {
                            "timecos": True, "timesin": True, "latcos": True, "latsin": True,
                            "loncos": True, "lonsin": True, "sat": True, "sst": True, "sss": True, "ssh": True
                        }
                        logger.warning("Using dummy PCA fallback; predictions are not meaningful.")
                    except ImportError as import_e:
                        logger.error("Failed to import sklearn: %s", import_e)
                        raise e
    return _pca_temp, _pca_sal, _input_params
