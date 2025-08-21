import torch
import os
import copy
import pickle
import logging

_MODEL_PATH = os.path.join(os.path.dirname(__file__), '../../models/ocean_tensorscript.pt')
_pca_stats_path = os.path.join(os.path.dirname(__file__), '../../models/pca_stats.pkl')
_model = None
_pca_temp = None
_pca_sal = None
_input_params = None

logger = logging.getLogger(__name__)

def _get_model():
    global _model
    if _model is None:
        _model = torch.jit.load(_MODEL_PATH, map_location='cpu')
        _model.eval()
    return _model

def infer(batch: torch.Tensor) -> torch.Tensor:
    """
    Run inference on the given batch tensor using the TorchScript model.
    Args:
        batch (torch.Tensor): Input tensor of shape (N, 9)
    Returns:
        torch.Tensor: Output tensor of shape (N, 30)
    """
    model = _get_model()
    with torch.no_grad():
        return model(batch)

def get_pca_objects():
    global _pca_temp, _pca_sal, _input_params
    if _pca_temp is None or _pca_sal is None or _input_params is None:
        try:
            with open(_pca_stats_path, 'rb') as f:
                stats = pickle.load(f)
            _pca_temp = stats['pca_temp']
            _pca_sal = stats['pca_sal']
            _input_params = stats['input_params']
            logger.info("Successfully loaded PCA objects from pickle file")
        except Exception as e:
            logger.error(f"Failed to load PCA objects from pickle: {e}")
            # Try to create dummy PCA objects as fallback
            try:
                from sklearn.decomposition import PCA
                # Create dummy PCA objects with appropriate dimensions
                _pca_temp = PCA(n_components=15)
                _pca_sal = PCA(n_components=15)
                _input_params = {
                    "timecos": True, "timesin": True, "latcos": True, "latsin": True,
                    "loncos": True, "lonsin": True, "sat": True, "sst": True, "sss": True, "ssh": True
                }
                logger.warning("Created dummy PCA objects as fallback - predictions may not be accurate")
            except ImportError as import_error:
                logger.error(f"Could not import sklearn: {import_error}")
                raise RuntimeError("PCA objects could not be loaded and sklearn is not available")
    return _pca_temp, _pca_sal, _input_params 